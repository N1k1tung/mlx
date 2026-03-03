// Copyright © 2026 Apple Inc.

#include "mlx/backend/ane/private_runtime.h"

#if defined(__APPLE__)

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <objc/runtime.h>

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core::ane::private_runtime {

namespace {

static constexpr const char* kBuildInfo =
    "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
    "{\"coremltools-version\", \"9.0\"}})]\n";

struct RuntimeState {
  bool initialized{false};
  bool available{false};
  std::string reason{"uninitialized"};
  Class client_cls{nil};
  Class model_cls{nil};
  Class request_cls{nil};
  Class iosurface_cls{nil};
  id client{nil};
};

RuntimeState& runtime_state() {
  static RuntimeState state;
  return state;
}

std::mutex& runtime_mutex() {
  static std::mutex mtx;
  return mtx;
}

bool require_probe() {
  static bool enabled = env::get_var("MLX_ANE_REQUIRE_PROBE", 0) == 1;
  return enabled;
}

bool debug_mode() {
  static bool enabled = env::get_var("MLX_ANE_DEBUG", 0) == 1;
  return enabled;
}

bool dump_mil_enabled() {
  static bool enabled = env::get_var("MLX_ANE_DUMP_MIL", debug_mode() ? 1 : 0) == 1;
  return enabled;
}

void runtime_log(std::string_view message) {
    std::cerr << "[ane::runtime] " << message << "\n";
}

template <typename Fn>
void runtime_log_lazy(Fn&& builder) {
    std::cerr << "[ane::runtime] " << builder() << "\n";
}

#if DEBUG
#define DRUNTIME_LOG(MESSAGE) runtime_log((MESSAGE))
#define DRUNTIME_LOG_LAZY(BUILDER) runtime_log_lazy((BUILDER))
#else
#define DRUNTIME_LOG(MESSAGE) do {} while(0)
#define DRUNTIME_LOG_LAZY(BUILDER) do {} while(0)
#endif

static constexpr unsigned int kQoS = 21;
static constexpr int kMLComputeUnitsAll = 2;
static constexpr const char* kDefaultPrewarmModel =
    "/System/Library/PrivateFrameworks/TuriCore.framework/Versions/A/Resources/maml-video-light.mlmodel";

NSDictionary* mil_compile_options() {
  return @{
    @"kANEFModelType" : @"kANEFModelMIL",
    @"kANEFNetPlistFilenameKey" : @"model.mil",
  };
}

bool prewarm_enabled() {
  static bool enabled = env::get_var("MLX_ANE_PREWARM", 1) == 1;
  return enabled;
}

std::string prewarm_model_path() {
  const char* path = std::getenv("MLX_ANE_PREWARM_MODEL");
  if (path != nullptr && path[0] != '\0') {
    return std::string(path);
  }
  return std::string(kDefaultPrewarmModel);
}

std::string error_to_string(NSError* e, std::string_view fallback) {
  if (e == nil) {
    return std::string(fallback);
  }
  std::string message = [[e description] UTF8String];
  NSError* under = e.userInfo[NSUnderlyingErrorKey];
  if (under != nil) {
    message += " | underlying: ";
    message += [[under description] UTF8String];
  }
  return message;
}

void maybe_dump_mil(std::string_view tag, const std::string& mil) {
  if (!dump_mil_enabled()) {
    return;
  }
  std::cerr << "[ane::mil] tag=" << tag << "\n" << mil << "\n";
}

std::string shape_to_mil(const Shape& shape) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      os << ", ";
    }
    os << shape[i];
  }
  os << "]";
  return os.str();
}

bool normalize_shape_for_mil(
    const Shape& in_shape,
    Shape& out_shape,
    std::string& reason) {
  out_shape.clear();
  if (in_shape.size() <= 4) {
    out_shape.reserve(4);
    for (size_t i = in_shape.size(); i < 4; ++i) {
      out_shape.push_back(1);
    }
    for (auto d : in_shape) {
      out_shape.push_back(d);
    }
    return true;
  }

  // ANE MIL paths in this runtime are 4D-oriented. For higher ranks, collapse
  // the leading dimensions into a single batch dimension and preserve the last
  // three dimensions.
  int64_t collapsed = 1;
  for (size_t i = 0; i + 3 < in_shape.size(); ++i) {
    if (in_shape[i] <= 0) {
      reason = "unsupported-nonpositive-dim";
      return false;
    }
    if (collapsed > std::numeric_limits<int64_t>::max() / in_shape[i]) {
      reason = "unsupported-rank-collapse-overflow";
      return false;
    }
    collapsed *= in_shape[i];
  }
  out_shape.reserve(4);
  out_shape.push_back(collapsed);
  out_shape.push_back(in_shape[in_shape.size() - 3]);
  out_shape.push_back(in_shape[in_shape.size() - 2]);
  out_shape.push_back(in_shape[in_shape.size() - 1]);
  return true;
}

bool normalize_axis(int axis, int rank, int& out_axis) {
  if (rank <= 0) {
    return false;
  }
  int ax = axis;
  if (ax < 0) {
    ax += rank;
  }
  if (ax < 0 || ax >= rank) {
    return false;
  }
  out_axis = ax;
  return true;
}

bool axis_to_mil_axis(
    const Shape& original_shape,
    int axis,
    int& mil_axis,
    std::string& reason) {
  int rank = static_cast<int>(original_shape.size());
  int normalized_axis = 0;
  if (!normalize_axis(axis, rank, normalized_axis)) {
    reason = "invalid-axis";
    return false;
  }
  if (rank <= 4) {
    mil_axis = normalized_axis + static_cast<int>(4 - rank);
    return true;
  }
  // Collapsed shape is [prod(leading), d[-3], d[-2], d[-1]].
  if (normalized_axis <= rank - 4) {
    reason = "axis-in-collapsed-leading-dims";
    return false;
  }
  mil_axis = normalized_axis - (rank - 4);
  return true;
}

std::string int32_tensor_literal(const std::array<int32_t, 4>& values) {
  std::ostringstream os;
  os << "tensor<int32, [4]>([" << values[0] << ", " << values[1] << ", " << values[2]
     << ", " << values[3] << "])";
  return os.str();
}

bool dtype_supported(Dtype dtype) {
  return dtype == float16 || dtype == float32;
}

const char* mil_dtype(Dtype dtype) {
  switch (dtype) {
    case float16:
      return "fp16";
    case float32:
      return "fp32";
    default:
      return nullptr;
  }
}

bool io_layout_supported(const array& arr) {
  return arr.flags().row_contiguous;
}

bool build_mil(
    const array& arr,
    std::string& mil,
    std::string& reason) {
  auto& primitive = arr.primitive();
  const auto& inputs = arr.inputs();
  auto outputs = arr.outputs();

  if (outputs.size() != 1) {
    reason = "unsupported-output-arity";
    return false;
  }
  if (!dtype_supported(arr.dtype())) {
    reason = "unsupported-output-dtype";
    return false;
  }
  Shape out_shape_mil;
  if (!normalize_shape_for_mil(arr.shape(), out_shape_mil, reason)) {
    return false;
  }
  if (!io_layout_supported(arr)) {
    reason = "unsupported-output-layout";
    return false;
  }
  std::vector<Shape> input_shapes_mil;
  input_shapes_mil.reserve(inputs.size());
  for (const auto& in : inputs) {
    if (!dtype_supported(in.dtype())) {
      reason = "unsupported-input-dtype";
      return false;
    }
    Shape in_shape_mil;
    if (!normalize_shape_for_mil(in.shape(), in_shape_mil, reason)) {
      return false;
    }
    input_shapes_mil.push_back(std::move(in_shape_mil));
    if (!io_layout_supported(in)) {
      reason = "unsupported-input-layout";
      return false;
    }
  }

  const char* out_dtype = mil_dtype(arr.dtype());
  if (out_dtype == nullptr) {
    reason = "unsupported-output-dtype-token";
    return false;
  }

  auto emit_binary = [&](const char* op_name) -> bool {
    if (inputs.size() != 2) {
      reason = "binary-op-arity-mismatch";
      return false;
    }
    auto in0_dtype = mil_dtype(inputs[0].dtype());
    auto in1_dtype = mil_dtype(inputs[1].dtype());
    if (in0_dtype == nullptr || in1_dtype == nullptr) {
      reason = "unsupported-input-dtype-token";
      return false;
    }
    std::ostringstream os;
    os << "program(1.3)\n" << kBuildInfo
       << "{\n"
       << "    func main<ios18>(tensor<" << in0_dtype << ", "
       << shape_to_mil(input_shapes_mil[0]) << "> x, tensor<" << in1_dtype
       << ", " << shape_to_mil(input_shapes_mil[1]) << "> y) {\n"
       << "        tensor<" << out_dtype << ", " << shape_to_mil(out_shape_mil)
       << "> out = " << op_name << "(x = x, y = y)[name = string(\"ane_op\")];\n"
       << "    } -> (out);\n"
       << "}\n";
    mil = os.str();
    return true;
  };

  auto emit_unary = [&](const char* op_name) -> bool {
    if (inputs.size() != 1) {
      reason = "unary-op-arity-mismatch";
      return false;
    }
    auto in0_dtype = mil_dtype(inputs[0].dtype());
    if (in0_dtype == nullptr) {
      reason = "unsupported-input-dtype-token";
      return false;
    }
    std::ostringstream os;
    os << "program(1.3)\n" << kBuildInfo
       << "{\n"
       << "    func main<ios18>(tensor<" << in0_dtype << ", "
       << shape_to_mil(input_shapes_mil[0]) << "> x) {\n"
       << "        tensor<" << out_dtype << ", " << shape_to_mil(out_shape_mil)
       << "> out = " << op_name << "(x = x)[name = string(\"ane_op\")];\n"
       << "    } -> (out);\n"
       << "}\n";
    mil = os.str();
    return true;
  };

  auto emit_reshape_alias = [&](const char* tag) -> bool {
    if (inputs.size() != 1) {
      reason = std::string(tag) + "-arity-mismatch";
      return false;
    }
    auto in0_dtype = mil_dtype(inputs[0].dtype());
    if (in0_dtype == nullptr) {
      reason = "unsupported-input-dtype-token";
      return false;
    }
    std::array<int32_t, 4> sh = {
        static_cast<int32_t>(out_shape_mil[0]),
        static_cast<int32_t>(out_shape_mil[1]),
        static_cast<int32_t>(out_shape_mil[2]),
        static_cast<int32_t>(out_shape_mil[3]),
    };
    std::ostringstream os;
    os << "program(1.3)\n" << kBuildInfo
       << "{\n"
       << "    func main<ios18>(tensor<" << in0_dtype << ", "
       << shape_to_mil(input_shapes_mil[0]) << "> x) {\n"
       << "        tensor<int32, [4]> sh = const()[name = string(\"sh\"), val = "
       << int32_tensor_literal(sh) << "];\n"
       << "        tensor<" << out_dtype << ", " << shape_to_mil(out_shape_mil)
       << "> out = reshape(shape = sh, x = x)[name = string(\"ane_op\")];\n"
       << "    } -> (out);\n"
       << "}\n";
    mil = os.str();
    return true;
  };

  if (typeid(primitive) == typeid(Add)) {
    return emit_binary("add");
  }
  if (typeid(primitive) == typeid(Subtract)) {
    return emit_binary("sub");
  }
  if (typeid(primitive) == typeid(Multiply)) {
    return emit_binary("mul");
  }
  if (typeid(primitive) == typeid(Divide)) {
    return emit_binary("real_div");
  }
  if (typeid(primitive) == typeid(Sigmoid)) {
    return emit_unary("sigmoid");
  }

  if (
      typeid(primitive) == typeid(Reshape) ||
      typeid(primitive) == typeid(ExpandDims) ||
      typeid(primitive) == typeid(Squeeze) ||
      typeid(primitive) == typeid(Contiguous)) {
    return emit_reshape_alias("reshape-alias");
  }

  if (const auto* cast_prim = dynamic_cast<const AsType*>(&primitive);
      cast_prim != nullptr) {
    if (inputs.size() != 1) {
      reason = "astype-arity-mismatch";
      return false;
    }
    auto in0_dtype = mil_dtype(inputs[0].dtype());
    auto cast_dtype = mil_dtype(cast_prim->state());
    if (in0_dtype == nullptr || cast_dtype == nullptr) {
      reason = "unsupported-astype-dtype";
      return false;
    }
    std::ostringstream os;
    os << "program(1.3)\n" << kBuildInfo
       << "{\n"
       << "    func main<ios18>(tensor<" << in0_dtype << ", "
       << shape_to_mil(input_shapes_mil[0]) << "> x) {\n"
       << "        string cast_t = const()[name = string(\"cast_t\"), val = string(\""
       << cast_dtype << "\")];\n"
       << "        tensor<" << out_dtype << ", " << shape_to_mil(out_shape_mil)
       << "> out = cast(dtype = cast_t, x = x)[name = string(\"ane_op\")];\n"
       << "    } -> (out);\n"
       << "}\n";
    mil = os.str();
    return true;
  }

  if (const auto* transpose_prim = dynamic_cast<const Transpose*>(&primitive);
      transpose_prim != nullptr) {
    if (inputs.size() != 1) {
      reason = "transpose-arity-mismatch";
      return false;
    }
    const auto& in_shape = inputs[0].shape();
    if (in_shape.size() > 4) {
      reason = "transpose-rank>4";
      return false;
    }
    auto axes = transpose_prim->state();
    if (axes.size() != in_shape.size()) {
      reason = "transpose-axes-rank-mismatch";
      return false;
    }
    std::array<int32_t, 4> perm = {0, 1, 2, 3};
    std::array<bool, 4> used = {false, false, false, false};
    size_t shift = 4 - in_shape.size();
    for (size_t i = 0; i < shift; ++i) {
      used[i] = true;
    }
    for (size_t i = 0; i < axes.size(); ++i) {
      int normalized_axis = 0;
      if (!normalize_axis(axes[i], static_cast<int>(in_shape.size()), normalized_axis)) {
        reason = "transpose-invalid-axis";
        return false;
      }
      int mapped = normalized_axis + static_cast<int>(shift);
      if (used[mapped]) {
        reason = "transpose-duplicate-axis";
        return false;
      }
      used[mapped] = true;
      perm[shift + i] = mapped;
    }
    auto in0_dtype = mil_dtype(inputs[0].dtype());
    if (in0_dtype == nullptr) {
      reason = "unsupported-input-dtype-token";
      return false;
    }
    std::ostringstream os;
    os << "program(1.3)\n" << kBuildInfo
       << "{\n"
       << "    func main<ios18>(tensor<" << in0_dtype << ", "
       << shape_to_mil(input_shapes_mil[0]) << "> x) {\n"
       << "        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = "
       << int32_tensor_literal(perm) << "];\n"
       << "        tensor<" << out_dtype << ", " << shape_to_mil(out_shape_mil)
       << "> out = transpose(perm = pm, x = x)[name = string(\"ane_op\")];\n"
       << "    } -> (out);\n"
       << "}\n";
    mil = os.str();
    return true;
  }

  if (const auto* concat_prim = dynamic_cast<const Concatenate*>(&primitive);
      concat_prim != nullptr) {
    if (inputs.size() < 2) {
      reason = "concat-input-count<2";
      return false;
    }
    int axis = concat_prim->state();
    int mil_axis = 0;
    if (!axis_to_mil_axis(inputs[0].shape(), axis, mil_axis, reason)) {
      return false;
    }
    auto in0_dtype = mil_dtype(inputs[0].dtype());
    if (in0_dtype == nullptr) {
      reason = "unsupported-input-dtype-token";
      return false;
    }
    std::ostringstream sig;
    std::ostringstream values;
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto in_dtype = mil_dtype(inputs[i].dtype());
      if (in_dtype == nullptr || std::string_view(in_dtype) != std::string_view(in0_dtype)) {
        reason = "concat-input-dtype-mismatch";
        return false;
      }
      if (i > 0) {
        sig << ", ";
        values << ", ";
      }
      sig << "tensor<" << in_dtype << ", " << shape_to_mil(input_shapes_mil[i])
          << "> x" << i;
      values << "x" << i;
    }
    std::ostringstream os;
    os << "program(1.3)\n" << kBuildInfo
       << "{\n"
       << "    func main<ios18>(" << sig.str() << ") {\n"
       << "        int32 ax = const()[name = string(\"ax\"), val = int32(" << mil_axis
       << ")];\n"
       << "        bool inter = const()[name = string(\"inter\"), val = bool(false)];\n"
       << "        tensor<" << out_dtype << ", " << shape_to_mil(out_shape_mil)
       << "> out = concat(axis = ax, interleave = inter, values = (" << values.str()
       << "))[name = string(\"ane_op\")];\n"
       << "    } -> (out);\n"
       << "}\n";
    mil = os.str();
    return true;
  }

  if (const auto* slice_prim = dynamic_cast<const Slice*>(&primitive);
      slice_prim != nullptr) {
    if (inputs.size() != 1) {
      reason = "slice-arity-mismatch";
      return false;
    }
    const auto& in_shape = inputs[0].shape();
    if (in_shape.size() > 4) {
      reason = "slice-rank>4";
      return false;
    }
    auto [start_indices, end_indices, strides] = slice_prim->state();
    if (
        start_indices.size() != in_shape.size() ||
        end_indices.size() != in_shape.size() ||
        strides.size() != in_shape.size()) {
      reason = "slice-index-rank-mismatch";
      return false;
    }
    std::array<int32_t, 4> begin = {0, 0, 0, 0};
    std::array<int32_t, 4> size = {1, 1, 1, 1};
    size_t shift = 4 - in_shape.size();
    for (size_t i = 0; i < in_shape.size(); ++i) {
      if (strides[i] != 1) {
        reason = "slice-unsupported-stride";
        return false;
      }
      int64_t dim = in_shape[i];
      int64_t s = start_indices[i];
      int64_t e = end_indices[i];
      if (s < 0) {
        s += dim;
      }
      if (e < 0) {
        e += dim;
      }
      s = std::max<int64_t>(0, std::min<int64_t>(s, dim));
      e = std::max<int64_t>(0, std::min<int64_t>(e, dim));
      if (e < s) {
        reason = "slice-invalid-range";
        return false;
      }
      begin[shift + i] = static_cast<int32_t>(s);
      size[shift + i] = static_cast<int32_t>(e - s);
    }
    auto in0_dtype = mil_dtype(inputs[0].dtype());
    if (in0_dtype == nullptr) {
      reason = "unsupported-input-dtype-token";
      return false;
    }
    std::ostringstream os;
    os << "program(1.3)\n" << kBuildInfo
       << "{\n"
       << "    func main<ios18>(tensor<" << in0_dtype << ", "
       << shape_to_mil(input_shapes_mil[0]) << "> x) {\n"
       << "        tensor<int32, [4]> b = const()[name = string(\"b\"), val = "
       << int32_tensor_literal(begin) << "];\n"
       << "        tensor<int32, [4]> sz = const()[name = string(\"sz\"), val = "
       << int32_tensor_literal(size) << "];\n"
       << "        tensor<" << out_dtype << ", " << shape_to_mil(out_shape_mil)
       << "> out = slice_by_size(x = x, begin = b, size = sz)[name = string(\"ane_op\")];\n"
       << "    } -> (out);\n"
       << "}\n";
    mil = os.str();
    return true;
  }

  if (typeid(primitive) == typeid(Matmul)) {
    if (inputs.size() != 2) {
      reason = "matmul-arity-mismatch";
      return false;
    }
    auto in0_dtype = mil_dtype(inputs[0].dtype());
    auto in1_dtype = mil_dtype(inputs[1].dtype());
    if (in0_dtype == nullptr || in1_dtype == nullptr) {
      reason = "unsupported-input-dtype-token";
      return false;
    }
    std::ostringstream os;
    os << "program(1.3)\n" << kBuildInfo
       << "{\n"
       << "    func main<ios18>(tensor<" << in0_dtype << ", "
       << shape_to_mil(input_shapes_mil[0]) << "> x, tensor<" << in1_dtype
       << ", " << shape_to_mil(input_shapes_mil[1]) << "> y) {\n"
       << "        bool tx = const()[name = string(\"tx\"), val = bool(false)];\n"
       << "        bool ty = const()[name = string(\"ty\"), val = bool(false)];\n"
       << "        tensor<" << out_dtype << ", " << shape_to_mil(out_shape_mil)
       << "> out = matmul(transpose_x = tx, transpose_y = ty, x = x, y = y)"
       << "[name = string(\"ane_op\")];\n"
       << "    } -> (out);\n"
       << "}\n";
    mil = os.str();
    return true;
  }
  if (typeid(primitive) == typeid(Softmax)) {
    if (inputs.size() != 1) {
      reason = "softmax-arity-mismatch";
      return false;
    }
    auto in0_dtype = mil_dtype(inputs[0].dtype());
    if (in0_dtype == nullptr) {
      reason = "unsupported-input-dtype-token";
      return false;
    }
    std::ostringstream os;
    os << "program(1.3)\n" << kBuildInfo
       << "{\n"
       << "    func main<ios18>(tensor<" << in0_dtype << ", "
       << shape_to_mil(input_shapes_mil[0]) << "> x) {\n"
       << "        int32 ax = const()[name = string(\"ax\"), val = int32(-1)];\n"
       << "        tensor<" << out_dtype << ", " << shape_to_mil(out_shape_mil)
       << "> out = softmax(axis = ax, x = x)[name = string(\"ane_op\")];\n"
       << "    } -> (out);\n"
       << "}\n";
    mil = os.str();
    return true;
  }

  reason = "unsupported-primitive";
  return false;
}

IOSurfaceRef create_surface(size_t bytes) {
  size_t alloc_size = std::max<size_t>(bytes, 1);
  NSDictionary* props = @{
    (id)kIOSurfaceWidth : @(alloc_size),
    (id)kIOSurfaceHeight : @1,
    (id)kIOSurfaceBytesPerElement : @1,
    (id)kIOSurfaceBytesPerRow : @(alloc_size),
    (id)kIOSurfaceAllocSize : @(alloc_size),
    (id)kIOSurfacePixelFormat : @0,
  };
  return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

NSString* create_mil_model_dir(const std::string& mil, std::string& reason) {
  NSString* dir = [NSTemporaryDirectory()
      stringByAppendingPathComponent:[NSString
          stringWithFormat:@"mlx_ane_%@", [NSUUID UUID].UUIDString]];
  NSFileManager* fm = [NSFileManager defaultManager];
  NSError* e = nil;
  if (![fm createDirectoryAtPath:dir
      withIntermediateDirectories:YES
                       attributes:nil
                            error:&e]) {
    reason = "model-dir-create-failed:" + error_to_string(e, "no-error");
    return nil;
  }

  NSString* mil_path = [dir stringByAppendingPathComponent:@"model.mil"];
  NSData* mil_data = [NSData dataWithBytes:mil.data() length:mil.size()];
  if (![mil_data writeToFile:mil_path options:NSDataWritingAtomic error:&e]) {
    reason = "model-mil-write-failed:" + error_to_string(e, "no-error");
    [fm removeItemAtPath:dir error:nil];
    return nil;
  }
  return dir;
}

id create_model(RuntimeState& s, NSString* model_dir, std::string& reason) {
  NSURL* model_url = [NSURL fileURLWithPath:model_dir];
  NSString* key = [NSString stringWithFormat:@"mlx-ane-%@", [NSUUID UUID].UUIDString];
  id model = ((id(*)(Class, SEL, id, id))objc_msgSend)(
      s.model_cls, @selector(modelAtURL:key:), model_url, key);
  if (model == nil) {
    reason = "model-create-failed";
  }
  return model;
}

void unload_model(RuntimeState& s, id model);

bool compile_load_model_with_options(
    RuntimeState& s,
    id model,
    id options,
    std::string& reason);

bool compile_load_model(RuntimeState& s, id model, std::string& reason) {
  return compile_load_model_with_options(s, model, mil_compile_options(), reason);
}

bool compile_load_model_with_options(
    RuntimeState& s,
    id model,
    id options,
    std::string& reason) {
  NSError* e = nil;
  BOOL ok = ((BOOL(*)(id, SEL, id, id, unsigned int, NSError**))objc_msgSend)(
      s.client,
      @selector(compileModel:options:qos:error:),
      model,
      options,
      kQoS,
      &e);
  if (!ok) {
    reason = "compile-failed:" + error_to_string(e, "no-error");
    DRUNTIME_LOG(reason);
    return false;
  }

  e = nil;
  ok = ((BOOL(*)(id, SEL, id, id, unsigned int, NSError**))objc_msgSend)(
      s.client, @selector(loadModel:options:qos:error:), model, @{}, kQoS, &e);
  if (!ok) {
    reason = "load-failed:" + error_to_string(e, "no-error");
    DRUNTIME_LOG(reason);
    return false;
  }
  DRUNTIME_LOG("compile+load ok");
  reason = "ok";
  return true;
}

bool prewarm_client(RuntimeState& s, std::string& reason) {
  if (!prewarm_enabled()) {
    reason = "prewarm-disabled";
    return true;
  }

  std::string model_path = prewarm_model_path();
  NSString* src_path = [NSString stringWithUTF8String:model_path.c_str()];
  if (![[NSFileManager defaultManager] fileExistsAtPath:src_path]) {
    reason = "prewarm-model-not-found:" + model_path;
    return false;
  }

  NSURL* src_url = [NSURL fileURLWithPath:src_path];
  NSURL* compiled_url = src_url;
  bool compiled_tmp = false;

  if (![[src_path pathExtension] isEqualToString:@"mlmodelc"]) {
    Class MLModelCls = NSClassFromString(@"MLModel");
    if (MLModelCls == nil) {
      reason = "prewarm-coreml-class-missing";
      return false;
    }
    NSError* e = nil;
    compiled_url = ((id(*)(Class, SEL, id, NSError**))objc_msgSend)(
        MLModelCls, @selector(compileModelAtURL:error:), src_url, &e);
    if (compiled_url == nil) {
      reason = "prewarm-coreml-compile-failed:" + error_to_string(e, "no-error");
      return false;
    }
    compiled_tmp = true;
  }

  Class MLModelCls = NSClassFromString(@"MLModel");
  Class MLModelCfgCls = NSClassFromString(@"MLModelConfiguration");
  if (MLModelCls == nil || MLModelCfgCls == nil) {
    if (compiled_tmp) {
      [[NSFileManager defaultManager] removeItemAtPath:compiled_url.path error:nil];
    }
    reason = "prewarm-coreml-class-missing";
    return false;
  }

  id cfg = ((id(*)(Class, SEL))objc_msgSend)(MLModelCfgCls, @selector(new));
  if (cfg != nil && [cfg respondsToSelector:@selector(setComputeUnits:)]) {
    ((void(*)(id, SEL, NSInteger))objc_msgSend)(
        cfg, @selector(setComputeUnits:), kMLComputeUnitsAll);
  }

  NSError* e = nil;
  id loaded = ((id(*)(Class, SEL, id, id, NSError**))objc_msgSend)(
      MLModelCls, @selector(modelWithContentsOfURL:configuration:error:), compiled_url, cfg, &e);
  if (loaded == nil) {
    if (compiled_tmp) {
      [[NSFileManager defaultManager] removeItemAtPath:compiled_url.path error:nil];
    }
    reason = "prewarm-coreml-load-failed:" + error_to_string(e, "no-error");
    return false;
  }
  (void)loaded;

  NSString* key = @"mlx-ane-prewarm";
  id model = ((id(*)(Class, SEL, id, id))objc_msgSend)(
      s.model_cls, @selector(modelAtURL:key:), compiled_url, key);
  if (model == nil) {
    if (compiled_tmp) {
      [[NSFileManager defaultManager] removeItemAtPath:compiled_url.path error:nil];
    }
    reason = "prewarm-ane-model-create-failed";
    return false;
  }

  std::string local_reason;
  bool ok = compile_load_model_with_options(s, model, nil, local_reason);
  if (ok) {
    unload_model(s, model);
  }
  if (compiled_tmp) {
    [[NSFileManager defaultManager] removeItemAtPath:compiled_url.path error:nil];
  }
  reason = ok ? std::string("ok")
              : std::string("prewarm-ane-compile-load-failed:") + local_reason;
  DRUNTIME_LOG_LAZY([&]() { return std::string("prewarm result: ") + reason; });
  return ok;
}

void unload_model(RuntimeState& s, id model) {
  if (s.client == nil || model == nil) {
    return;
  }
  NSError* e = nil;
  (void)((BOOL(*)(id, SEL, id, id, unsigned int, NSError**))objc_msgSend)(
      s.client, @selector(unloadModel:options:qos:error:), model, @{}, kQoS, &e);
}

bool compile_probe(RuntimeState& s, std::string& reason) {
  std::string probe_mil =
      "program(1.3)\n"
      "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
      "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
      "{\"coremltools-version\", \"9.0\"}})]\n"
      "{\n"
      "    func main<ios18>(tensor<fp16, [1, 1, 1, 4]> a, tensor<fp16, [1, 1, 1, 4]> b) {\n"
      "        tensor<fp16, [1, 1, 1, 4]> out = add(x = a, y = b)[name = string(\"probe\")];\n"
      "    } -> (out);\n"
      "}\n";
  maybe_dump_mil("probe", probe_mil);
  std::string local_reason;
  NSString* model_dir = create_mil_model_dir(probe_mil, local_reason);
  if (model_dir == nil) {
    reason = local_reason;
    return false;
  }
  id model = create_model(s, model_dir, local_reason);
  if (model == nil) {
    reason = local_reason;
    [[NSFileManager defaultManager] removeItemAtPath:model_dir error:nil];
    return false;
  }
  bool ok = compile_load_model(s, model, local_reason);
  if (ok) {
    unload_model(s, model);
  }
  [[NSFileManager defaultManager] removeItemAtPath:model_dir error:nil];
  reason = local_reason;
  return ok;
}

bool initialize_locked(std::string* reason_out) {
  auto& s = runtime_state();
  if (s.initialized) {
    if (reason_out) {
      *reason_out = s.reason;
    }
    return s.available;
  }
  s.initialized = true;

  void* handle = dlopen(
      "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
      RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    s.reason = "ane-framework-dlopen-failed";
    if (reason_out) {
      *reason_out = s.reason;
    }
    return false;
  }

  s.client_cls = NSClassFromString(@"_ANEClient");
  s.model_cls = NSClassFromString(@"_ANEModel");
  s.request_cls = NSClassFromString(@"_ANERequest");
  s.iosurface_cls = NSClassFromString(@"_ANEIOSurfaceObject");
  if (
      s.client_cls == nil || s.model_cls == nil || s.request_cls == nil ||
      s.iosurface_cls == nil) {
    s.reason = "ane-required-classes-missing";
    if (reason_out) {
      *reason_out = s.reason;
    }
    return false;
  }

  s.client = ((id(*)(Class, SEL))objc_msgSend)(
      s.client_cls, @selector(sharedConnection));
  if (s.client == nil) {
    s.reason = "ane-client-shared-connection-failed";
    if (reason_out) {
      *reason_out = s.reason;
    }
    return false;
  }

  std::string prewarm_reason;
  if (!prewarm_client(s, prewarm_reason)) {
    s.reason = "ane-runtime-prewarm-failed:" + prewarm_reason;
    if (reason_out) {
      *reason_out = s.reason;
    }
    return false;
  }
  DRUNTIME_LOG("prewarm complete");

  if (require_probe()) {
    std::string probe_reason;
    if (!compile_probe(s, probe_reason)) {
      s.reason = "ane-runtime-probe-failed:" + probe_reason;
      if (reason_out) {
        *reason_out = s.reason;
      }
      return false;
    }
  }

  s.available = true;
  s.reason = "ok";
  DRUNTIME_LOG("runtime initialized: available");
  if (reason_out) {
    *reason_out = s.reason;
  }
  return true;
}

} // namespace

struct Program {
  id __strong client{nil};
  id __strong model{nil};
  NSString* __strong model_dir{nil};
  std::vector<IOSurfaceRef> input_surfaces;
  std::vector<IOSurfaceRef> output_surfaces;
  std::vector<size_t> input_nbytes;
  std::vector<size_t> output_nbytes;

  ~Program() {
    if (client != nil && model != nil) {
      NSError* e = nil;
      (void)((BOOL(*)(id, SEL, id, id, unsigned int, NSError**))objc_msgSend)(
          client, @selector(unloadModel:options:qos:error:), model, @{}, kQoS, &e);
    }
    for (auto s : input_surfaces) {
      if (s != nullptr) {
        CFRelease(s);
      }
    }
    for (auto s : output_surfaces) {
      if (s != nullptr) {
        CFRelease(s);
      }
    }
    if (model_dir != nil) {
      [[NSFileManager defaultManager] removeItemAtPath:model_dir error:nil];
    }
  }
};

bool available(std::string* reason) {
  std::lock_guard<std::mutex> lk(runtime_mutex());
  return initialize_locked(reason);
}

std::shared_ptr<Program> compile(const array& arr, std::string* reason) {
  std::lock_guard<std::mutex> lk(runtime_mutex());

  std::string init_reason;
  if (!initialize_locked(&init_reason)) {
    if (reason) {
      *reason = init_reason;
    }
    return nullptr;
  }

  std::string mil;
  std::string mil_reason;
  if (!build_mil(arr, mil, mil_reason)) {
    if (reason) {
      *reason = mil_reason;
    }
    return nullptr;
  }
  maybe_dump_mil(arr.primitive().name(), mil);

  auto prog = std::make_shared<Program>();
  auto& s = runtime_state();
  prog->client = s.client;

  std::string model_reason;
  NSString* model_dir = create_mil_model_dir(mil, model_reason);
  if (model_dir == nil) {
    if (reason) {
      *reason = model_reason + std::string(":op=") + arr.primitive().name();
    }
    return nullptr;
  }
  prog->model_dir = [model_dir copy];

  prog->model = create_model(s, model_dir, model_reason);
  if (prog->model == nil) {
    if (reason) {
      *reason = model_reason + std::string(":op=") + arr.primitive().name();
    }
    return nullptr;
  }
  if (!compile_load_model(s, prog->model, model_reason)) {
    if (reason) {
      *reason = model_reason + std::string(":op=") + arr.primitive().name();
    }
    return nullptr;
  }
  DRUNTIME_LOG("compile step: model compiled+loaded");

  auto outputs = arr.outputs();
  prog->input_nbytes.reserve(arr.inputs().size());
  prog->output_nbytes.reserve(outputs.size());
  prog->input_surfaces.reserve(arr.inputs().size());
  prog->output_surfaces.reserve(outputs.size());

  for (const auto& in : arr.inputs()) {
    prog->input_nbytes.push_back(in.nbytes());
    auto surface = create_surface(in.nbytes());
    if (surface == nullptr) {
      if (reason) {
        *reason = "input-surface-create-failed";
      }
      return nullptr;
    }
    prog->input_surfaces.push_back(surface);
  }
  DRUNTIME_LOG("compile step: input IOSurfaces allocated");
  for (const auto& out : outputs) {
    prog->output_nbytes.push_back(out.nbytes());
    auto surface = create_surface(out.nbytes());
    if (surface == nullptr) {
      if (reason) {
        *reason = "output-surface-create-failed";
      }
      return nullptr;
    }
    prog->output_surfaces.push_back(surface);
  }
  DRUNTIME_LOG("compile step: output IOSurfaces allocated");

  if (reason) {
    *reason = "ok";
  }
  DRUNTIME_LOG("compile step: program ready");
  return prog;
}

bool dispatch(Program& program, array& arr, std::string* reason) {
  if (program.client == nil || program.model == nil) {
    if (reason) {
      *reason = "program-client-or-model-missing";
    }
    return false;
  }
  auto inputs = arr.inputs();
  auto outputs = arr.outputs();
  if (inputs.size() != program.input_surfaces.size()) {
    if (reason) {
      *reason = "input-count-mismatch";
    }
    return false;
  }
  if (outputs.size() != program.output_surfaces.size()) {
    if (reason) {
      *reason = "output-count-mismatch";
    }
    return false;
  }

  @autoreleasepool {
    DRUNTIME_LOG("dispatch start: staging inputs");
    DRUNTIME_LOG("dispatch staging pre-sync begin");
    gpu::synchronize(arr.primitive().stream());
    DRUNTIME_LOG("dispatch staging pre-sync complete");
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto& in = inputs[i];
      DRUNTIME_LOG_LAZY([&]() {
        return "dispatch stage input[" + std::to_string(i) +
            "] begin status=" + std::to_string(static_cast<int>(in.status()));
      });
      if (in.status() == array::Status::unscheduled) {
        if (reason) {
          *reason = "input-unscheduled-at-dispatch:" + std::to_string(i);
        }
        return false;
      }
      if (!in.is_available()) {
        DRUNTIME_LOG_LAZY([&]() {
          return "dispatch stage input[" + std::to_string(i) +
              "] not-marked-available after pre-sync; proceeding";
        });
      }
      auto* src = in.data<char>();
      if (src == nullptr) {
        if (reason) {
          *reason = "input-data-null:" + std::to_string(i);
        }
        return false;
      }
      auto surface = program.input_surfaces[i];
      IOSurfaceLock(surface, 0, nullptr);
      std::memcpy(
          IOSurfaceGetBaseAddress(surface),
          src,
          std::min(in.nbytes(), program.input_nbytes[i]));
      IOSurfaceUnlock(surface, 0, nullptr);
      DRUNTIME_LOG_LAZY(
          [&]() { return "dispatch stage input[" + std::to_string(i) + "] complete"; });
    }
    DRUNTIME_LOG("dispatch memcpy to IOSurfaces complete");

    auto& s = runtime_state();
    NSMutableArray* input_objs =
        [NSMutableArray arrayWithCapacity:program.input_surfaces.size()];
    NSMutableArray* input_indices =
        [NSMutableArray arrayWithCapacity:program.input_surfaces.size()];
    for (size_t i = 0; i < program.input_surfaces.size(); ++i) {
      id wrapped = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
          s.iosurface_cls, @selector(objectWithIOSurface:), program.input_surfaces[i]);
      if (wrapped == nil) {
        if (reason) {
          *reason = "request-input-wrap-failed";
        }
        return false;
      }
      [input_objs addObject:wrapped];
      [input_indices addObject:@(i)];
    }

    NSMutableArray* output_objs =
        [NSMutableArray arrayWithCapacity:program.output_surfaces.size()];
    NSMutableArray* output_indices =
        [NSMutableArray arrayWithCapacity:program.output_surfaces.size()];
    for (size_t i = 0; i < program.output_surfaces.size(); ++i) {
      id wrapped = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
          s.iosurface_cls, @selector(objectWithIOSurface:), program.output_surfaces[i]);
      if (wrapped == nil) {
        if (reason) {
          *reason = "request-output-wrap-failed";
        }
        return false;
      }
      [output_objs addObject:wrapped];
      [output_indices addObject:@(i)];
    }

    DRUNTIME_LOG("dispatch request build begin");
    id request_obj = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
        s.request_cls,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        input_objs,
        input_indices,
        output_objs,
        output_indices,
        nil,
        nil,
        @0);
    if (request_obj == nil) {
      if (reason) {
        *reason = "request-create-failed";
      }
      return false;
    }

    NSError* e = nil;
    DRUNTIME_LOG("dispatch evaluate begin");
    BOOL ok = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError**))objc_msgSend)(
        program.client,
        @selector(evaluateWithModel:options:request:qos:error:),
        program.model,
        @{},
        request_obj,
        kQoS,
        &e);
    DRUNTIME_LOG(ok ? "dispatch evaluate end ok=1" : "dispatch evaluate end ok=0");
    if (!ok) {
      if (reason) {
        *reason = e ? std::string([[e description] UTF8String])
                    : std::string("evaluate-failed-no-error");
      }
      return false;
    }
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    auto& out = outputs[i];
    DRUNTIME_LOG_LAZY(
        [&]() { return "dispatch output[" + std::to_string(i) + "] begin"; });
    auto data_ref = out.data_shared_ptr();
    bool need_alloc = (data_ref == nullptr || data_ref->buffer.ptr() == nullptr);
    if (!need_alloc) {
      need_alloc = (out.data<char>() == nullptr);
    }
    if (need_alloc) {
      out.set_data(allocator::malloc(out.nbytes()));
    }
    auto* dst = out.data<char>();
    if (dst == nullptr) {
      if (reason) {
        *reason = "output-data-null:" + std::to_string(i);
      }
      return false;
    }
    auto surface = program.output_surfaces[i];
    IOSurfaceLock(surface, kIOSurfaceLockReadOnly, nullptr);
    auto* src = IOSurfaceGetBaseAddress(surface);
    if (src == nullptr) {
      IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, nullptr);
      if (reason) {
        *reason = "output-surface-base-null:" + std::to_string(i);
      }
      return false;
    }
    std::memcpy(
        dst,
        src,
        std::min(out.nbytes(), program.output_nbytes[i]));
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, nullptr);
    DRUNTIME_LOG_LAZY([&]() {
      return "dispatch output[" + std::to_string(i) + "] complete";
    });
  }
  DRUNTIME_LOG("dispatch output copy complete");

  if (reason) {
    *reason = "ok";
  }
  return true;
}

#undef DRUNTIME_LOG
#undef DRUNTIME_LOG_LAZY

} // namespace mlx::core::ane::private_runtime

#else

namespace mlx::core::ane::private_runtime {

struct Program {};

bool available(std::string* reason) {
  if (reason) {
    *reason = "private-runtime-not-available-on-this-platform";
  }
  return false;
}

std::shared_ptr<Program> compile(const array&, std::string* reason) {
  if (reason) {
    *reason = "private-runtime-not-available-on-this-platform";
  }
  return nullptr;
}

bool dispatch(Program&, array&, std::string* reason) {
  if (reason) {
    *reason = "private-runtime-not-available-on-this-platform";
  }
  return false;
}

} // namespace mlx::core::ane::private_runtime

#endif
