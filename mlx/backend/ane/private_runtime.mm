// Copyright © 2026 Apple Inc.

#include "mlx/backend/ane/private_runtime.h"

#if defined(__APPLE__)

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <objc/runtime.h>

#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/ane/support.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/fast_primitives.h"
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

bool debug_mode() {
  static bool enabled = env::get_var("MLX_ANE_DEBUG", 0) == 1;
  return enabled;
}

bool runtime_verbose_mode() {
  static bool enabled = env::get_var(
                            "MLX_ANE_RUNTIME_VERBOSE",
                            env::get_var("MLX_ANE_VERBOSE", 0)) == 1;
  return enabled;
}

bool dump_mil_enabled() {
  static bool enabled = env::get_var("MLX_ANE_DUMP_MIL", debug_mode() ? 1 : 0) == 1;
  return enabled;
}

void runtime_log(std::string_view message) {
  if (!runtime_verbose_mode()) {
    return;
  }
  std::cerr << "[ane::runtime] " << message << "\n";
}

template <typename Fn>
void runtime_log_lazy(Fn&& builder) {
  if (!runtime_verbose_mode()) {
    return;
  }
  std::cerr << "[ane::runtime] " << builder() << "\n";
}

#if DEBUG_ANE
#define DRUNTIME_LOG(MESSAGE) runtime_log((MESSAGE))
#define DRUNTIME_LOG_LAZY(BUILDER) runtime_log_lazy((BUILDER))
#else
#define DRUNTIME_LOG(MESSAGE) do {} while(0)
#define DRUNTIME_LOG_LAZY(BUILDER) do {} while(0)
#endif

inline bool profile_mode() {
  static bool enabled = env::get_var("MLX_ANE_PROFILE", 0) == 1;
  return enabled;
}

inline bool metadata_fastpath_enabled() {
  static bool enabled = env::get_var("MLX_ANE_METADATA_FASTPATH", 1) == 1;
  return enabled;
}

inline bool strict_input_ready_mode() {
  static bool enabled = env::get_var("MLX_ANE_STRICT_INPUT_READY", 0) == 1;
  return enabled;
}

inline int profile_every_dispatches() {
  static int value = std::max(0, env::get_var("MLX_ANE_PROFILE_EVERY", 0));
  return value;
}

using SteadyClock = std::chrono::steady_clock;

uint64_t now_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          SteadyClock::now().time_since_epoch())
          .count());
}

double ns_to_ms(uint64_t ns) {
  return static_cast<double>(ns) * 1e-6;
}

double bytes_to_mib(uint64_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

struct RuntimeProfileCounters {
  std::atomic<uint64_t> fastpath_calls{0};
  std::atomic<uint64_t> fastpath_ns{0};
  std::atomic<uint64_t> compile_calls{0};
  std::atomic<uint64_t> compile_ns{0};
  std::atomic<uint64_t> compile_model_ns{0};
  std::atomic<uint64_t> compile_alloc_ns{0};
  std::atomic<uint64_t> dispatch_calls{0};
  std::atomic<uint64_t> dispatch_failures{0};
  std::atomic<uint64_t> dispatch_ns{0};
  std::atomic<uint64_t> pre_sync_ns{0};
  std::atomic<uint64_t> input_copy_ns{0};
  std::atomic<uint64_t> input_copy_bytes{0};
  std::atomic<uint64_t> request_build_ns{0};
  std::atomic<uint64_t> evaluate_ns{0};
  std::atomic<uint64_t> output_copy_ns{0};
  std::atomic<uint64_t> output_copy_bytes{0};
  std::atomic<uint64_t> last_print_dispatches{0};
};

RuntimeProfileCounters& runtime_profile() {
  static RuntimeProfileCounters counters;
  return counters;
}

void print_profile_summary(const char* tag) {
  if (!profile_mode()) {
    return;
  }
  auto& p = runtime_profile();
  const auto fastpath_calls = p.fastpath_calls.load(std::memory_order_relaxed);
  const auto fastpath_ns = p.fastpath_ns.load(std::memory_order_relaxed);
  const auto compile_calls = p.compile_calls.load(std::memory_order_relaxed);
  const auto dispatch_calls = p.dispatch_calls.load(std::memory_order_relaxed);
  const auto dispatch_failures = p.dispatch_failures.load(std::memory_order_relaxed);
  const auto compile_ns = p.compile_ns.load(std::memory_order_relaxed);
  const auto compile_model_ns = p.compile_model_ns.load(std::memory_order_relaxed);
  const auto compile_alloc_ns = p.compile_alloc_ns.load(std::memory_order_relaxed);
  const auto dispatch_ns = p.dispatch_ns.load(std::memory_order_relaxed);
  const auto pre_sync_ns = p.pre_sync_ns.load(std::memory_order_relaxed);
  const auto input_copy_ns = p.input_copy_ns.load(std::memory_order_relaxed);
  const auto input_copy_bytes = p.input_copy_bytes.load(std::memory_order_relaxed);
  const auto request_build_ns = p.request_build_ns.load(std::memory_order_relaxed);
  const auto evaluate_ns = p.evaluate_ns.load(std::memory_order_relaxed);
  const auto output_copy_ns = p.output_copy_ns.load(std::memory_order_relaxed);
  const auto output_copy_bytes = p.output_copy_bytes.load(std::memory_order_relaxed);

  std::cerr << "[ane::profile] tag=" << tag << " fastpath_calls=" << fastpath_calls
            << " compile_calls=" << compile_calls
            << " dispatch_calls=" << dispatch_calls
            << " dispatch_failures=" << dispatch_failures
            << " fastpath_ms=" << ns_to_ms(fastpath_ns)
            << " compile_ms=" << ns_to_ms(compile_ns)
            << " compile_model_ms=" << ns_to_ms(compile_model_ns)
            << " compile_alloc_ms=" << ns_to_ms(compile_alloc_ns)
            << " dispatch_ms=" << ns_to_ms(dispatch_ns)
            << " pre_sync_ms=" << ns_to_ms(pre_sync_ns)
            << " input_copy_ms=" << ns_to_ms(input_copy_ns)
            << " input_copy_mib=" << bytes_to_mib(input_copy_bytes)
            << " request_build_ms=" << ns_to_ms(request_build_ns)
            << " evaluate_ms=" << ns_to_ms(evaluate_ns)
            << " output_copy_ms=" << ns_to_ms(output_copy_ns)
            << " output_copy_mib=" << bytes_to_mib(output_copy_bytes);
  if (dispatch_calls > 0) {
    std::cerr << " avg_dispatch_ms="
              << (ns_to_ms(dispatch_ns) / static_cast<double>(dispatch_calls));
  }
  std::cerr << "\n";
}

void maybe_print_profile_periodic() {
  if (!profile_mode()) {
    return;
  }
  const int every = profile_every_dispatches();
  if (every <= 0) {
    return;
  }
  auto& p = runtime_profile();
  const uint64_t current = p.dispatch_calls.load(std::memory_order_relaxed);
  uint64_t last = p.last_print_dispatches.load(std::memory_order_relaxed);
  if (current < static_cast<uint64_t>(every) || current - last < static_cast<uint64_t>(every)) {
    return;
  }
  if (!p.last_print_dispatches.compare_exchange_strong(
          last, current, std::memory_order_relaxed, std::memory_order_relaxed)) {
    return;
  }
  print_profile_summary("periodic");
}

void install_profile_exit_reporter() {
  static bool installed = [] {
    if (profile_mode()) {
      std::atexit([]() { print_profile_summary("final"); });
    }
    return true;
  }();
  (void)installed;
}

static constexpr unsigned int kQoS = 21;
static constexpr int kMLComputeUnitsAll = 2;

NSDictionary* mil_compile_options() {
  return @{
    @"kANEFModelType" : @"kANEFModelMIL",
    @"kANEFNetPlistFilenameKey" : @"model.mil",
  };
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

  if (const auto* rms_prim = dynamic_cast<const fast::RMSNorm*>(&primitive);
      rms_prim != nullptr) {
    if (inputs.size() != 2) {
      reason = "rmsnorm-arity-mismatch";
      return false;
    }
    if (inputs[1].ndim() > 1) {
      reason = "rmsnorm-weight-rank>1";
      return false;
    }
    auto in0_dtype = mil_dtype(inputs[0].dtype());
    auto in1_dtype = mil_dtype(inputs[1].dtype());
    if (in0_dtype == nullptr || in1_dtype == nullptr) {
      reason = "unsupported-input-dtype-token";
      return false;
    }
    int64_t axis_dim = input_shapes_mil[0][3];
    if (axis_dim <= 0) {
      reason = "rmsnorm-invalid-last-dim";
      return false;
    }

    Shape reduce_shape_mil = {
        input_shapes_mil[0][0],
        input_shapes_mil[0][1],
        input_shapes_mil[0][2],
        1,
    };

    const double invd = 1.0 / static_cast<double>(axis_dim);
    const float eps = rms_prim->state().second;

    std::ostringstream os;
    os << std::setprecision(std::numeric_limits<float>::max_digits10);
    os << "program(1.3)\n" << kBuildInfo
       << "{\n"
       << "    func main<ios18>(tensor<" << in0_dtype << ", "
       << shape_to_mil(input_shapes_mil[0]) << "> x, tensor<" << in1_dtype
       << ", " << shape_to_mil(input_shapes_mil[1]) << "> w) {\n"
       << "        string fp32_t = const()[name = string(\"fp32_t\"), val = string(\"fp32\")];\n"
       << "        string out_t = const()[name = string(\"out_t\"), val = string(\""
       << out_dtype << "\")];\n"
       << "        tensor<int32, [1]> rax = const()[name = string(\"rax\"), val = tensor<int32, [1]>([3])];\n"
       << "        bool kd = const()[name = string(\"kd\"), val = bool(true)];\n"
       << "        fp32 invd = const()[name = string(\"invd\"), val = fp32(" << invd
       << ")];\n"
       << "        fp32 eps = const()[name = string(\"eps\"), val = fp32(" << eps
       << ")];\n"
       << "        fp32 nhalf = const()[name = string(\"nhalf\"), val = fp32(-0.5)];\n"
       << "        tensor<fp32, " << shape_to_mil(input_shapes_mil[0])
       << "> x32 = cast(dtype = fp32_t, x = x)[name = string(\"x32\")];\n"
       << "        tensor<fp32, " << shape_to_mil(input_shapes_mil[0])
       << "> sq = mul(x = x32, y = x32)[name = string(\"sq\")];\n"
       << "        tensor<fp32, " << shape_to_mil(reduce_shape_mil)
       << "> ss = reduce_sum(x = sq, axes = rax, keep_dims = kd)[name = string(\"ss\")];\n"
       << "        tensor<fp32, " << shape_to_mil(reduce_shape_mil)
       << "> ss2 = mul(x = ss, y = invd)[name = string(\"ss2\")];\n"
       << "        tensor<fp32, " << shape_to_mil(reduce_shape_mil)
       << "> ss3 = add(x = ss2, y = eps)[name = string(\"ss3\")];\n"
       << "        tensor<fp32, " << shape_to_mil(reduce_shape_mil)
       << "> rrms = pow(x = ss3, y = nhalf)[name = string(\"rrms\")];\n"
       << "        tensor<fp32, " << shape_to_mil(input_shapes_mil[0])
       << "> xr = mul(x = x32, y = rrms)[name = string(\"xr\")];\n"
       << "        tensor<" << out_dtype << ", " << shape_to_mil(input_shapes_mil[0])
       << "> xn = cast(dtype = out_t, x = xr)[name = string(\"xn\")];\n"
       << "        tensor<" << out_dtype << ", " << shape_to_mil(input_shapes_mil[1])
       << "> rw = cast(dtype = out_t, x = w)[name = string(\"rw\")];\n"
       << "        tensor<" << out_dtype << ", " << shape_to_mil(out_shape_mil)
       << "> out = mul(x = xn, y = rw)[name = string(\"ane_op\")];\n"
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
      s.client, @selector(loadModel:options:qos:error:), model, nil, kQoS, &e);
  if (!ok) {
    reason = "load-failed:" + error_to_string(e, "no-error");
    DRUNTIME_LOG(reason);
    return false;
  }
  DRUNTIME_LOG("compile+load ok");
  reason = "ok";
  return true;
}

void unload_model(RuntimeState& s, id model) {
  if (s.client == nil || model == nil) {
    return;
  }
  NSError* e = nil;
  (void)((BOOL(*)(id, SEL, id, id, unsigned int, NSError**))objc_msgSend)(
      s.client, @selector(unloadModel:options:qos:error:), model, nil, kQoS, &e);
}

bool initialize_locked(std::string* reason_out) {
  install_profile_exit_reporter();
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
  NSArray* __strong input_wrappers{nil};
  NSArray* __strong input_indices{nil};
  NSArray* __strong output_wrappers{nil};
  NSArray* __strong output_indices{nil};
  std::vector<IOSurfaceRef> input_surfaces;
  std::vector<IOSurfaceRef> output_surfaces;
  std::vector<size_t> input_nbytes;
  std::vector<size_t> output_nbytes;

  ~Program() {
    if (client != nil && model != nil) {
      NSError* e = nil;
      (void)((BOOL(*)(id, SEL, id, id, unsigned int, NSError**))objc_msgSend)(
          client, @selector(unloadModel:options:qos:error:), model, nil, kQoS, &e);
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

struct CompileProfileScope {
  bool enabled{false};
  uint64_t begin_ns{0};
  uint64_t model_ns{0};
  uint64_t alloc_ns{0};

  CompileProfileScope() : enabled(profile_mode()) {
    if (enabled) {
      begin_ns = now_ns();
    }
  }

  ~CompileProfileScope() {
    if (!enabled) {
      return;
    }
    auto& p = runtime_profile();
    const uint64_t total = now_ns() - begin_ns;
    p.compile_calls.fetch_add(1, std::memory_order_relaxed);
    p.compile_ns.fetch_add(total, std::memory_order_relaxed);
    p.compile_model_ns.fetch_add(model_ns, std::memory_order_relaxed);
    p.compile_alloc_ns.fetch_add(alloc_ns, std::memory_order_relaxed);
  }
};

struct DispatchProfileScope {
  bool enabled{false};
  uint64_t begin_ns{0};
  uint64_t pre_sync_ns{0};
  uint64_t input_copy_ns{0};
  uint64_t input_copy_bytes{0};
  uint64_t request_build_ns{0};
  uint64_t evaluate_ns{0};
  uint64_t output_copy_ns{0};
  uint64_t output_copy_bytes{0};
  bool success{false};

  DispatchProfileScope() : enabled(profile_mode()) {
    if (enabled) {
      begin_ns = now_ns();
    }
  }

  ~DispatchProfileScope() {
    if (!enabled) {
      return;
    }
    auto& p = runtime_profile();
    const uint64_t total = now_ns() - begin_ns;
    p.dispatch_calls.fetch_add(1, std::memory_order_relaxed);
    if (!success) {
      p.dispatch_failures.fetch_add(1, std::memory_order_relaxed);
    }
    p.dispatch_ns.fetch_add(total, std::memory_order_relaxed);
    p.pre_sync_ns.fetch_add(pre_sync_ns, std::memory_order_relaxed);
    p.input_copy_ns.fetch_add(input_copy_ns, std::memory_order_relaxed);
    p.input_copy_bytes.fetch_add(input_copy_bytes, std::memory_order_relaxed);
    p.request_build_ns.fetch_add(request_build_ns, std::memory_order_relaxed);
    p.evaluate_ns.fetch_add(evaluate_ns, std::memory_order_relaxed);
    p.output_copy_ns.fetch_add(output_copy_ns, std::memory_order_relaxed);
    p.output_copy_bytes.fetch_add(output_copy_bytes, std::memory_order_relaxed);
    maybe_print_profile_periodic();
  }
};

bool available(std::string* reason) {
  std::lock_guard<std::mutex> lk(runtime_mutex());
  return initialize_locked(reason);
}

bool dispatch_fastpath(array& arr, std::string* reason) {
  if (!metadata_fastpath_enabled()) {
    return false;
  }

  const auto& inputs = arr.inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].status() == array::Status::unscheduled) {
      if (reason) {
        *reason = "metadata-fastpath-input-unscheduled:" + std::to_string(i);
      }
      return false;
    }
  }

  bool needs_sync = false;
  for (const auto& in : inputs) {
    if (!in.is_available()) {
      needs_sync = true;
      break;
    }
  }
  if (needs_sync) {
      gpu::synchronize(arr.primitive().stream());
      for (size_t i = 0; i < inputs.size(); ++i) {
        if (!inputs[i].is_available()) {
          if (strict_input_ready_mode()) {
            if (reason) {
              *reason = "metadata-fastpath-input-not-available-after-sync:" +
                  std::to_string(i);
            }
            return false;
          }
          DRUNTIME_LOG_LAZY([&]() {
            return "metadata fastpath input[" + std::to_string(i) +
                "] not-marked-available after sync; proceeding";
          });
        }
      }
  }

  auto& primitive = arr.primitive();
  if (!is_metadata_fastpath_primitive(primitive)) {
    return false;
  }
  auto* unary = dynamic_cast<UnaryPrimitive*>(&primitive);
  if (unary == nullptr) {
    if (reason) {
      *reason = "metadata-fastpath-not-unary";
    }
    return false;
  }

  const uint64_t begin_ns = profile_mode() ? now_ns() : 0;
  try {
    unary->eval_cpu(arr.inputs(), arr);
  } catch (const std::exception& e) {
    if (reason) {
      *reason = std::string("metadata-fastpath-failed:") + e.what();
    }
    return false;
  } catch (...) {
    if (reason) {
      *reason = "metadata-fastpath-failed:unknown-exception";
    }
    return false;
  }

  if (profile_mode()) {
    auto& p = runtime_profile();
    p.fastpath_calls.fetch_add(1, std::memory_order_relaxed);
    p.fastpath_ns.fetch_add(now_ns() - begin_ns, std::memory_order_relaxed);
  }
  if (reason) {
    *reason = "metadata-fastpath";
  }
  return true;
}

std::shared_ptr<Program> compile(const array& arr, std::string* reason) {
  std::lock_guard<std::mutex> lk(runtime_mutex());
  CompileProfileScope profile_scope;

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
  uint64_t model_begin_ns = 0;
  if (profile_scope.enabled) {
    model_begin_ns = now_ns();
  }

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
  if (profile_scope.enabled) {
    profile_scope.model_ns += now_ns() - model_begin_ns;
  }
  DRUNTIME_LOG("compile step: model compiled+loaded");

  auto outputs = arr.outputs();
  prog->input_nbytes.reserve(arr.inputs().size());
  prog->output_nbytes.reserve(outputs.size());
  prog->input_surfaces.reserve(arr.inputs().size());
  prog->output_surfaces.reserve(outputs.size());
  uint64_t alloc_begin_ns = 0;
  if (profile_scope.enabled) {
    alloc_begin_ns = now_ns();
  }

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
  if (profile_scope.enabled) {
    profile_scope.alloc_ns += now_ns() - alloc_begin_ns;
  }

  NSMutableArray* input_wrappers =
      [NSMutableArray arrayWithCapacity:prog->input_surfaces.size()];
  NSMutableArray* input_indices =
      [NSMutableArray arrayWithCapacity:prog->input_surfaces.size()];
  for (size_t i = 0; i < prog->input_surfaces.size(); ++i) {
    id wrapped = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
        s.iosurface_cls, @selector(objectWithIOSurface:), prog->input_surfaces[i]);
    if (wrapped == nil) {
      if (reason) {
        *reason = "compile-input-wrap-failed";
      }
      return nullptr;
    }
    [input_wrappers addObject:wrapped];
    [input_indices addObject:@(i)];
  }
  prog->input_wrappers = [input_wrappers copy];
  prog->input_indices = [input_indices copy];

  NSMutableArray* output_wrappers =
      [NSMutableArray arrayWithCapacity:prog->output_surfaces.size()];
  NSMutableArray* output_indices =
      [NSMutableArray arrayWithCapacity:prog->output_surfaces.size()];
  for (size_t i = 0; i < prog->output_surfaces.size(); ++i) {
    id wrapped = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
        s.iosurface_cls, @selector(objectWithIOSurface:), prog->output_surfaces[i]);
    if (wrapped == nil) {
      if (reason) {
        *reason = "compile-output-wrap-failed";
      }
      return nullptr;
    }
    [output_wrappers addObject:wrapped];
    [output_indices addObject:@(i)];
  }
  prog->output_wrappers = [output_wrappers copy];
  prog->output_indices = [output_indices copy];

  if (reason) {
    *reason = "ok";
  }
  DRUNTIME_LOG("compile step: program ready");
  return prog;
}

bool dispatch(Program& program, array& arr, std::string* reason) {
  DispatchProfileScope profile_scope;
  if (program.client == nil || program.model == nil) {
    if (reason) {
      *reason = "program-client-or-model-missing";
    }
    return false;
  }
  const auto& inputs = arr.inputs();
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
  if (
      program.input_wrappers == nil || program.input_indices == nil ||
      program.output_wrappers == nil || program.output_indices == nil) {
    if (reason) {
      *reason = "program-wrappers-missing";
    }
    return false;
  }

  @autoreleasepool {
    DRUNTIME_LOG("dispatch start: staging inputs");
    bool needs_sync = false;
    for (const auto& in : inputs) {
      if (!in.is_available()) {
        needs_sync = true;
        break;
      }
    }
    if (needs_sync) {
      DRUNTIME_LOG("dispatch staging pre-sync begin");
      const uint64_t sync_begin_ns = profile_scope.enabled ? now_ns() : 0;
      gpu::synchronize(arr.primitive().stream());
      if (profile_scope.enabled) {
        profile_scope.pre_sync_ns += now_ns() - sync_begin_ns;
      }
      DRUNTIME_LOG("dispatch staging pre-sync complete");
    }

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
        if (strict_input_ready_mode()) {
          if (reason) {
            *reason = "input-not-available-after-sync:" + std::to_string(i);
          }
          return false;
        }
        DRUNTIME_LOG_LAZY([&]() {
          return "dispatch stage input[" + std::to_string(i) +
              "] not-marked-available after readiness step; proceeding";
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
      const size_t copy_nbytes = std::min(in.nbytes(), program.input_nbytes[i]);
      const uint64_t input_copy_begin_ns = profile_scope.enabled ? now_ns() : 0;
      IOSurfaceLock(surface, 0, nullptr);
      std::memcpy(
          IOSurfaceGetBaseAddress(surface),
          src,
          copy_nbytes);
      IOSurfaceUnlock(surface, 0, nullptr);
      if (profile_scope.enabled) {
        profile_scope.input_copy_ns += now_ns() - input_copy_begin_ns;
        profile_scope.input_copy_bytes += copy_nbytes;
      }
      DRUNTIME_LOG_LAZY(
          [&]() { return "dispatch stage input[" + std::to_string(i) + "] complete"; });
    }
    DRUNTIME_LOG("dispatch memcpy to IOSurfaces complete");

    auto& s = runtime_state();
    DRUNTIME_LOG("dispatch request build begin");
    const uint64_t request_build_begin_ns = profile_scope.enabled ? now_ns() : 0;
    id request_obj = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
        s.request_cls,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        program.input_wrappers,
        program.input_indices,
        program.output_wrappers,
        program.output_indices,
        nil,
        nil,
        @0);
    if (profile_scope.enabled) {
      profile_scope.request_build_ns += now_ns() - request_build_begin_ns;
    }
    if (request_obj == nil) {
      if (reason) {
        *reason = "request-create-failed";
      }
      return false;
    }

    NSError* e = nil;
    DRUNTIME_LOG("dispatch evaluate begin");
    const uint64_t evaluate_begin_ns = profile_scope.enabled ? now_ns() : 0;
    BOOL ok = ((BOOL(*)(id, SEL, id, id, id, unsigned int, NSError**))objc_msgSend)(
        program.client,
        @selector(evaluateWithModel:options:request:qos:error:),
        program.model,
        nil,
        request_obj,
        kQoS,
        &e);
    if (profile_scope.enabled) {
      profile_scope.evaluate_ns += now_ns() - evaluate_begin_ns;
    }
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
    const size_t copy_nbytes = std::min(out.nbytes(), program.output_nbytes[i]);
    const uint64_t output_copy_begin_ns = profile_scope.enabled ? now_ns() : 0;
    std::memcpy(
        dst,
        src,
        copy_nbytes);
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, nullptr);
    if (profile_scope.enabled) {
      profile_scope.output_copy_ns += now_ns() - output_copy_begin_ns;
      profile_scope.output_copy_bytes += copy_nbytes;
    }
    DRUNTIME_LOG_LAZY([&]() {
      return "dispatch output[" + std::to_string(i) + "] complete";
    });
  }
  DRUNTIME_LOG("dispatch output copy complete");

  if (reason) {
    *reason = "ok";
  }
  profile_scope.success = true;
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

bool dispatch_fastpath(array&, std::string* reason) {
  if (reason) {
    *reason = "private-runtime-not-available-on-this-platform";
  }
  return false;
}

bool dispatch(Program&, array&, std::string* reason) {
  if (reason) {
    *reason = "private-runtime-not-available-on-this-platform";
  }
  return false;
}

} // namespace mlx::core::ane::private_runtime

#endif
