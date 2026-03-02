// Copyright © 2026 Apple Inc.

#include "mlx/backend/ane/private_runtime.h"

#if defined(__APPLE__)

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <objc/runtime.h>

#include <dlfcn.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/primitives.h"

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
  Class desc_cls{nil};
  Class model_cls{nil};
  Class request_cls{nil};
  Class iosurface_cls{nil};
};

RuntimeState& runtime_state() {
  static RuntimeState state;
  return state;
}

std::mutex& runtime_mutex() {
  static std::mutex mtx;
  return mtx;
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

bool dtype_supported(Dtype dtype) {
  // Keep runtime path conservative and deterministic for now.
  return dtype == float16;
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
  if (!io_layout_supported(arr)) {
    reason = "unsupported-output-layout";
    return false;
  }
  for (const auto& in : inputs) {
    if (!dtype_supported(in.dtype())) {
      reason = "unsupported-input-dtype";
      return false;
    }
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
       << shape_to_mil(inputs[0].shape()) << "> x, tensor<" << in1_dtype
       << ", " << shape_to_mil(inputs[1].shape()) << "> y) {\n"
       << "        tensor<" << out_dtype << ", " << shape_to_mil(arr.shape())
       << "> out = " << op_name
       << "(x=x, y=y)[name = string(\"ane_op\")];\n"
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
       << shape_to_mil(inputs[0].shape()) << "> x, tensor<" << in1_dtype
       << ", " << shape_to_mil(inputs[1].shape()) << "> y) {\n"
       << "        bool tx = const()[name = string(\"tx\"), val = bool(false)];\n"
       << "        bool ty = const()[name = string(\"ty\"), val = bool(false)];\n"
       << "        tensor<" << out_dtype << ", " << shape_to_mil(arr.shape())
       << "> out = matmul(transpose_x=tx, transpose_y=ty, x=x, y=y)"
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
       << shape_to_mil(inputs[0].shape()) << "> x) {\n"
       << "        int32 ax = const()[name = string(\"ax\"), val = int32(-1)];\n"
       << "        tensor<" << out_dtype << ", " << shape_to_mil(arr.shape())
       << "> out = softmax(axis=ax, x=x)[name = string(\"ane_op\")];\n"
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

bool compile_probe(RuntimeState& s, std::string& reason) {
  NSString* mil =
      @"program(1.3)\n"
      "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
      "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
      "{\"coremltools-version\", \"9.0\"}})]\n"
      "{\n"
      "    func main<ios18>(tensor<fp16, [1, 1, 1, 4]> a, tensor<fp16, [1, 1, 1, 4]> b) {\n"
      "        tensor<fp16, [1, 1, 1, 4]> out = add(x=a, y=b)[name = string(\"probe\")];\n"
      "    } -> (out);\n"
      "}\n";
  NSData* mil_data = [mil dataUsingEncoding:NSUTF8StringEncoding];
  id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
      s.desc_cls, @selector(modelWithMILText:weights:optionsPlist:), mil_data, @{}, nil);
  if (desc == nil) {
    reason = "probe-descriptor-create-failed";
    return false;
  }
  id model = ((id(*)(Class, SEL, id))objc_msgSend)(
      s.model_cls, @selector(inMemoryModelWithDescriptor:), desc);
  if (model == nil) {
    reason = "probe-model-create-failed";
    return false;
  }
  (void)((id(*)(id, SEL))objc_msgSend)(model, @selector(saveModelFiles));
  NSError* e = nil;
  BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
      model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
  if (!ok) {
    reason = e ? std::string([[e description] UTF8String])
               : std::string("probe-compile-failed-no-error");
    return false;
  }
  e = nil;
  ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
      model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
  if (!ok) {
    reason =
        e ? std::string([[e description] UTF8String])
          : std::string("probe-load-failed-no-error");
    return false;
  }
  e = nil;
  (void)((BOOL(*)(id, SEL, unsigned int, NSError**))objc_msgSend)(
      model, @selector(unloadWithQoS:error:), 21, &e);
  return true;
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

  s.desc_cls = NSClassFromString(@"_ANEInMemoryModelDescriptor");
  s.model_cls = NSClassFromString(@"_ANEInMemoryModel");
  s.request_cls = NSClassFromString(@"_ANERequest");
  s.iosurface_cls = NSClassFromString(@"_ANEIOSurfaceObject");
  if (
      s.desc_cls == nil || s.model_cls == nil || s.request_cls == nil ||
      s.iosurface_cls == nil) {
    s.reason = "ane-required-classes-missing";
    if (reason_out) {
      *reason_out = s.reason;
    }
    return false;
  }

  std::string probe_reason;
  if (!compile_probe(s, probe_reason)) {
    s.reason = "ane-runtime-probe-failed:" + probe_reason;
    if (reason_out) {
      *reason_out = s.reason;
    }
    return false;
  }

  s.available = true;
  s.reason = "ok";
  if (reason_out) {
    *reason_out = s.reason;
  }
  return true;
}

} // namespace

struct Program {
  id model{nil};
  id request{nil};
  NSString* model_dir{nil};
  std::vector<IOSurfaceRef> input_surfaces;
  std::vector<IOSurfaceRef> output_surfaces;
  std::vector<size_t> input_nbytes;
  std::vector<size_t> output_nbytes;

  ~Program() {
    if (model != nil) {
      NSError* e = nil;
      (void)((BOOL(*)(id, SEL, unsigned int, NSError**))objc_msgSend)(
          model, @selector(unloadWithQoS:error:), 21, &e);
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

  auto prog = std::make_shared<Program>();
  NSString* mil_ns = [NSString stringWithUTF8String:mil.c_str()];
  if (mil_ns == nil) {
    if (reason) {
      *reason = "mil-utf8-conversion-failed";
    }
    return nullptr;
  }
  NSData* mil_data = [mil_ns dataUsingEncoding:NSUTF8StringEncoding];
  auto& s = runtime_state();

  id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
      s.desc_cls, @selector(modelWithMILText:weights:optionsPlist:), mil_data, @{}, nil);
  if (desc == nil) {
    if (reason) {
      *reason = "descriptor-create-failed";
    }
    return nullptr;
  }

  prog->model = ((id(*)(Class, SEL, id))objc_msgSend)(
      s.model_cls, @selector(inMemoryModelWithDescriptor:), desc);
  if (prog->model == nil) {
    if (reason) {
      *reason = "model-create-failed";
    }
    return nullptr;
  }

  NSURL* model_url = ((id(*)(id, SEL))objc_msgSend)(prog->model, @selector(saveModelFiles));
  if (model_url != nil) {
    prog->model_dir = [model_url.path copy];
  }

  NSError* e = nil;
  BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
      prog->model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
  if (!ok) {
    if (reason) {
      *reason = e ? std::string([[e description] UTF8String])
                  : std::string("compile-failed-no-error");
    }
    return nullptr;
  }

  e = nil;
  ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError**))objc_msgSend)(
      prog->model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
  if (!ok) {
    if (reason) {
      *reason = e ? std::string([[e description] UTF8String])
                  : std::string("load-failed-no-error");
    }
    return nullptr;
  }

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

  NSMutableArray* input_objs =
      [NSMutableArray arrayWithCapacity:prog->input_surfaces.size()];
  NSMutableArray* input_indices =
      [NSMutableArray arrayWithCapacity:prog->input_surfaces.size()];
  for (size_t i = 0; i < prog->input_surfaces.size(); ++i) {
    id wrapped = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
        s.iosurface_cls, @selector(objectWithIOSurface:), prog->input_surfaces[i]);
    [input_objs addObject:wrapped];
    [input_indices addObject:@(i)];
  }

  NSMutableArray* output_objs =
      [NSMutableArray arrayWithCapacity:prog->output_surfaces.size()];
  NSMutableArray* output_indices =
      [NSMutableArray arrayWithCapacity:prog->output_surfaces.size()];
  for (size_t i = 0; i < prog->output_surfaces.size(); ++i) {
    id wrapped = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
        s.iosurface_cls, @selector(objectWithIOSurface:), prog->output_surfaces[i]);
    [output_objs addObject:wrapped];
    [output_indices addObject:@(i)];
  }

  prog->request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
      s.request_cls,
      @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
      input_objs,
      input_indices,
      output_objs,
      output_indices,
      nil,
      nil,
      @0);
  if (prog->request == nil) {
    if (reason) {
      *reason = "request-create-failed";
    }
    return nullptr;
  }

  if (reason) {
    *reason = "ok";
  }
  return prog;
}

bool dispatch(Program& program, array& arr, std::string* reason) {
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

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto surface = program.input_surfaces[i];
    IOSurfaceLock(surface, 0, nullptr);
    std::memcpy(
        IOSurfaceGetBaseAddress(surface),
        inputs[i].data<char>(),
        std::min(inputs[i].nbytes(), program.input_nbytes[i]));
    IOSurfaceUnlock(surface, 0, nullptr);
  }

  NSError* e = nil;
  BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError**))objc_msgSend)(
      program.model,
      @selector(evaluateWithQoS:options:request:error:),
      21,
      @{},
      program.request,
      &e);
  if (!ok) {
    if (reason) {
      *reason = e ? std::string([[e description] UTF8String])
                  : std::string("evaluate-failed-no-error");
    }
    return false;
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    auto& out = outputs[i];
    if (out.buffer().ptr() == nullptr) {
      out.set_data(allocator::malloc(out.nbytes()));
    }
    auto surface = program.output_surfaces[i];
    IOSurfaceLock(surface, kIOSurfaceLockReadOnly, nullptr);
    std::memcpy(
        out.data<char>(),
        IOSurfaceGetBaseAddress(surface),
        std::min(out.nbytes(), program.output_nbytes[i]));
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, nullptr);
  }

  if (reason) {
    *reason = "ok";
  }
  return true;
}

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
