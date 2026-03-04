// Copyright © 2026 Apple Inc.

#include "mlx/backend/ane/runtime.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#include "mlx/backend/ane/diagnostics.h"
#include "mlx/backend/ane/memory.h"
#include "mlx/backend/ane/private_runtime.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core::ane {

namespace {

bool private_runtime_enabled() {
  static bool enabled = env::get_var("MLX_ANE_PRIVATE_RUNTIME", 1) == 1;
  return enabled;
}

} // namespace

Runtime& Runtime::instance() {
  static Runtime runtime;
  return runtime;
}

Runtime& runtime() {
  return Runtime::instance();
}

void Runtime::new_stream(Stream stream) {
  // ANE streams currently reuse Metal queue/command-buffer lifecycle when
  // available. Keep this optional so ANE does not hard-crash on systems where
  // GPU backend is unavailable.
  if (gpu::is_available()) {
    gpu::new_stream(stream);
  }
}

void Runtime::finalize(Stream stream) {
  if (gpu::is_available()) {
    gpu::finalize(stream);
  }
}

void Runtime::synchronize(Stream stream) {
  if (gpu::is_available()) {
    gpu::synchronize(stream);
  }
}

bool Runtime::is_runtime_available() {
  std::lock_guard<std::mutex> lk(mutex_);
  (void)try_initialize_runtime();
  return runtime_available_;
}

std::string Runtime::runtime_unavailable_reason() {
  std::lock_guard<std::mutex> lk(mutex_);
  (void)try_initialize_runtime();
  return runtime_unavailable_reason_;
}

std::string Runtime::make_cache_key(const array& arr) const {
  std::ostringstream os;
  os << std::setprecision(std::numeric_limits<float>::max_digits10);
  auto& p = arr.primitive();
  os << p.name() << "|inputs=" << arr.inputs().size()
     << "|outputs=" << arr.outputs().size();

  if (const auto* rms = dynamic_cast<const fast::RMSNorm*>(&p); rms != nullptr) {
    os << "|eps=" << rms->state().second;
  }

  for (const auto& in : arr.inputs()) {
    os << "|in:" << in.dtype() << ":" << in.shape();
  }
  for (const auto& out : arr.outputs()) {
    os << "|out:" << out.dtype() << ":" << out.shape();
  }
  return os.str();
}

std::shared_ptr<Runtime::CompiledProgram> Runtime::get_or_compile(
    const array& arr) {
  const bool diagnostics = diagnostics_mode();
  auto& primitive = arr.primitive();
  const auto primitive_id = arr.primitive_id();

  auto pit = primitive_cache_.find(primitive_id);
  if (pit != primitive_cache_.end()) {
    const auto& cached = pit->second;
    if (cached != nullptr && program_matches(*cached, arr)) {
      if (diagnostics) {
        note_compile_cache_hit(primitive);
      }
      return cached;
    }
    primitive_cache_.erase(pit);
  }

  auto key = make_cache_key(arr);
  auto it = compile_cache_.find(key);
  if (it != compile_cache_.end()) {
    if (diagnostics) {
      note_compile_cache_hit(primitive);
    }
    primitive_cache_[primitive_id] = it->second;
    return it->second;
  }
  if (diagnostics) {
    note_compile_cache_miss(primitive);
  }
  auto program = std::make_shared<CompiledProgram>();
  program->key = key;
  program->primitive = primitive.name();
  program->primitive_id = primitive_id;
  program->num_inputs = arr.inputs().size();
  program->num_outputs = arr.outputs().size();
  program->input_shapes.reserve(arr.inputs().size());
  program->input_dtypes.reserve(arr.inputs().size());
  for (const auto& in : arr.inputs()) {
    program->input_shapes.push_back(in.shape());
    program->input_dtypes.push_back(in.dtype());
  }
  auto outputs = arr.outputs();
  program->output_shapes.reserve(outputs.size());
  program->output_dtypes.reserve(outputs.size());
  for (const auto& out : outputs) {
    program->output_shapes.push_back(out.shape());
    program->output_dtypes.push_back(out.dtype());
  }
  if (const auto* rms = dynamic_cast<const fast::RMSNorm*>(&primitive); rms != nullptr) {
    program->has_rms_eps = true;
    program->rms_eps = rms->state().second;
  }
  if (runtime_available_) {
    std::string reason;
    program->native_program = private_runtime::compile(arr, &reason);
    program->native_compile_reason = reason;
  } else {
    program->native_compile_reason = runtime_unavailable_reason_;
  }
  compile_cache_.emplace(key, program);
  primitive_cache_[primitive_id] = program;
  return program;
}

bool Runtime::program_matches(const CompiledProgram& program, const array& arr) const {
  if (program.num_inputs != arr.inputs().size() ||
      program.num_outputs != arr.outputs().size()) {
    return false;
  }
  if (program.input_shapes.size() != arr.inputs().size() ||
      program.input_dtypes.size() != arr.inputs().size()) {
    return false;
  }
  auto outputs = arr.outputs();
  if (program.output_shapes.size() != outputs.size() ||
      program.output_dtypes.size() != outputs.size()) {
    return false;
  }
  for (size_t i = 0; i < arr.inputs().size(); ++i) {
    if (program.input_dtypes[i] != arr.inputs()[i].dtype() ||
        program.input_shapes[i] != arr.inputs()[i].shape()) {
      return false;
    }
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (program.output_dtypes[i] != outputs[i].dtype() ||
        program.output_shapes[i] != outputs[i].shape()) {
      return false;
    }
  }
  if (program.has_rms_eps) {
    const auto* rms = dynamic_cast<const fast::RMSNorm*>(&arr.primitive());
    if (rms == nullptr || rms->state().second != program.rms_eps) {
      return false;
    }
  }
  return true;
}

bool Runtime::try_initialize_runtime() {
  if (runtime_checked_) {
    return runtime_available_;
  }
  runtime_checked_ = true;

  if (!private_runtime_enabled()) {
    runtime_available_ = false;
    runtime_unavailable_reason_ = "private-runtime-disabled";
    return runtime_available_;
  }

  runtime_available_ =
      private_runtime::available(&runtime_unavailable_reason_);
  if (!runtime_available_) {
    std::cerr << "[ane::runtime] unavailable: " << runtime_unavailable_reason_
              << "\n";
  }
  return runtime_available_;
}

bool Runtime::should_use_iosurface() const {
  static bool use_iosurface = env::get_var("MLX_ANE_ENABLE_IOSURFACE", 0) == 1;
  return use_iosurface;
}

DispatchResult Runtime::dispatch(array& arr) {
  {
    std::string fast_reason;
    if (private_runtime::dispatch_fastpath(arr, &fast_reason)) {
      return {DispatchStatus::dispatched, fast_reason};
    }
  }

  std::shared_ptr<CompiledProgram> program;
  {
    std::lock_guard<std::mutex> lk(mutex_);
    (void)try_initialize_runtime();
    program = get_or_compile(arr);
  }

  // Build explicit IOSurface bindings when requested so buffer wrapping is
  // validated even before private-runtime dispatch is fully integrated.
  if (should_use_iosurface()) {
    for (const auto& in : arr.inputs()) {
      (void)wrap_array_to_surface(in);
    }
  }

  if (runtime_available_ && program && program->native_program) {
    std::string reason;
    if (private_runtime::dispatch(*program->native_program, arr, &reason)) {
      return {DispatchStatus::dispatched, "private-runtime-dispatch"};
    }
    return {DispatchStatus::dispatch_failed, reason};
  }

  if (runtime_available_ && program && !program->native_program) {
    return {DispatchStatus::dispatch_failed, program->native_compile_reason};
  }

  if (!runtime_available_) {
    return {
        DispatchStatus::runtime_unavailable,
        "runtime-unavailable:" + runtime_unavailable_reason_,
    };
  }

  return {DispatchStatus::dispatch_failed, "unexpected-runtime-state"};
}

} // namespace mlx::core::ane
