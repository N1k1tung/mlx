// Copyright © 2026 Apple Inc.

#include "mlx/backend/ane/runtime.h"

#include <dlfcn.h>

#include <cstdlib>
#include <sstream>

#include "mlx/backend/ane/diagnostics.h"
#include "mlx/backend/ane/memory.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core::ane {

namespace {

std::string read_env(const char* name) {
  if (const char* value = std::getenv(name)) {
    return std::string(value);
  }
  return {};
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
  // ANE streams currently reuse Metal queue/command-buffer lifecycle.
  gpu::new_stream(stream);
}

void Runtime::finalize(Stream stream) {
  gpu::finalize(stream);
}

void Runtime::synchronize(Stream stream) {
  gpu::synchronize(stream);
}

std::string Runtime::make_cache_key(const array& arr) const {
  std::ostringstream os;
  auto& p = arr.primitive();
  os << p.name() << "|inputs=" << arr.inputs().size()
     << "|outputs=" << arr.outputs().size();

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
  auto key = make_cache_key(arr);
  auto& primitive = arr.primitive();
  auto it = compile_cache_.find(key);
  if (it != compile_cache_.end()) {
    note_compile_cache_hit(primitive);
    return it->second;
  }
  note_compile_cache_miss(primitive);
  auto program = std::make_shared<CompiledProgram>();
  program->key = key;
  program->primitive = primitive.name();
  program->num_inputs = arr.inputs().size();
  program->num_outputs = arr.outputs().size();
  compile_cache_.emplace(key, program);
  return program;
}

bool Runtime::try_initialize_runtime() {
  if (runtime_checked_) {
    return runtime_available_;
  }
  runtime_checked_ = true;

  // Allow an explicit private-runtime dylib path for internal deployments.
  // We do not assume a stable private ABI in open-source builds.
  auto dylib_path = read_env("MLX_ANE_RUNTIME_DYLIB");
  if (dylib_path.empty()) {
    runtime_available_ = false;
    return runtime_available_;
  }

  runtime_handle_ = dlopen(dylib_path.c_str(), RTLD_LOCAL | RTLD_LAZY);
  runtime_available_ = (runtime_handle_ != nullptr);
  return runtime_available_;
}

bool Runtime::should_use_iosurface() const {
  static bool use_iosurface = env::get_var("MLX_ANE_ENABLE_IOSURFACE", 0) == 1;
  return use_iosurface;
}

bool Runtime::emulation_enabled() const {
  // Default on to preserve current behavior until a private runtime is wired.
  static bool emulate = env::get_var("MLX_ANE_EMULATE", 1) == 1;
  return emulate;
}

DispatchResult Runtime::dispatch(array& arr) {
  {
    std::lock_guard<std::mutex> lk(mutex_);
    get_or_compile(arr);
    (void)try_initialize_runtime();
  }

  // Build explicit IOSurface bindings when requested so buffer wrapping is
  // validated even before private-runtime dispatch is fully integrated.
  if (should_use_iosurface()) {
    for (const auto& in : arr.inputs()) {
      (void)wrap_array_to_surface(in);
    }
  }

  if (runtime_available_) {
    return {
        DispatchStatus::dispatch_failed,
        "private-runtime-abi-not-integrated",
    };
  }

  if (!emulation_enabled()) {
    return {DispatchStatus::runtime_unavailable, "runtime-unavailable"};
  }

  gpu::eval(arr);
  return {DispatchStatus::dispatched_emulated, "emulated-via-gpu"};
}

} // namespace mlx::core::ane
