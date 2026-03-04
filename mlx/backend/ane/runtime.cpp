// Copyright © 2026 Apple Inc.

#include "mlx/backend/ane/runtime.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#include "mlx/backend/ane/diagnostics.h"
#include "mlx/backend/ane/memory.h"
#include "mlx/backend/ane/private_runtime.h"
#include "mlx/backend/ane/support.h"
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

inline bool profile_mode() {
  static bool enabled = env::get_var("MLX_ANE_PROFILE", 0) == 1;
  return enabled;
}

inline int profile_every_dispatches() {
  static int value = std::max(0, env::get_var("MLX_ANE_PROFILE_EVERY", 0));
  return value;
}

enum class DispatchReturnKind {
  fastpath = 0,
  private_dispatch_success = 1,
  private_dispatch_failed = 2,
  compile_missing = 3,
  runtime_unavailable = 4,
  unexpected = 5,
};

struct RuntimeDispatchProfileCounters {
  std::atomic<uint64_t> dispatch_calls{0};
  std::atomic<uint64_t> dispatch_ns{0};
  std::atomic<uint64_t> fastpath_calls{0};
  std::atomic<uint64_t> fastpath_ns{0};
  std::atomic<uint64_t> lock_ns{0};
  std::atomic<uint64_t> init_ns{0};
  std::atomic<uint64_t> get_or_compile_ns{0};
  std::atomic<uint64_t> iosurface_wrap_ns{0};
  std::atomic<uint64_t> private_dispatch_attempts{0};
  std::atomic<uint64_t> private_dispatch_success{0};
  std::atomic<uint64_t> private_dispatch_ns{0};
  std::atomic<uint64_t> return_fastpath{0};
  std::atomic<uint64_t> return_private_dispatch_success{0};
  std::atomic<uint64_t> return_private_dispatch_failed{0};
  std::atomic<uint64_t> return_compile_missing{0};
  std::atomic<uint64_t> return_runtime_unavailable{0};
  std::atomic<uint64_t> return_unexpected{0};

  std::atomic<uint64_t> goc_calls{0};
  std::atomic<uint64_t> goc_primitive_cache_lookup_ns{0};
  std::atomic<uint64_t> goc_primitive_cache_hits{0};
  std::atomic<uint64_t> goc_primitive_cache_stale{0};
  std::atomic<uint64_t> goc_key_build_ns{0};
  std::atomic<uint64_t> goc_compile_cache_lookup_ns{0};
  std::atomic<uint64_t> goc_compile_cache_hits{0};
  std::atomic<uint64_t> goc_compile_invocations{0};
  std::atomic<uint64_t> goc_compile_ns{0};

  std::atomic<uint64_t> last_print_dispatches{0};
};

RuntimeDispatchProfileCounters& runtime_dispatch_profile() {
  static RuntimeDispatchProfileCounters counters;
  return counters;
}

void print_runtime_dispatch_profile_summary(const char* tag) {
  if (!profile_mode()) {
    return;
  }
  auto& p = runtime_dispatch_profile();
  const auto dispatch_calls = p.dispatch_calls.load(std::memory_order_relaxed);
  const auto dispatch_ns = p.dispatch_ns.load(std::memory_order_relaxed);
  const auto fastpath_calls = p.fastpath_calls.load(std::memory_order_relaxed);
  const auto fastpath_ns = p.fastpath_ns.load(std::memory_order_relaxed);
  const auto lock_ns = p.lock_ns.load(std::memory_order_relaxed);
  const auto init_ns = p.init_ns.load(std::memory_order_relaxed);
  const auto get_or_compile_ns = p.get_or_compile_ns.load(std::memory_order_relaxed);
  const auto iosurface_wrap_ns = p.iosurface_wrap_ns.load(std::memory_order_relaxed);
  const auto private_dispatch_attempts =
      p.private_dispatch_attempts.load(std::memory_order_relaxed);
  const auto private_dispatch_success =
      p.private_dispatch_success.load(std::memory_order_relaxed);
  const auto private_dispatch_ns =
      p.private_dispatch_ns.load(std::memory_order_relaxed);

  const auto return_fastpath = p.return_fastpath.load(std::memory_order_relaxed);
  const auto return_private_dispatch_success =
      p.return_private_dispatch_success.load(std::memory_order_relaxed);
  const auto return_private_dispatch_failed =
      p.return_private_dispatch_failed.load(std::memory_order_relaxed);
  const auto return_compile_missing =
      p.return_compile_missing.load(std::memory_order_relaxed);
  const auto return_runtime_unavailable =
      p.return_runtime_unavailable.load(std::memory_order_relaxed);
  const auto return_unexpected =
      p.return_unexpected.load(std::memory_order_relaxed);

  const auto goc_calls = p.goc_calls.load(std::memory_order_relaxed);
  const auto goc_primitive_cache_lookup_ns =
      p.goc_primitive_cache_lookup_ns.load(std::memory_order_relaxed);
  const auto goc_primitive_cache_hits =
      p.goc_primitive_cache_hits.load(std::memory_order_relaxed);
  const auto goc_primitive_cache_stale =
      p.goc_primitive_cache_stale.load(std::memory_order_relaxed);
  const auto goc_key_build_ns = p.goc_key_build_ns.load(std::memory_order_relaxed);
  const auto goc_compile_cache_lookup_ns =
      p.goc_compile_cache_lookup_ns.load(std::memory_order_relaxed);
  const auto goc_compile_cache_hits =
      p.goc_compile_cache_hits.load(std::memory_order_relaxed);
  const auto goc_compile_invocations =
      p.goc_compile_invocations.load(std::memory_order_relaxed);
  const auto goc_compile_ns = p.goc_compile_ns.load(std::memory_order_relaxed);

  std::cerr << "[ane::runtime_dispatch_profile] tag=" << tag
            << " dispatch_calls=" << dispatch_calls
            << " dispatch_ms=" << ns_to_ms(dispatch_ns)
            << " fastpath_calls=" << fastpath_calls
            << " fastpath_ms=" << ns_to_ms(fastpath_ns)
            << " lock_ms=" << ns_to_ms(lock_ns)
            << " init_ms=" << ns_to_ms(init_ns)
            << " get_or_compile_ms=" << ns_to_ms(get_or_compile_ns)
            << " iosurface_wrap_ms=" << ns_to_ms(iosurface_wrap_ns)
            << " private_dispatch_attempts=" << private_dispatch_attempts
            << " private_dispatch_success=" << private_dispatch_success
            << " private_dispatch_ms=" << ns_to_ms(private_dispatch_ns)
            << " return_fastpath=" << return_fastpath
            << " return_private_ok=" << return_private_dispatch_success
            << " return_private_fail=" << return_private_dispatch_failed
            << " return_compile_missing=" << return_compile_missing
            << " return_runtime_unavailable=" << return_runtime_unavailable
            << " return_unexpected=" << return_unexpected
            << " goc_calls=" << goc_calls
            << " goc_primitive_lookup_ms="
            << ns_to_ms(goc_primitive_cache_lookup_ns)
            << " goc_primitive_hits=" << goc_primitive_cache_hits
            << " goc_primitive_stale=" << goc_primitive_cache_stale
            << " goc_key_build_ms=" << ns_to_ms(goc_key_build_ns)
            << " goc_compile_lookup_ms=" << ns_to_ms(goc_compile_cache_lookup_ns)
            << " goc_compile_hits=" << goc_compile_cache_hits
            << " goc_compile_invocations=" << goc_compile_invocations
            << " goc_compile_ms=" << ns_to_ms(goc_compile_ns);
  if (dispatch_calls > 0) {
    std::cerr << " avg_dispatch_ms="
              << (ns_to_ms(dispatch_ns) / static_cast<double>(dispatch_calls));
  }
  if (private_dispatch_attempts > 0) {
    std::cerr << " avg_private_dispatch_ms="
              << (ns_to_ms(private_dispatch_ns) /
                  static_cast<double>(private_dispatch_attempts));
  }
  std::cerr << "\n";
}

void maybe_print_runtime_dispatch_profile_periodic() {
  if (!profile_mode()) {
    return;
  }
  const int every = profile_every_dispatches();
  if (every <= 0) {
    return;
  }
  auto& p = runtime_dispatch_profile();
  const uint64_t current = p.dispatch_calls.load(std::memory_order_relaxed);
  uint64_t last = p.last_print_dispatches.load(std::memory_order_relaxed);
  if (current < static_cast<uint64_t>(every) ||
      current - last < static_cast<uint64_t>(every)) {
    return;
  }
  if (!p.last_print_dispatches.compare_exchange_strong(
          last, current, std::memory_order_relaxed, std::memory_order_relaxed)) {
    return;
  }
  print_runtime_dispatch_profile_summary("periodic");
}

void install_runtime_dispatch_profile_exit_reporter() {
  static bool installed = [] {
    if (profile_mode()) {
      std::atexit([]() { print_runtime_dispatch_profile_summary("final"); });
    }
    return true;
  }();
  (void)installed;
}

struct RuntimeDispatchProfileScope {
  bool enabled{false};
  uint64_t begin_ns{0};
  uint64_t fastpath_ns{0};
  uint64_t lock_ns{0};
  uint64_t init_ns{0};
  uint64_t get_or_compile_ns{0};
  uint64_t iosurface_wrap_ns{0};
  uint64_t private_dispatch_ns{0};
  bool fastpath_hit{false};
  bool private_dispatch_attempted{false};
  bool private_dispatch_ok{false};
  DispatchReturnKind return_kind{DispatchReturnKind::unexpected};

  RuntimeDispatchProfileScope() : enabled(profile_mode()) {
    if (enabled) {
      install_runtime_dispatch_profile_exit_reporter();
      begin_ns = now_ns();
    }
  }

  ~RuntimeDispatchProfileScope() {
    if (!enabled) {
      return;
    }
    auto& p = runtime_dispatch_profile();
    p.dispatch_calls.fetch_add(1, std::memory_order_relaxed);
    p.dispatch_ns.fetch_add(now_ns() - begin_ns, std::memory_order_relaxed);
    p.fastpath_ns.fetch_add(fastpath_ns, std::memory_order_relaxed);
    if (fastpath_hit) {
      p.fastpath_calls.fetch_add(1, std::memory_order_relaxed);
    }
    p.lock_ns.fetch_add(lock_ns, std::memory_order_relaxed);
    p.init_ns.fetch_add(init_ns, std::memory_order_relaxed);
    p.get_or_compile_ns.fetch_add(get_or_compile_ns, std::memory_order_relaxed);
    p.iosurface_wrap_ns.fetch_add(iosurface_wrap_ns, std::memory_order_relaxed);
    if (private_dispatch_attempted) {
      p.private_dispatch_attempts.fetch_add(1, std::memory_order_relaxed);
      p.private_dispatch_ns.fetch_add(private_dispatch_ns, std::memory_order_relaxed);
      if (private_dispatch_ok) {
        p.private_dispatch_success.fetch_add(1, std::memory_order_relaxed);
      }
    }
    switch (return_kind) {
      case DispatchReturnKind::fastpath:
        p.return_fastpath.fetch_add(1, std::memory_order_relaxed);
        break;
      case DispatchReturnKind::private_dispatch_success:
        p.return_private_dispatch_success.fetch_add(1, std::memory_order_relaxed);
        break;
      case DispatchReturnKind::private_dispatch_failed:
        p.return_private_dispatch_failed.fetch_add(1, std::memory_order_relaxed);
        break;
      case DispatchReturnKind::compile_missing:
        p.return_compile_missing.fetch_add(1, std::memory_order_relaxed);
        break;
      case DispatchReturnKind::runtime_unavailable:
        p.return_runtime_unavailable.fetch_add(1, std::memory_order_relaxed);
        break;
      case DispatchReturnKind::unexpected:
        p.return_unexpected.fetch_add(1, std::memory_order_relaxed);
        break;
    }
    maybe_print_runtime_dispatch_profile_periodic();
  }
};

struct GetOrCompileProfileScope {
  bool enabled{false};
  uint64_t primitive_lookup_ns{0};
  uint64_t key_build_ns{0};
  uint64_t compile_lookup_ns{0};
  uint64_t compile_ns{0};
  bool primitive_hit{false};
  bool primitive_stale{false};
  bool compile_hit{false};
  bool compile_invoked{false};

  GetOrCompileProfileScope() : enabled(profile_mode()) {}

  ~GetOrCompileProfileScope() {
    if (!enabled) {
      return;
    }
    auto& p = runtime_dispatch_profile();
    p.goc_calls.fetch_add(1, std::memory_order_relaxed);
    p.goc_primitive_cache_lookup_ns.fetch_add(
        primitive_lookup_ns,
        std::memory_order_relaxed);
    p.goc_key_build_ns.fetch_add(key_build_ns, std::memory_order_relaxed);
    p.goc_compile_cache_lookup_ns.fetch_add(
        compile_lookup_ns,
        std::memory_order_relaxed);
    if (primitive_hit) {
      p.goc_primitive_cache_hits.fetch_add(1, std::memory_order_relaxed);
    }
    if (primitive_stale) {
      p.goc_primitive_cache_stale.fetch_add(1, std::memory_order_relaxed);
    }
    if (compile_hit) {
      p.goc_compile_cache_hits.fetch_add(1, std::memory_order_relaxed);
    }
    if (compile_invoked) {
      p.goc_compile_invocations.fetch_add(1, std::memory_order_relaxed);
      p.goc_compile_ns.fetch_add(compile_ns, std::memory_order_relaxed);
    }
  }
};

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
  GetOrCompileProfileScope profile_scope;
  auto& primitive = arr.primitive();
  const auto primitive_id = arr.primitive_id();

  const uint64_t primitive_lookup_begin_ns =
      profile_scope.enabled ? now_ns() : 0;
  auto pit = primitive_cache_.find(primitive_id);
  if (profile_scope.enabled) {
    profile_scope.primitive_lookup_ns += now_ns() - primitive_lookup_begin_ns;
  }
  if (pit != primitive_cache_.end()) {
    const auto& cached = pit->second;
    if (cached != nullptr && program_matches(*cached, arr)) {
      profile_scope.primitive_hit = true;
      if (diagnostics) {
        note_compile_cache_hit(primitive);
      }
      return cached;
    }
    profile_scope.primitive_stale = true;
    primitive_cache_.erase(pit);
  }

  const uint64_t key_build_begin_ns = profile_scope.enabled ? now_ns() : 0;
  auto key = make_cache_key(arr);
  if (profile_scope.enabled) {
    profile_scope.key_build_ns += now_ns() - key_build_begin_ns;
  }

  const uint64_t compile_lookup_begin_ns = profile_scope.enabled ? now_ns() : 0;
  auto it = compile_cache_.find(key);
  if (profile_scope.enabled) {
    profile_scope.compile_lookup_ns += now_ns() - compile_lookup_begin_ns;
  }
  if (it != compile_cache_.end()) {
    profile_scope.compile_hit = true;
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
    profile_scope.compile_invoked = true;
    const uint64_t compile_begin_ns = profile_scope.enabled ? now_ns() : 0;
    std::string reason;
    program->native_program = private_runtime::compile(arr, &reason);
    program->native_compile_reason = reason;
    if (profile_scope.enabled) {
      profile_scope.compile_ns += now_ns() - compile_begin_ns;
    }
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
  RuntimeDispatchProfileScope profile_scope;
  if (is_metadata_fastpath_primitive(arr.primitive())) {
    const uint64_t fastpath_begin_ns = profile_scope.enabled ? now_ns() : 0;
    std::string fast_reason;
    if (private_runtime::dispatch_fastpath(arr, &fast_reason)) {
      if (profile_scope.enabled) {
        profile_scope.fastpath_ns += now_ns() - fastpath_begin_ns;
        profile_scope.fastpath_hit = true;
      }
      profile_scope.return_kind = DispatchReturnKind::fastpath;
      return {DispatchStatus::dispatched, fast_reason};
    }
    if (profile_scope.enabled) {
      profile_scope.fastpath_ns += now_ns() - fastpath_begin_ns;
    }
  }

  std::shared_ptr<CompiledProgram> program;
  const uint64_t lock_begin_ns = profile_scope.enabled ? now_ns() : 0;
  {
    std::lock_guard<std::mutex> lk(mutex_);
    const uint64_t init_begin_ns = profile_scope.enabled ? now_ns() : 0;
    (void)try_initialize_runtime();
    if (profile_scope.enabled) {
      profile_scope.init_ns += now_ns() - init_begin_ns;
    }
    const uint64_t get_or_compile_begin_ns = profile_scope.enabled ? now_ns() : 0;
    program = get_or_compile(arr);
    if (profile_scope.enabled) {
      profile_scope.get_or_compile_ns += now_ns() - get_or_compile_begin_ns;
    }
  }
  if (profile_scope.enabled) {
    profile_scope.lock_ns += now_ns() - lock_begin_ns;
  }

  // Build explicit IOSurface bindings when requested so buffer wrapping is
  // validated even before private-runtime dispatch is fully integrated.
  if (should_use_iosurface()) {
    const uint64_t wrap_begin_ns = profile_scope.enabled ? now_ns() : 0;
    for (const auto& in : arr.inputs()) {
      (void)wrap_array_to_surface(in);
    }
    if (profile_scope.enabled) {
      profile_scope.iosurface_wrap_ns += now_ns() - wrap_begin_ns;
    }
  }

  if (runtime_available_ && program && program->native_program) {
    profile_scope.private_dispatch_attempted = true;
    const uint64_t private_dispatch_begin_ns = profile_scope.enabled ? now_ns() : 0;
    std::string reason;
    if (private_runtime::dispatch(*program->native_program, arr, &reason)) {
      if (profile_scope.enabled) {
        profile_scope.private_dispatch_ns += now_ns() - private_dispatch_begin_ns;
        profile_scope.private_dispatch_ok = true;
      }
      profile_scope.return_kind = DispatchReturnKind::private_dispatch_success;
      return {DispatchStatus::dispatched, "private-runtime-dispatch"};
    }
    if (profile_scope.enabled) {
      profile_scope.private_dispatch_ns += now_ns() - private_dispatch_begin_ns;
    }
    profile_scope.return_kind = DispatchReturnKind::private_dispatch_failed;
    return {DispatchStatus::dispatch_failed, reason};
  }

  if (runtime_available_ && program && !program->native_program) {
    profile_scope.return_kind = DispatchReturnKind::compile_missing;
    return {DispatchStatus::dispatch_failed, program->native_compile_reason};
  }

  if (!runtime_available_) {
    profile_scope.return_kind = DispatchReturnKind::runtime_unavailable;
    return {
        DispatchStatus::runtime_unavailable,
        "runtime-unavailable:" + runtime_unavailable_reason_,
    };
  }

  profile_scope.return_kind = DispatchReturnKind::unexpected;
  return {DispatchStatus::dispatch_failed, "unexpected-runtime-state"};
}

} // namespace mlx::core::ane
