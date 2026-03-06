#include "mlx/backend/ane/diagnostics.h"
#include "mlx/backend/ane/eval.h"
#include "mlx/backend/ane/partition.h"
#include "mlx/backend/ane/runtime.h"
#include "mlx/backend/ane/support.h"
#include "mlx/backend/cpu/eval.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>

namespace mlx::core::ane {

namespace {

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

inline int profile_every_evals() {
  static int value = std::max(0, env::get_var("MLX_ANE_PROFILE_EVERY", 1000));
  return value;
}

struct BackendProfileCounters {
  std::atomic<uint64_t> eval_calls{0};
  std::atomic<uint64_t> eval_ns{0};
  std::atomic<uint64_t> route_ns{0};
  std::atomic<uint64_t> runtime_dispatch_attempts{0};
  std::atomic<uint64_t> runtime_dispatch_success{0};
  std::atomic<uint64_t> runtime_dispatch_fail{0};
  std::atomic<uint64_t> runtime_dispatch_ns{0};
  std::atomic<uint64_t> gpu_fallback_calls{0};
  std::atomic<uint64_t> gpu_fallback_eval_ns{0};
  std::atomic<uint64_t> cpu_fallback_calls{0};
  std::atomic<uint64_t> cpu_fallback_eval_ns{0};
  std::atomic<uint64_t> synchronize_calls{0};
  std::atomic<uint64_t> synchronize_ns{0};
  std::atomic<uint64_t> finalize_calls{0};
  std::atomic<uint64_t> finalize_ns{0};
  std::atomic<uint64_t> last_print_evals{0};
};

BackendProfileCounters& backend_profile() {
  static BackendProfileCounters counters;
  return counters;
}

void print_backend_profile_summary(const char* tag) {
  if (!profile_mode()) {
    return;
  }

  auto& p = backend_profile();
  const auto eval_calls = p.eval_calls.load(std::memory_order_relaxed);
  const auto eval_ns = p.eval_ns.load(std::memory_order_relaxed);
  const auto route_ns = p.route_ns.load(std::memory_order_relaxed);
  const auto runtime_dispatch_attempts =
      p.runtime_dispatch_attempts.load(std::memory_order_relaxed);
  const auto runtime_dispatch_success =
      p.runtime_dispatch_success.load(std::memory_order_relaxed);
  const auto runtime_dispatch_fail =
      p.runtime_dispatch_fail.load(std::memory_order_relaxed);
  const auto runtime_dispatch_ns =
      p.runtime_dispatch_ns.load(std::memory_order_relaxed);
  const auto gpu_fallback_calls =
      p.gpu_fallback_calls.load(std::memory_order_relaxed);
  const auto gpu_fallback_eval_ns =
      p.gpu_fallback_eval_ns.load(std::memory_order_relaxed);
  const auto cpu_fallback_calls =
      p.cpu_fallback_calls.load(std::memory_order_relaxed);
  const auto cpu_fallback_eval_ns =
      p.cpu_fallback_eval_ns.load(std::memory_order_relaxed);
  const auto synchronize_calls =
      p.synchronize_calls.load(std::memory_order_relaxed);
  const auto synchronize_ns = p.synchronize_ns.load(std::memory_order_relaxed);
  const auto finalize_calls = p.finalize_calls.load(std::memory_order_relaxed);
  const auto finalize_ns = p.finalize_ns.load(std::memory_order_relaxed);

  const uint64_t eval_accounted_ns =
      route_ns + runtime_dispatch_ns + gpu_fallback_eval_ns + cpu_fallback_eval_ns;
  const uint64_t eval_untracked_ns =
      (eval_ns > eval_accounted_ns) ? (eval_ns - eval_accounted_ns) : 0;
  const uint64_t wall_ns = eval_ns + synchronize_ns + finalize_ns;
  const uint64_t known_ns = eval_accounted_ns + synchronize_ns + finalize_ns;
  const uint64_t wall_untracked_ns = (wall_ns > known_ns) ? (wall_ns - known_ns) : 0;

  std::cerr << "[ane::backend_profile] tag=" << tag
            << " eval_calls=" << eval_calls
            << " eval_ms=" << ns_to_ms(eval_ns)
            << " route_ms=" << ns_to_ms(route_ns)
            << " runtime_dispatch_attempts=" << runtime_dispatch_attempts
            << " runtime_dispatch_success=" << runtime_dispatch_success
            << " runtime_dispatch_fail=" << runtime_dispatch_fail
            << " runtime_dispatch_ms=" << ns_to_ms(runtime_dispatch_ns)
            << " gpu_fallback_calls=" << gpu_fallback_calls
            << " gpu_fallback_eval_ms=" << ns_to_ms(gpu_fallback_eval_ns)
            << " cpu_fallback_calls=" << cpu_fallback_calls
            << " cpu_fallback_eval_ms=" << ns_to_ms(cpu_fallback_eval_ns)
            << " eval_untracked_ms=" << ns_to_ms(eval_untracked_ns)
            << " synchronize_calls=" << synchronize_calls
            << " synchronize_ms=" << ns_to_ms(synchronize_ns)
            << " finalize_calls=" << finalize_calls
            << " finalize_ms=" << ns_to_ms(finalize_ns)
            << " backend_wall_ms=" << ns_to_ms(wall_ns)
            << " backend_known_ms=" << ns_to_ms(known_ns)
            << " backend_untracked_ms=" << ns_to_ms(wall_untracked_ns);
  if (eval_calls > 0) {
    std::cerr << " avg_eval_ms="
              << (ns_to_ms(eval_ns) / static_cast<double>(eval_calls));
  }
  if (synchronize_calls > 0) {
    std::cerr << " avg_synchronize_ms="
              << (ns_to_ms(synchronize_ns) / static_cast<double>(synchronize_calls));
  }
  std::cerr << "\n";
}

void maybe_print_backend_profile_periodic() {
  if (!profile_mode()) {
    return;
  }
  const int every = profile_every_evals();
  if (every <= 0) {
    return;
  }
  auto& p = backend_profile();
  const uint64_t current = p.eval_calls.load(std::memory_order_relaxed);
  uint64_t last = p.last_print_evals.load(std::memory_order_relaxed);
  if (current < static_cast<uint64_t>(every) ||
      current - last < static_cast<uint64_t>(every)) {
    return;
  }
  if (!p.last_print_evals.compare_exchange_strong(
          last, current, std::memory_order_relaxed, std::memory_order_relaxed)) {
    return;
  }
  print_backend_profile_summary("periodic");
}

void install_backend_profile_exit_reporter() {
  static bool installed = [] {
    if (profile_mode()) {
      std::atexit([]() { print_backend_profile_summary("final"); });
    }
    return true;
  }();
  (void)installed;
}

struct BackendEvalProfileScope {
  bool enabled{false};
  uint64_t begin_ns{0};
  uint64_t route_ns{0};
  uint64_t runtime_dispatch_ns{0};
  uint64_t gpu_fallback_eval_ns{0};
  uint64_t cpu_fallback_eval_ns{0};
  bool runtime_dispatch_attempted{false};
  bool runtime_dispatch_success{false};
  bool gpu_fallback_taken{false};
  bool cpu_fallback_taken{false};

  BackendEvalProfileScope() : enabled(profile_mode()) {
    if (enabled) {
      install_backend_profile_exit_reporter();
      begin_ns = now_ns();
    }
  }

  ~BackendEvalProfileScope() {
    if (!enabled) {
      return;
    }
    auto& p = backend_profile();
    p.eval_calls.fetch_add(1, std::memory_order_relaxed);
    p.eval_ns.fetch_add(now_ns() - begin_ns, std::memory_order_relaxed);
    p.route_ns.fetch_add(route_ns, std::memory_order_relaxed);
    if (runtime_dispatch_attempted) {
      p.runtime_dispatch_attempts.fetch_add(1, std::memory_order_relaxed);
      if (runtime_dispatch_success) {
        p.runtime_dispatch_success.fetch_add(1, std::memory_order_relaxed);
      } else {
        p.runtime_dispatch_fail.fetch_add(1, std::memory_order_relaxed);
      }
      p.runtime_dispatch_ns.fetch_add(runtime_dispatch_ns, std::memory_order_relaxed);
    }
    if (gpu_fallback_taken) {
      p.gpu_fallback_calls.fetch_add(1, std::memory_order_relaxed);
      p.gpu_fallback_eval_ns.fetch_add(
          gpu_fallback_eval_ns,
          std::memory_order_relaxed);
    }
    if (cpu_fallback_taken) {
      p.cpu_fallback_calls.fetch_add(1, std::memory_order_relaxed);
      p.cpu_fallback_eval_ns.fetch_add(
          cpu_fallback_eval_ns,
          std::memory_order_relaxed);
    }
    maybe_print_backend_profile_periodic();
  }
};

void note_synchronize_profile(uint64_t ns) {
  if (!profile_mode()) {
    return;
  }
  install_backend_profile_exit_reporter();
  auto& p = backend_profile();
  p.synchronize_calls.fetch_add(1, std::memory_order_relaxed);
  p.synchronize_ns.fetch_add(ns, std::memory_order_relaxed);
}

void note_finalize_profile(uint64_t ns) {
  if (!profile_mode()) {
    return;
  }
  install_backend_profile_exit_reporter();
  auto& p = backend_profile();
  p.finalize_calls.fetch_add(1, std::memory_order_relaxed);
  p.finalize_ns.fetch_add(ns, std::memory_order_relaxed);
}

} // namespace

void new_stream(Stream stream) {
  runtime().new_stream(stream);
}

void eval(array& arr) {
  BackendEvalProfileScope profile_scope;
  auto& primitive = arr.primitive();

  const uint64_t route_begin_ns = profile_scope.enabled ? now_ns() : 0;
  auto decision = decide_route(arr);
  if (profile_scope.enabled) {
    profile_scope.route_ns += now_ns() - route_begin_ns;
  }

  const bool diagnostics = diagnostics_mode();
  if (diagnostics) {
    note_total(primitive, decision.supported);
    track_route_boundary(primitive.stream(), decision.route);
  }

  if (decision.route == Route::ane) {
    const uint64_t dispatch_begin_ns = profile_scope.enabled ? now_ns() : 0;
    auto result = runtime().dispatch(arr);
    if (profile_scope.enabled) {
      profile_scope.runtime_dispatch_attempted = true;
      profile_scope.runtime_dispatch_success = result.executed();
      profile_scope.runtime_dispatch_ns += now_ns() - dispatch_begin_ns;
    }
    if (result.executed()) {
      if (diagnostics) {
        note_ane_dispatch(primitive);
      }
      return;
    }
    if (diagnostics) {
      note_gpu_fallback(primitive, result.reason);
    }
  } else {
    if (diagnostics) {
      note_gpu_fallback(primitive, decision.reason);
    }
  }

  // Route to GPU first and then CPU as the terminal fallback.
  const uint64_t gpu_begin_ns = profile_scope.enabled ? now_ns() : 0;
  try {
    gpu::eval(arr);
    if (profile_scope.enabled) {
      profile_scope.gpu_fallback_taken = true;
      profile_scope.gpu_fallback_eval_ns += now_ns() - gpu_begin_ns;
    }
    return;
  } catch (const std::runtime_error&) {
    if (profile_scope.enabled) {
      profile_scope.gpu_fallback_taken = true;
      profile_scope.gpu_fallback_eval_ns += now_ns() - gpu_begin_ns;
    }
    if (diagnostics) {
      note_cpu_fallback(primitive, "gpu-eval-failed");
    }
    const uint64_t cpu_begin_ns = profile_scope.enabled ? now_ns() : 0;
    cpu::eval(arr);
    if (profile_scope.enabled) {
      profile_scope.cpu_fallback_taken = true;
      profile_scope.cpu_fallback_eval_ns += now_ns() - cpu_begin_ns;
    }
  }
}

void finalize(Stream s) {
  const uint64_t begin_ns = profile_mode() ? now_ns() : 0;
  runtime().finalize(s);
  if (profile_mode()) {
    note_finalize_profile(now_ns() - begin_ns);
  }
}

void synchronize(Stream s) {
  const uint64_t begin_ns = profile_mode() ? now_ns() : 0;
  runtime().synchronize(s);
  if (profile_mode()) {
    note_synchronize_profile(now_ns() - begin_ns);
  }
}

} // namespace mlx::core::ane
