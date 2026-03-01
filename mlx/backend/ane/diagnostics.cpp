// Copyright © 2026 Apple Inc.

#include "mlx/backend/ane/diagnostics.h"

#include <cstdlib>
#include <iostream>
#include <mutex>

#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core::ane {

namespace {

struct DiagnosticsState {
  std::mutex mutex;
  DiagnosticsSnapshot snapshot;
  bool report_registered{false};
};

DiagnosticsState& state() {
  static DiagnosticsState s;
  return s;
}

void print_summary() {
  auto s = get_diagnostics();
  std::cerr << "[ane::diagnostics] total_ops=" << s.total_ops
            << " supported_ops=" << s.supported_ops
            << " ane_dispatches=" << s.ane_dispatches
            << " ane_emulated_dispatches=" << s.ane_emulated_dispatches
            << " gpu_fallbacks=" << s.gpu_fallbacks
            << " cpu_fallbacks=" << s.cpu_fallbacks
            << " strict_rejections=" << s.strict_rejections
            << " compile_cache_hits=" << s.compile_cache_hits
            << " compile_cache_misses=" << s.compile_cache_misses
            << " partition_boundaries=" << s.partition_boundaries << "\n";
}

void register_summary_report() {
  if (!report_mode()) {
    return;
  }
  auto& s = state();
  std::lock_guard<std::mutex> lk(s.mutex);
  if (!s.report_registered) {
    std::atexit(print_summary);
    s.report_registered = true;
  }
}

void verbose_log(
    const Primitive& primitive,
    std::string_view route,
    std::string_view reason) {
  if (!verbose_mode()) {
    return;
  }
  std::cerr << "[ane::route] op=" << primitive.name() << " route=" << route;
  if (!reason.empty()) {
    std::cerr << " reason=" << reason;
  }
  std::cerr << "\n";
}

} // namespace

DiagnosticsSnapshot get_diagnostics() {
  auto& s = state();
  std::lock_guard<std::mutex> lk(s.mutex);
  return s.snapshot;
}

void reset_diagnostics() {
  auto& s = state();
  std::lock_guard<std::mutex> lk(s.mutex);
  s.snapshot = {};
}

bool strict_mode() {
  static bool strict = env::get_var("MLX_ANE_STRICT", 0) == 1;
  return strict;
}

bool verbose_mode() {
  static bool verbose = env::get_var("MLX_ANE_VERBOSE", 0) == 1;
  return verbose;
}

bool report_mode() {
  static bool report = env::get_var("MLX_ANE_REPORT_FALLBACKS", 0) == 1;
  return report;
}

void note_total(const Primitive& primitive, bool supported) {
  register_summary_report();
  auto& s = state();
  std::lock_guard<std::mutex> lk(s.mutex);
  s.snapshot.total_ops++;
  if (supported) {
    s.snapshot.supported_ops++;
  }
  verbose_log(primitive, supported ? "ane-candidate" : "fallback-candidate", "");
}

void note_ane_dispatch(const Primitive& primitive, bool emulated) {
  auto& s = state();
  std::lock_guard<std::mutex> lk(s.mutex);
  s.snapshot.ane_dispatches++;
  if (emulated) {
    s.snapshot.ane_emulated_dispatches++;
    verbose_log(primitive, "ane-emulated", "");
  } else {
    verbose_log(primitive, "ane", "");
  }
}

void note_gpu_fallback(const Primitive& primitive, std::string_view reason) {
  auto& s = state();
  std::lock_guard<std::mutex> lk(s.mutex);
  s.snapshot.gpu_fallbacks++;
  verbose_log(primitive, "gpu-fallback", reason);
}

void note_cpu_fallback(const Primitive& primitive, std::string_view reason) {
  auto& s = state();
  std::lock_guard<std::mutex> lk(s.mutex);
  s.snapshot.cpu_fallbacks++;
  verbose_log(primitive, "cpu-fallback", reason);
}

void note_strict_rejection(const Primitive& primitive, std::string_view reason) {
  auto& s = state();
  std::lock_guard<std::mutex> lk(s.mutex);
  s.snapshot.strict_rejections++;
  verbose_log(primitive, "strict-reject", reason);
}

void note_compile_cache_hit(const Primitive&) {
  auto& s = state();
  std::lock_guard<std::mutex> lk(s.mutex);
  s.snapshot.compile_cache_hits++;
}

void note_compile_cache_miss(const Primitive&) {
  auto& s = state();
  std::lock_guard<std::mutex> lk(s.mutex);
  s.snapshot.compile_cache_misses++;
}

void note_partition_boundary(Stream stream, const char* from, const char* to) {
  auto& s = state();
  std::lock_guard<std::mutex> lk(s.mutex);
  s.snapshot.partition_boundaries++;
  if (verbose_mode()) {
    std::cerr << "[ane::partition] stream=" << stream.index << " from=" << from
              << " to=" << to << "\n";
  }
}

} // namespace mlx::core::ane
