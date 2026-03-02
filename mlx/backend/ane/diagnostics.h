// Copyright © 2026 Apple Inc.

#pragma once

#include <cstddef>
#include <string_view>

#include "mlx/api.h"
#include "mlx/stream.h"

namespace mlx::core {

class Primitive;

namespace ane {

struct MLX_API DiagnosticsSnapshot {
  size_t total_ops{0};
  size_t supported_ops{0};
  size_t ane_dispatches{0};
  size_t gpu_fallbacks{0};
  size_t cpu_fallbacks{0};
  size_t compile_cache_hits{0};
  size_t compile_cache_misses{0};
  size_t partition_boundaries{0};
};

MLX_API DiagnosticsSnapshot get_diagnostics();
MLX_API void reset_diagnostics();

bool diagnostics_mode();
bool verbose_mode();
bool report_mode();

void note_total(const Primitive& primitive, bool supported);
void note_ane_dispatch(const Primitive& primitive);
void note_gpu_fallback(const Primitive& primitive, std::string_view reason);
void note_cpu_fallback(const Primitive& primitive, std::string_view reason);
void note_compile_cache_hit(const Primitive& primitive);
void note_compile_cache_miss(const Primitive& primitive);
void note_partition_boundary(Stream stream, const char* from, const char* to);

} // namespace ane
} // namespace mlx::core
