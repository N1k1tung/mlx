#include "mlx/backend/ane/diagnostics.h"

namespace mlx::core::ane {

DiagnosticsSnapshot get_diagnostics() {
  return {};
}

void reset_diagnostics() {}

bool diagnostics_mode() {
  return false;
}

bool verbose_mode() {
  return false;
}

bool report_mode() {
  return false;
}

void note_total(const Primitive&, bool) {}

void note_ane_dispatch(const Primitive&) {}

void note_gpu_fallback(const Primitive&, std::string_view) {}

void note_cpu_fallback(const Primitive&, std::string_view) {}

void note_compile_cache_hit(const Primitive&) {}

void note_compile_cache_miss(const Primitive&) {}

void note_partition_boundary(Stream, const char*, const char*) {}

} // namespace mlx::core::ane
