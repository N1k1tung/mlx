// Copyright © 2026 Apple Inc.

#include <stdexcept>
#include <string>

#include "mlx/backend/ane/diagnostics.h"
#include "mlx/backend/ane/eval.h"
#include "mlx/backend/ane/partition.h"
#include "mlx/backend/ane/runtime.h"
#include "mlx/backend/ane/support.h"
#include "mlx/backend/cpu/eval.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/primitives.h"

namespace mlx::core::ane {

void new_stream(Stream stream) {
  runtime().new_stream(stream);
}

void eval(array& arr) {
  auto& primitive = arr.primitive();
  auto decision = decide_route(arr);
  note_total(primitive, decision.supported);
  track_route_boundary(primitive.stream(), decision.route);

  if (!decision.supported && strict_mode()) {
    note_strict_rejection(primitive, decision.reason);
    throw std::runtime_error(
        std::string("[ane::eval] Primitive not supported in strict ANE mode: ") +
        primitive.name());
  }

  if (decision.route == Route::ane) {
    auto result = runtime().dispatch(arr);
    if (result.executed()) {
      note_ane_dispatch(primitive, result.emulated());
      return;
    }
    note_gpu_fallback(primitive, result.reason);
  } else {
    note_gpu_fallback(primitive, decision.reason);
  }

  // Route to GPU first and then CPU as the terminal fallback.
  try {
    gpu::eval(arr);
    return;
  } catch (const std::runtime_error&) {
    note_cpu_fallback(primitive, "gpu-eval-failed");
    cpu::eval(arr);
  }
}

void finalize(Stream s) {
  runtime().finalize(s);
}

void synchronize(Stream s) {
  runtime().synchronize(s);
}

} // namespace mlx::core::ane
