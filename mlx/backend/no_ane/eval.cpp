#include <stdexcept>

#include "mlx/backend/ane/eval.h"

namespace mlx::core::ane {

void new_stream(Stream) {}

void eval(array&) {
  throw std::runtime_error("[ane::eval] ANE backend is not available");
}

void finalize(Stream) {
  throw std::runtime_error("[ane::finalize] ANE backend is not available");
}

void synchronize(Stream) {
  throw std::runtime_error("[ane::synchronize] ANE backend is not available");
}

} // namespace mlx::core::ane
