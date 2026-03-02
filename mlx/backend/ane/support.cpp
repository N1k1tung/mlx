// Copyright © 2026 Apple Inc.

#include "mlx/backend/ane/support.h"

#include <typeinfo>

#include "mlx/primitives.h"

namespace mlx::core::ane {

bool supports_ane(const Primitive& p) {
  // Keep allowlist aligned with currently implemented native ANE runtime ops.
  return typeid(p) == typeid(Add) || typeid(p) == typeid(Subtract) ||
      typeid(p) == typeid(Multiply) || typeid(p) == typeid(Divide) ||
      typeid(p) == typeid(Matmul) || typeid(p) == typeid(Softmax);
}

} // namespace mlx::core::ane
