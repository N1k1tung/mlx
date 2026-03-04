// Copyright © 2026 Apple Inc.

#include "mlx/backend/ane/support.h"

#include <typeinfo>

#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

namespace mlx::core::ane {

bool is_metadata_fastpath_primitive(const Primitive& p) {
  return typeid(p) == typeid(Reshape) ||
      typeid(p) == typeid(ExpandDims) ||
      typeid(p) == typeid(Squeeze) ||
      typeid(p) == typeid(Transpose) ||
      typeid(p) == typeid(Slice) ||
      typeid(p) == typeid(Contiguous);
}

bool supports_ane(const Primitive& p) {
  // Keep allowlist aligned with concrete MIL generation in private_runtime.mm.
  return typeid(p) == typeid(Add) || typeid(p) == typeid(Subtract) ||
      typeid(p) == typeid(Multiply) || typeid(p) == typeid(Divide) ||
      typeid(p) == typeid(Matmul) || typeid(p) == typeid(Softmax) ||
      typeid(p) == typeid(Reshape) ||
      typeid(p) == typeid(ExpandDims) || typeid(p) == typeid(Squeeze) ||
      typeid(p) == typeid(Transpose) || typeid(p) == typeid(Concatenate) ||
      typeid(p) == typeid(Slice) || typeid(p) == typeid(Sigmoid) ||
      typeid(p) == typeid(Contiguous) || typeid(p) == typeid(fast::RMSNorm);
}

} // namespace mlx::core::ane
