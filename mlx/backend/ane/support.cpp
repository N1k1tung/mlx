// Copyright © 2026 Apple Inc.

#include "mlx/backend/ane/support.h"

#include <typeinfo>

#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

namespace mlx::core::ane {

bool supports_ane(const Primitive& p) {
  // Targeted op-set for transformer inference paths on ANE.
  return typeid(p) == typeid(Add) || typeid(p) == typeid(AddMM) ||
      typeid(p) == typeid(AsType) || typeid(p) == typeid(Broadcast) ||
      typeid(p) == typeid(BroadcastAxes) ||
      typeid(p) == typeid(Concatenate) || typeid(p) == typeid(Contiguous) ||
      typeid(p) == typeid(Divide) || typeid(p) == typeid(ExpandDims) ||
      typeid(p) == typeid(Gather) || typeid(p) == typeid(GatherAxis) ||
      typeid(p) == typeid(Matmul) || typeid(p) == typeid(Multiply) ||
      typeid(p) == typeid(Reshape) || typeid(p) == typeid(Reduce) ||
      typeid(p) == typeid(Slice) || typeid(p) == typeid(SliceUpdate) ||
      typeid(p) == typeid(DynamicSlice) ||
      typeid(p) == typeid(DynamicSliceUpdate) || typeid(p) == typeid(Softmax) ||
      typeid(p) == typeid(Squeeze) || typeid(p) == typeid(Subtract) ||
      typeid(p) == typeid(Transpose) || typeid(p) == typeid(Compiled) ||
      typeid(p) == typeid(fast::LayerNorm) ||
      typeid(p) == typeid(fast::RMSNorm) || typeid(p) == typeid(fast::RoPE) ||
      typeid(p) == typeid(fast::ScaledDotProductAttention);
}

} // namespace mlx::core::ane
