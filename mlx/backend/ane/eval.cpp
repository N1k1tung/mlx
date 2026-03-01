// Copyright © 2026 Apple Inc.

#include <stdexcept>
#include <string>
#include <typeinfo>

#include "mlx/backend/ane/eval.h"
#include "mlx/backend/cpu/eval.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core::ane {

namespace {

bool supports_ane(const Primitive& p) {
  // This list reflects the current ANE backend target set for transformer
  // inference. Unsupported ops can be routed to the GPU fallback path.
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

bool strict_mode() {
  static bool strict = env::get_var("MLX_ANE_STRICT", 0) == 1;
  return strict;
}

} // namespace

void new_stream(Stream stream) {
  // Reuse the GPU stream implementation for command queue management.
  gpu::new_stream(stream);
}

void eval(array& arr) {
  bool supported = supports_ane(arr.primitive());
  if (!supported && strict_mode()) {
    throw std::runtime_error(
        std::string("[ane::eval] Primitive not supported in strict ANE mode: ") +
        arr.primitive().name());
  }

  // TODO: route supports_ane(...) primitives to a dedicated ANE runtime.
  // For now, supported ANE ops execute through the GPU dispatch layer.
  if (supported) {
    gpu::eval(arr);
    return;
  }

  // Non-targeted ops first try the GPU path, then fall back to CPU eval.
  try {
    gpu::eval(arr);
  } catch (const std::runtime_error&) {
    cpu::eval(arr);
  }
}

void finalize(Stream s) {
  gpu::finalize(s);
}

void synchronize(Stream s) {
  gpu::synchronize(s);
}

} // namespace mlx::core::ane
