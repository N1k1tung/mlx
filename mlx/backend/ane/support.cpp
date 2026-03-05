// Copyright © 2026 Apple Inc.

#include "mlx/backend/ane/support.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <typeinfo>
#include <iostream>

#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

namespace mlx::core::ane {

namespace {

bool dtype_supported_for_runtime(Dtype dtype) {
  return dtype == float16 || dtype == float32;
}

bool io_layout_supported_for_runtime(const array& arr) {
  return arr.flags().row_contiguous;
}

bool normalize_axis(int axis, int rank, int& out_axis) {
  if (rank <= 0) {
    return false;
  }
  int ax = axis;
  if (ax < 0) {
    ax += rank;
  }
  if (ax < 0 || ax >= rank) {
    return false;
  }
  out_axis = ax;
  return true;
}

bool axis_supported_for_concat(const Shape& shape, int axis) {
  int normalized_axis = 0;
  const int rank = static_cast<int>(shape.size());
  if (!normalize_axis(axis, rank, normalized_axis)) {
    return false;
  }
  if (rank <= 4) {
    return true;
  }
  // Private runtime collapses leading dims into a single batch dim for rank > 4.
  return normalized_axis > rank - 4;
}

bool binary_runtime_supported(const array& arr) {
  const auto& inputs = arr.inputs();
  if (inputs.size() != 2) {
    return false;
  }
  // Keep route precheck aligned with private_runtime.mm conservative binary gate.
  if (
      inputs[0].dtype() != float16 || inputs[1].dtype() != float16 ||
      arr.dtype() != float16) {
    return false;
  }
  return inputs[0].shape() == inputs[1].shape() &&
      inputs[0].shape() == arr.shape();
}

bool transpose_runtime_supported(const array& arr) {
  const auto* transpose = dynamic_cast<const Transpose*>(&arr.primitive());
  if (transpose == nullptr) {
    return false;
  }
  const auto& inputs = arr.inputs();
  if (inputs.size() != 1) {
    return false;
  }
  const auto& in_shape = inputs[0].shape();
  if (in_shape.size() > 4) {
    return false;
  }
  auto axes = transpose->state();
  if (axes.size() != in_shape.size()) {
    return false;
  }
  std::array<bool, 4> used = {false, false, false, false};
  const size_t shift = 4 - in_shape.size();
  for (size_t i = 0; i < shift; ++i) {
    used[i] = true;
  }
  for (size_t i = 0; i < axes.size(); ++i) {
    int normalized_axis = 0;
    if (!normalize_axis(axes[i], static_cast<int>(in_shape.size()), normalized_axis)) {
      return false;
    }
    const int mapped = normalized_axis + static_cast<int>(shift);
    if (used[mapped]) {
      return false;
    }
    used[mapped] = true;
  }
  return true;
}

bool concatenate_runtime_supported(const array& arr) {
  const auto* concat = dynamic_cast<const Concatenate*>(&arr.primitive());
  if (concat == nullptr) {
    return false;
  }
  const auto& inputs = arr.inputs();
  if (inputs.size() < 2) {
    return false;
  }
  if (!axis_supported_for_concat(inputs[0].shape(), concat->state())) {
    return false;
  }
  const Dtype dtype0 = inputs[0].dtype();
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (inputs[i].dtype() != dtype0) {
      return false;
    }
  }
  return true;
}

bool slice_runtime_supported(const array& arr) {
  const auto* slice = dynamic_cast<const Slice*>(&arr.primitive());
  if (slice == nullptr) {
    return false;
  }
  const auto& inputs = arr.inputs();
  if (inputs.size() != 1) {
    return false;
  }
  const auto& in_shape = inputs[0].shape();
  if (in_shape.size() > 4) {
    return false;
  }
  auto [start_indices, end_indices, strides] = slice->state();
  if (
      start_indices.size() != in_shape.size() ||
      end_indices.size() != in_shape.size() ||
      strides.size() != in_shape.size()) {
    return false;
  }
  for (size_t i = 0; i < in_shape.size(); ++i) {
    if (strides[i] != 1) {
      return false;
    }
    int64_t dim = in_shape[i];
    int64_t s = start_indices[i];
    int64_t e = end_indices[i];
    if (s < 0) {
      s += dim;
    }
    if (e < 0) {
      e += dim;
    }
    s = std::max<int64_t>(0, std::min<int64_t>(s, dim));
    e = std::max<int64_t>(0, std::min<int64_t>(e, dim));
    if (e < s) {
      return false;
    }
  }
  return true;
}

bool rmsnorm_runtime_supported(const array& arr) {
  const auto* rms = dynamic_cast<const fast::RMSNorm*>(&arr.primitive());
  if (rms == nullptr) {
    return false;
  }
  const auto& inputs = arr.inputs();
  if (inputs.size() != 2) {
    return false;
  }
  (void)rms;
  return inputs[1].ndim() <= 1;
}

} // namespace

bool is_metadata_fastpath_primitive(const Primitive& p) {
  return typeid(p) == typeid(Reshape) ||
      typeid(p) == typeid(ExpandDims) ||
      typeid(p) == typeid(Squeeze) ||
      typeid(p) == typeid(Transpose) ||
      typeid(p) == typeid(Slice) ||
      typeid(p) == typeid(Contiguous) ||
      typeid(p) == typeid(Flatten) ||
      typeid(p) == typeid(Unflatten);
}

bool is_view_only_fastpath_primitive(const Primitive& p) {
  return typeid(p) == typeid(ExpandDims) ||
      typeid(p) == typeid(Squeeze) ||
      typeid(p) == typeid(Transpose) ||
      typeid(p) == typeid(Slice);
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
      typeid(p) == typeid(Contiguous) || typeid(p) == typeid(Flatten) ||
      typeid(p) == typeid(Unflatten) || typeid(p) == typeid(fast::RMSNorm) ||
      is_compiled_sigmoid_multiply_primitive(p);
}

bool supports_ane(const array& arr) {
  const auto& primitive = arr.primitive();
  if (!supports_ane(primitive)) {
    return false;
  }

  // Metadata fastpath primitives execute through private_runtime::dispatch_fastpath
  // (unary->eval_cpu) and do not require ANE runtime IO layout constraints.
  if (is_metadata_fastpath_primitive(primitive)) {
    return arr.inputs().size() == 1;
  }

  auto outputs = arr.outputs();
  if (outputs.size() != 1) {
    return false;
  }
  if (!dtype_supported_for_runtime(arr.dtype()) || !io_layout_supported_for_runtime(arr)) {
    return false;
  }
  const auto& inputs = arr.inputs();
  for (const auto& in : inputs) {
    if (
        !dtype_supported_for_runtime(in.dtype()) ||
        !io_layout_supported_for_runtime(in)) {
      return false;
    }
  }

  if (
      typeid(primitive) == typeid(Add) || typeid(primitive) == typeid(Subtract) ||
      typeid(primitive) == typeid(Multiply) || typeid(primitive) == typeid(Divide) ||
      is_compiled_sigmoid_multiply_primitive(primitive)) {
    return binary_runtime_supported(arr);
  }
  if (typeid(primitive) == typeid(Sigmoid) || typeid(primitive) == typeid(Softmax)) {
    return inputs.size() == 1;
  }
  if (typeid(primitive) == typeid(Matmul)) {
    return inputs.size() == 2;
  }
  if (
      typeid(primitive) == typeid(Reshape) ||
      typeid(primitive) == typeid(ExpandDims) ||
      typeid(primitive) == typeid(Squeeze) ||
      typeid(primitive) == typeid(Contiguous) ||
      typeid(primitive) == typeid(Flatten) ||
      typeid(primitive) == typeid(Unflatten)) {
    return inputs.size() == 1;
  }
  if (typeid(primitive) == typeid(Transpose)) {
    return transpose_runtime_supported(arr);
  }
  if (typeid(primitive) == typeid(Concatenate)) {
    return concatenate_runtime_supported(arr);
  }
  if (typeid(primitive) == typeid(Slice)) {
    return slice_runtime_supported(arr);
  }
  if (dynamic_cast<const fast::RMSNorm*>(&primitive) != nullptr) {
    return rmsnorm_runtime_supported(arr);
  }

  return false;
}

bool is_compiled_sigmoid_multiply_primitive(const Primitive& p) {
    const char* name = p.name();
    if (name == nullptr) {
        return false;
    }

    return std::strcmp(name, "CompiledSigmoidMultiply") == 0;
}

} // namespace mlx::core::ane
