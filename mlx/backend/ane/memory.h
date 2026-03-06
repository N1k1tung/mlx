#pragma once

#include <cstddef>
#include <memory>

#include "mlx/array.h"
#include "mlx/api.h"

namespace mlx::core::ane {

class MLX_API SurfaceBuffer {
 public:
  SurfaceBuffer(void* handle, size_t size);
  ~SurfaceBuffer();

  SurfaceBuffer(const SurfaceBuffer&) = delete;
  SurfaceBuffer& operator=(const SurfaceBuffer&) = delete;
  SurfaceBuffer(SurfaceBuffer&&) = delete;
  SurfaceBuffer& operator=(SurfaceBuffer&&) = delete;

  void* data();
  const void* data() const;
  size_t size() const {
    return size_;
  }
  void* handle() const {
    return handle_;
  }

 private:
  void* handle_{nullptr};
  size_t size_{0};
};

MLX_API std::shared_ptr<SurfaceBuffer> allocate_surface(size_t bytes);
MLX_API std::shared_ptr<SurfaceBuffer> wrap_array_to_surface(const array& arr);
MLX_API void unwrap_surface_to_array(const SurfaceBuffer& surface, array& arr);

} // namespace mlx::core::ane
