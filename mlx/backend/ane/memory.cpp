#include "mlx/backend/ane/memory.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "mlx/allocator.h"

#if defined(__APPLE__)
#include <CoreFoundation/CoreFoundation.h>
#include <IOSurface/IOSurface.h>
#include <unistd.h>
#endif

namespace mlx::core::ane {

#if defined(__APPLE__)
namespace {

size_t align_to_page(size_t size) {
  if (size == 0) {
    return 0;
  }
  auto page_size = static_cast<size_t>(::getpagesize());
  return ((size + page_size - 1) / page_size) * page_size;
}

class ScopedCFNumber {
 public:
  explicit ScopedCFNumber(size_t value) {
    value_ = value;
    number_ = CFNumberCreate(
        kCFAllocatorDefault, kCFNumberSInt64Type, static_cast<void*>(&value_));
  }
  ~ScopedCFNumber() {
    if (number_ != nullptr) {
      CFRelease(number_);
    }
  }

  CFNumberRef get() const {
    return number_;
  }

 private:
  int64_t value_{0};
  CFNumberRef number_{nullptr};
};

IOSurfaceRef create_surface(size_t bytes) {
  auto alloc_size = std::max<size_t>(align_to_page(bytes), 1);
  ScopedCFNumber width(alloc_size);
  ScopedCFNumber height(1);
  ScopedCFNumber bytes_per_row(alloc_size);
  ScopedCFNumber bytes_per_elem(1);
  ScopedCFNumber surface_size(alloc_size);

  auto props = CFDictionaryCreateMutable(
      kCFAllocatorDefault,
      0,
      &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks);
  if (props == nullptr) {
    throw std::runtime_error("[ane::memory] Failed to allocate IOSurface props");
  }

  CFDictionarySetValue(props, kIOSurfaceWidth, width.get());
  CFDictionarySetValue(props, kIOSurfaceHeight, height.get());
  CFDictionarySetValue(props, kIOSurfaceBytesPerRow, bytes_per_row.get());
  CFDictionarySetValue(props, kIOSurfaceBytesPerElement, bytes_per_elem.get());
  CFDictionarySetValue(props, kIOSurfaceAllocSize, surface_size.get());

  auto surface = IOSurfaceCreate(props);
  CFRelease(props);

  if (surface == nullptr) {
    throw std::runtime_error("[ane::memory] IOSurfaceCreate failed");
  }
  return surface;
}

IOSurfaceRef create_surface_alias(void* base, size_t bytes) {
  if (base == nullptr) {
    throw std::runtime_error("[ane::memory] Cannot alias null array data");
  }

  auto alloc_size = std::max<size_t>(bytes, 1);
  ScopedCFNumber width(alloc_size);
  ScopedCFNumber height(1);
  ScopedCFNumber bytes_per_row(alloc_size);
  ScopedCFNumber bytes_per_elem(1);
  ScopedCFNumber surface_size(alloc_size);
  ScopedCFNumber address(reinterpret_cast<uintptr_t>(base));

  auto props = CFDictionaryCreateMutable(
      kCFAllocatorDefault,
      0,
      &kCFTypeDictionaryKeyCallBacks,
      &kCFTypeDictionaryValueCallBacks);
  if (props == nullptr) {
    throw std::runtime_error("[ane::memory] Failed to allocate IOSurface alias props");
  }

  CFDictionarySetValue(props, kIOSurfaceWidth, width.get());
  CFDictionarySetValue(props, kIOSurfaceHeight, height.get());
  CFDictionarySetValue(props, kIOSurfaceBytesPerRow, bytes_per_row.get());
  CFDictionarySetValue(props, kIOSurfaceBytesPerElement, bytes_per_elem.get());
  CFDictionarySetValue(props, kIOSurfaceAllocSize, surface_size.get());
  CFDictionarySetValue(props, CFSTR("IOSurfaceAddress"), address.get());

  auto surface = IOSurfaceCreate(props);
  CFRelease(props);
  if (surface == nullptr) {
    throw std::runtime_error("[ane::memory] IOSurface alias creation failed");
  }

  IOSurfaceLock(surface, 0, nullptr);
  void* alias_base = IOSurfaceGetBaseAddress(surface);
  IOSurfaceUnlock(surface, 0, nullptr);
  if (alias_base != base) {
    CFRelease(surface);
    throw std::runtime_error("[ane::memory] IOSurface alias base mismatch");
  }
  return surface;
}

size_t surface_size(IOSurfaceRef surface) {
  return static_cast<size_t>(IOSurfaceGetAllocSize(surface));
}

} // namespace
#endif

SurfaceBuffer::SurfaceBuffer(void* handle, size_t size)
    : handle_(handle), size_(size) {}

SurfaceBuffer::~SurfaceBuffer() {
#if defined(__APPLE__)
  if (handle_ != nullptr) {
    CFRelease(static_cast<IOSurfaceRef>(handle_));
    handle_ = nullptr;
  }
#endif
}

void* SurfaceBuffer::data() {
#if defined(__APPLE__)
  if (handle_ == nullptr) {
    return nullptr;
  }
  return IOSurfaceGetBaseAddress(static_cast<IOSurfaceRef>(handle_));
#else
  return nullptr;
#endif
}

const void* SurfaceBuffer::data() const {
  return const_cast<SurfaceBuffer*>(this)->data();
}

std::shared_ptr<SurfaceBuffer> allocate_surface(size_t bytes) {
#if defined(__APPLE__)
  auto surface = create_surface(bytes);
  return std::make_shared<SurfaceBuffer>(surface, surface_size(surface));
#else
  (void)bytes;
  throw std::runtime_error("[ane::memory] IOSurface is only available on Apple platforms");
#endif
}

std::shared_ptr<SurfaceBuffer> wrap_array_to_surface(const array& arr) {
  if (!arr.flags().row_contiguous) {
    throw std::runtime_error(
        "[ane::memory] wrap_array_to_surface requires row-contiguous array");
  }
  if (arr.offset() != 0) {
    throw std::runtime_error(
        "[ane::memory] wrap_array_to_surface requires zero-offset array");
  }
  if (arr.nbytes() == 0) {
    return allocate_surface(0);
  }

#if defined(__APPLE__)
  void* base = const_cast<array&>(arr).data<char>();
  auto surface = create_surface_alias(base, arr.nbytes());
  return std::make_shared<SurfaceBuffer>(surface, surface_size(surface));
#else
  (void)arr;
  throw std::runtime_error("[ane::memory] IOSurface is only available on Apple platforms");
#endif
}

void unwrap_surface_to_array(const SurfaceBuffer& surface, array& arr) {
  if (arr.nbytes() == 0 || surface.handle() == nullptr) {
    return;
  }
  if (!arr.flags().row_contiguous) {
    throw std::runtime_error(
        "[ane::memory] unwrap_surface_to_array requires row-contiguous array");
  }
  if (arr.offset() != 0) {
    throw std::runtime_error(
        "[ane::memory] unwrap_surface_to_array requires zero-offset array");
  }
  if (surface.size() < arr.nbytes()) {
    throw std::runtime_error(
        "[ane::memory] unwrap_surface_to_array requires surface >= array bytes");
  }

#if defined(__APPLE__)
  auto surface_ref = static_cast<IOSurfaceRef>(surface.handle());
  auto* base = IOSurfaceGetBaseAddress(surface_ref);
  if (base == nullptr) {
    throw std::runtime_error("[ane::memory] surface has null base address");
  }
  CFRetain(surface_ref);
  auto wrapped = allocator::make_buffer(base, arr.nbytes());
  if (wrapped.ptr() == nullptr) {
    CFRelease(surface_ref);
    throw std::runtime_error("[ane::memory] failed to wrap surface base into array");
  }
  arr.set_data(
      wrapped,
      [surface_ref](allocator::Buffer buffer) {
        allocator::release(buffer);
        CFRelease(surface_ref);
      });
#else
  (void)surface;
  (void)arr;
  throw std::runtime_error("[ane::memory] IOSurface is only available on Apple platforms");
#endif
}

} // namespace mlx::core::ane
