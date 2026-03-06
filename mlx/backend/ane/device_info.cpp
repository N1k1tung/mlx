#include "mlx/backend/ane/device_info.h"
#include "mlx/backend/gpu/device_info.h"

namespace mlx::core::ane {

bool is_available() {
  // ANE backend is currently layered on top of the Apple GPU runtime
  // infrastructure (streams, synchronization, and buffer residency).
  return gpu::is_available();
}

int device_count() {
  return is_available() ? 1 : 0;
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info(int device_index) {
  static auto info = []()
      -> std::unordered_map<std::string, std::variant<std::string, size_t>> {
    auto info = gpu::device_info(0);
    info["backend"] = std::string("ane");
    if (info.find("device_name") == info.end()) {
      info["device_name"] = std::string("Apple Neural Engine");
    }
    return info;
  }();
  static std::unordered_map<std::string, std::variant<std::string, size_t>>
      empty;

  if (!is_available() || device_index != 0) {
    return empty;
  }
  return info;
}

} // namespace mlx::core::ane
