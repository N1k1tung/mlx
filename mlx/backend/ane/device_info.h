#pragma once

#include <string>
#include <unordered_map>
#include <variant>

#include "mlx/api.h"

namespace mlx::core::ane {

MLX_API bool is_available();

/**
 * Get the number of available ANE devices.
 */
MLX_API int device_count();

/**
 * Get information about an ANE device.
 *
 * Returns a map of device properties.
 */
MLX_API const
    std::unordered_map<std::string, std::variant<std::string, size_t>>&
    device_info(int device_index = 0);

} // namespace mlx::core::ane
