#pragma once

#include <memory>
#include <string>

#include "mlx/array.h"

namespace mlx::core::ane::private_runtime {

struct Program;

bool available(std::string* reason = nullptr);
std::shared_ptr<Program> compile(const array& arr, std::string* reason = nullptr);
bool dispatch_fastpath(array& arr, std::string* reason = nullptr);
bool dispatch(Program& program, array& arr, std::string* reason = nullptr);
bool pin_to_surface(array& arr, std::string* reason = nullptr);

} // namespace mlx::core::ane::private_runtime
