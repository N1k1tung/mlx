#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx::core::ane {

void new_stream(Stream stream);
void eval(array& arr);
void finalize(Stream s);
void synchronize(Stream s);

} // namespace mlx::core::ane
