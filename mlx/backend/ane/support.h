// Copyright © 2026 Apple Inc.

#pragma once

namespace mlx::core {

class Primitive;

namespace ane {

bool supports_ane(const Primitive& p);
bool is_metadata_fastpath_primitive(const Primitive& p);

} // namespace ane
} // namespace mlx::core
