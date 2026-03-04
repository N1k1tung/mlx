// Copyright © 2026 Apple Inc.

#pragma once

namespace mlx::core {

class Primitive;
class array;

namespace ane {

bool supports_ane(const Primitive& p);
bool supports_ane(const array& arr);
bool is_metadata_fastpath_primitive(const Primitive& p);
bool is_view_only_fastpath_primitive(const Primitive& p);

} // namespace ane
} // namespace mlx::core
