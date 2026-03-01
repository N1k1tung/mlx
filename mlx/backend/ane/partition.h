// Copyright © 2026 Apple Inc.

#pragma once

#include "mlx/array.h"

namespace mlx::core::ane {

enum class Route {
  ane,
  gpu,
  cpu,
};

struct RouteDecision {
  Route route;
  bool supported;
  const char* reason;
};

const char* route_name(Route route);
RouteDecision decide_route(const array& arr);
void track_route_boundary(Stream stream, Route route);

} // namespace mlx::core::ane
