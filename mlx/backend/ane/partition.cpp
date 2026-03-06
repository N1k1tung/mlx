#include "mlx/backend/ane/partition.h"

#include <mutex>
#include <unordered_map>

#include "mlx/backend/ane/diagnostics.h"
#include "mlx/backend/ane/support.h"
#include "mlx/utils.h"

namespace mlx::core::ane {

namespace {

std::unordered_map<int, Route>& route_by_stream() {
  static std::unordered_map<int, Route> routes;
  return routes;
}

std::mutex& route_mutex() {
  static std::mutex mtx;
  return mtx;
}

} // namespace

const char* route_name(Route route) {
  switch (route) {
    case Route::ane:
      return "ane";
    case Route::gpu:
      return "gpu";
    case Route::cpu:
      return "cpu";
  }
  return "unknown";
}

RouteDecision decide_route(const array& arr) {
  if (!supports_ane(arr.primitive())) {
    return {Route::gpu, false, "unsupported-op"};
  }
  if (!supports_ane(arr)) {
    return {Route::gpu, false, "unsupported-constraints"};
  }
  return {Route::ane, true, "supported"};
}

void track_route_boundary(Stream stream, Route route) {
  if (!diagnostics_mode()) {
    return;
  }
  std::lock_guard<std::mutex> lk(route_mutex());
  auto& routes = route_by_stream();
  auto it = routes.find(stream.index);
  if (it == routes.end()) {
    routes.emplace(stream.index, route);
    return;
  }
  if (it->second != route) {
    note_partition_boundary(stream, route_name(it->second), route_name(route));
    it->second = route;
  }
}

} // namespace mlx::core::ane
