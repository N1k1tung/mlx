// Copyright © 2026 Apple Inc.

#include "doctest/doctest.h"

#include "mlx/backend/ane/diagnostics.h"
#include "mlx/mlx.h"

using namespace mlx::core;

namespace {

struct DeviceGuard {
  explicit DeviceGuard(Device device) : previous(default_device()) {
    set_default_device(device);
  }
  ~DeviceGuard() {
    try {
      set_default_device(previous);
    } catch (...) {
    }
  }
  Device previous;
};

} // namespace

TEST_CASE("ane diagnostics track fallback for unsupported ops") {
  if (!is_available(Device::ane)) {
    return;
  }

  DeviceGuard guard(Device::ane);
  ane::reset_diagnostics();

  auto x = sin(array(1.0f));
  eval(x);

  auto stats = ane::get_diagnostics();
  CHECK_GE(stats.total_ops, 1);
  CHECK_GE(stats.gpu_fallbacks + stats.cpu_fallbacks, 1);
}

TEST_CASE("ane diagnostics track compile cache hits") {
  if (!is_available(Device::ane)) {
    return;
  }

  DeviceGuard guard(Device::ane);
  ane::reset_diagnostics();

  {
    auto a = ones({2, 2});
    auto b = ones({2, 2});
    auto out = add(a, b);
    eval(out);
  }
  {
    auto a = ones({2, 2});
    auto b = ones({2, 2});
    auto out = add(a, b);
    eval(out);
  }

  auto stats = ane::get_diagnostics();
  CHECK_GE(stats.compile_cache_misses, 1);
  CHECK_GE(stats.compile_cache_hits, 1);
  CHECK_GE(stats.ane_dispatches, 2);
}
