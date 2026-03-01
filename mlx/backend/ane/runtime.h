// Copyright © 2026 Apple Inc.

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "mlx/array.h"

namespace mlx::core::ane {

enum class DispatchStatus {
  dispatched,
  dispatched_emulated,
  runtime_unavailable,
  dispatch_failed,
};

struct DispatchResult {
  DispatchStatus status{DispatchStatus::runtime_unavailable};
  std::string reason;

  bool executed() const {
    return status == DispatchStatus::dispatched ||
        status == DispatchStatus::dispatched_emulated;
  }

  bool emulated() const {
    return status == DispatchStatus::dispatched_emulated;
  }
};

class Runtime {
 public:
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  static Runtime& instance();

  void new_stream(Stream stream);
  void finalize(Stream stream);
  void synchronize(Stream stream);

  DispatchResult dispatch(array& arr);

 private:
  Runtime() = default;

  struct CompiledProgram {
    std::string key;
    std::string primitive;
    size_t num_inputs{0};
    size_t num_outputs{0};
  };

  std::string make_cache_key(const array& arr) const;
  std::shared_ptr<CompiledProgram> get_or_compile(const array& arr);
  bool should_use_iosurface() const;
  bool emulation_enabled() const;
  bool try_initialize_runtime();

  mutable std::mutex mutex_;
  bool runtime_checked_{false};
  bool runtime_available_{false};
  void* runtime_handle_{nullptr};
  std::unordered_map<std::string, std::shared_ptr<CompiledProgram>>
      compile_cache_;
};

Runtime& runtime();

} // namespace mlx::core::ane
