#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "mlx/array.h"
#include "mlx/backend/ane/private_runtime.h"

namespace mlx::core::ane {

enum class DispatchStatus {
  dispatched,
  runtime_unavailable,
  dispatch_failed,
};

struct DispatchResult {
  DispatchStatus status{DispatchStatus::runtime_unavailable};
  std::string reason;

  bool executed() const {
    return status == DispatchStatus::dispatched;
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

  bool is_runtime_available();
  std::string runtime_unavailable_reason();

  DispatchResult dispatch(array& arr);
  bool pin_to_surface(array& arr);

 private:
  Runtime() = default;

  struct CompiledProgram {
    std::string key;
    std::string primitive;
    std::uintptr_t primitive_id{0};
    size_t num_inputs{0};
    size_t num_outputs{0};
    std::vector<Shape> input_shapes;
    std::vector<Dtype> input_dtypes;
    std::vector<Shape> output_shapes;
    std::vector<Dtype> output_dtypes;
    bool has_rms_eps{false};
    float rms_eps{0.0f};
    std::shared_ptr<private_runtime::Program> native_program;
    std::string native_compile_reason;
  };

  std::string make_cache_key(const array& arr) const;
  std::shared_ptr<CompiledProgram> get_or_compile(const array& arr);
  bool try_initialize_runtime();
  bool program_matches(const CompiledProgram& program, const array& arr) const;

  mutable std::mutex mutex_;
  bool runtime_checked_{false};
  bool runtime_available_{false};
  std::string runtime_unavailable_reason_{"uninitialized"};
  void* runtime_handle_{nullptr};
  std::unordered_map<std::uintptr_t, std::shared_ptr<CompiledProgram>>
      primitive_cache_;
  std::unordered_map<std::string, std::shared_ptr<CompiledProgram>>
      compile_cache_;
};

Runtime& runtime();

} // namespace mlx::core::ane
