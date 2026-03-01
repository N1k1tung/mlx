target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/device_info.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/diagnostics.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/eval.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/memory.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/partition.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/runtime.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/support.cpp)
