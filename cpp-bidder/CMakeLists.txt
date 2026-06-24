cmake_minimum_required(VERSION 3.16)
project(cpp_bidder CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Dependencies installed system-wide in the Docker image.
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)
find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(prometheus-cpp CONFIG REQUIRED)

# ONNX Runtime ships as a prebuilt tarball (no cmake config), wire it by path.
set(ONNXRUNTIME_ROOT "/opt/onnxruntime" CACHE PATH "ONNX Runtime install root")

# ---- generate gRPC + protobuf C++ from the .proto contract ----
set(PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_DIR})
set(PROTO_FILE ${PROTO_DIR}/bid_service.proto)

find_program(PROTOC protoc)
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)

set(PROTO_SRCS ${GENERATED_DIR}/bid_service.pb.cc)
set(PROTO_HDRS ${GENERATED_DIR}/bid_service.pb.h)
set(GRPC_SRCS ${GENERATED_DIR}/bid_service.grpc.pb.cc)
set(GRPC_HDRS ${GENERATED_DIR}/bid_service.grpc.pb.h)

add_custom_command(
  OUTPUT ${PROTO_SRCS} ${PROTO_HDRS} ${GRPC_SRCS} ${GRPC_HDRS}
  COMMAND ${PROTOC}
  ARGS --cpp_out=${GENERATED_DIR}
       --grpc_out=${GENERATED_DIR}
       --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
       -I${PROTO_DIR}
       ${PROTO_FILE}
  DEPENDS ${PROTO_FILE}
  COMMENT "Generating C++ code from bid_service.proto")

# ---- the pieces of the server that do not need gRPC (also used by the test) ----
set(CORE_SRCS
  src/feature_store.cpp
  src/bid_optimizer.cpp
  src/model_session.cpp)

# ---- the full bid server ----
add_executable(bid_server
  src/main.cpp
  src/async_server.cpp
  src/bid_pipeline.cpp
  src/config.cpp
  src/metrics.cpp
  src/micro_batcher.cpp
  src/circuit_breaker.cpp
  src/dedup_cache.cpp
  ${CORE_SRCS}
  ${PROTO_SRCS}
  ${GRPC_SRCS})

target_include_directories(bid_server PRIVATE
  ${GENERATED_DIR}
  ${ONNXRUNTIME_ROOT}/include)
target_link_directories(bid_server PRIVATE ${ONNXRUNTIME_ROOT}/lib)

target_link_libraries(bid_server
  gRPC::grpc++
  gRPC::grpc++_reflection
  protobuf::libprotobuf
  yaml-cpp
  nlohmann_json::nlohmann_json
  prometheus-cpp::pull
  prometheus-cpp::core
  onnxruntime
  Threads::Threads)

# ---- the parity test (hot path only, no gRPC) ----
add_executable(parity_test
  test/parity_test.cpp
  ${CORE_SRCS})
target_include_directories(parity_test PRIVATE
  src
  ${ONNXRUNTIME_ROOT}/include)
target_link_directories(parity_test PRIVATE ${ONNXRUNTIME_ROOT}/lib)
target_link_libraries(parity_test
  nlohmann_json::nlohmann_json
  onnxruntime
  Threads::Threads)

# ---- end to end gRPC smoke test (client) ----
add_executable(grpc_smoke
  test/grpc_smoke.cpp
  ${PROTO_SRCS}
  ${GRPC_SRCS})
target_include_directories(grpc_smoke PRIVATE ${GENERATED_DIR})
target_link_libraries(grpc_smoke
  gRPC::grpc++
  protobuf::libprotobuf
  nlohmann_json::nlohmann_json
  Threads::Threads)

# ---- load generator / benchmark client ----
add_executable(bench
  test/bench.cpp
  ${PROTO_SRCS}
  ${GRPC_SRCS})
target_include_directories(bench PRIVATE ${GENERATED_DIR})
target_link_libraries(bench
  gRPC::grpc++
  protobuf::libprotobuf
  Threads::Threads)

# ---- in-process engine throughput benchmark (no gRPC) ----
add_executable(bench_engine
  test/bench_engine.cpp
  ${CORE_SRCS})
target_include_directories(bench_engine PRIVATE
  src
  ${ONNXRUNTIME_ROOT}/include)
target_link_directories(bench_engine PRIVATE ${ONNXRUNTIME_ROOT}/lib)
target_link_libraries(bench_engine
  nlohmann_json::nlohmann_json
  onnxruntime
  Threads::Threads)

# ---- async (open-loop) load generator ----
add_executable(bench_async
  test/bench_async.cpp
  ${PROTO_SRCS}
  ${GRPC_SRCS})
target_include_directories(bench_async PRIVATE ${GENERATED_DIR})
target_link_libraries(bench_async
  gRPC::grpc++
  protobuf::libprotobuf
  Threads::Threads)

# ---- component unit tests (no gRPC / ONNX, fast) ----
add_executable(unit_test
  test/unit_test.cpp
  src/circuit_breaker.cpp
  src/dedup_cache.cpp
  src/bid_optimizer.cpp)
target_include_directories(unit_test PRIVATE src)
target_link_libraries(unit_test Threads::Threads)

enable_testing()
add_test(NAME unit COMMAND unit_test)
add_test(NAME parity COMMAND parity_test
         ${CMAKE_CURRENT_SOURCE_DIR}/test/golden_fixture.json
         ${CMAKE_CURRENT_SOURCE_DIR}/models/feature_config.json
         ${CMAKE_CURRENT_SOURCE_DIR}/models/bid_model.onnx)
