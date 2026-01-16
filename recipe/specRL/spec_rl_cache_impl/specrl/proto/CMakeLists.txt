cmake_minimum_required(VERSION 3.14)
project(SpecRLProto CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Protobuf
find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)

# Find gRPC
pkg_check_modules(GRPC grpc++)
if(NOT GRPC_FOUND)
  find_library(GRPC_LIBRARY grpc++ PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib)
  if(GRPC_LIBRARY)
    set(GRPC_LIBRARIES ${GRPC_LIBRARY})
  else()
    message(FATAL_ERROR "gRPC library not found")
  endif()
endif()

# Create a static library for the protobuf code
# Using OBJECT library to avoid linking issues and allow direct inclusion
add_library(specrl_proto OBJECT
    rollout-cache.pb.cc
    rollout-cache.grpc.pb.cc
)

# Set position independent code (required for shared libraries that link to this)
set_target_properties(specrl_proto PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Include directories
target_include_directories(specrl_proto PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Protobuf_INCLUDE_DIRS}
    ${GRPC_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(specrl_proto PUBLIC
    ${Protobuf_LIBRARIES}
    ${GRPC_LIBRARIES}
)
