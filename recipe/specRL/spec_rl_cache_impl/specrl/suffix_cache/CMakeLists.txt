cmake_minimum_required(VERSION 3.14)
project(SuffixCache CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  message(STATUS "Setting build type to 'Release' as none was specified.")
endif()
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

# Optional aggressive optimizations (slower build, faster runtime)
if(ENABLE_AGGRESSIVE_OPTS)
  message(STATUS "Enabling aggressive optimizations (-O3 -march=native -flto)")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -ffast-math -flto -funroll-loops")
endif()

# Disable pybind11's FindPython to avoid hang - we'll configure Python manually
set(PYBIND11_FINDPYTHON OFF)

# Manually configure Python for pybind11
if(PYTHON_EXECUTABLE)
  message(STATUS "Manually configuring Python for pybind11: ${PYTHON_EXECUTABLE}")
  
  # Get Python version and paths by running Python
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    OUTPUT_VARIABLE PYTHON_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
    OUTPUT_VARIABLE PYTHON_LIBRARY_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  
  # Set variables that pybind11 expects
  set(PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE} CACHE FILEPATH "Python executable")
  set(PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIR} CACHE PATH "Python include dir")
  set(PYTHON_LIBRARIES ${PYTHON_LIBRARY_DIR} CACHE PATH "Python library dir")
  set(PYTHON_VERSION_STRING ${PYTHON_VERSION} CACHE STRING "Python version")
  
  message(STATUS "  Python version: ${PYTHON_VERSION}")
  message(STATUS "  Python include: ${PYTHON_INCLUDE_DIR}")
endif()

# Find required packages
message(STATUS "Finding pybind11...")
# Disable pybind11's LTO test which can hang in some environments
set(PYBIND11_TEST_LTO OFF CACHE BOOL "" FORCE)
if(pybind11_DIR)
  message(STATUS "  Using provided pybind11_DIR: ${pybind11_DIR}")
  find_package(pybind11 CONFIG REQUIRED NO_DEFAULT_PATH PATHS ${pybind11_DIR})
else()
  find_package(pybind11 REQUIRED CONFIG)
endif()
message(STATUS "  Found pybind11 ${pybind11_VERSION}")

message(STATUS "Finding PkgConfig...")
find_package(PkgConfig REQUIRED)

message(STATUS "Finding Threads...")
find_package(Threads REQUIRED)

# Find Protobuf
message(STATUS "Finding Protobuf...")
find_package(Protobuf REQUIRED)
message(STATUS "  Protobuf version: ${Protobuf_VERSION}")

# Find gRPC - try pkg-config first
message(STATUS "Finding gRPC...")
pkg_check_modules(GRPC QUIET grpc++)
pkg_check_modules(GRPC_REFLECTION QUIET grpc++_reflection)

# Always try to find the reflection library directly as backup
find_library(GRPC_REFLECTION_LIB grpc++_reflection 
  PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib /lib/x86_64-linux-gnu
  NO_DEFAULT_PATH)
find_library(GRPC_REFLECTION_LIB grpc++_reflection)

if(NOT GRPC_FOUND)
  message(STATUS "  gRPC not found via pkg-config, searching manually...")
  # Fallback: try to find libraries directly
  find_library(GRPC_LIBRARY grpc++ 
    PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib /lib/x86_64-linux-gnu
    NO_DEFAULT_PATH)
  find_library(GRPC_LIBRARY grpc++)
  
  find_library(GRPC_CORE_LIBRARY grpc 
    PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib /lib/x86_64-linux-gnu
    NO_DEFAULT_PATH)
  find_library(GRPC_CORE_LIBRARY grpc)
  
  if(GRPC_LIBRARY)
    set(GRPC_LIBRARIES ${GRPC_LIBRARY} ${GRPC_CORE_LIBRARY})
    message(STATUS "  Found gRPC library: ${GRPC_LIBRARY}")
  else()
    message(FATAL_ERROR "gRPC library not found")
  endif()
else()
  message(STATUS "  Found gRPC via pkg-config")
endif()

# Set reflection libraries - prefer direct find result
if(GRPC_REFLECTION_LIB)
  set(GRPC_REFLECTION_LIBS ${GRPC_REFLECTION_LIB})
  message(STATUS "  Found gRPC reflection library")
elseif(GRPC_REFLECTION_LIBRARIES)
  set(GRPC_REFLECTION_LIBS ${GRPC_REFLECTION_LIBRARIES})
  message(STATUS "  Found gRPC reflection via pkg-config")
else()
  message(WARNING "  gRPC reflection library not found, some features may not work")
  set(GRPC_REFLECTION_LIBS "")
endif()

# Find Boost - pkg-config doesn't work for boost, use find_package with optimizations
message(STATUS "Finding Boost...")
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)  # Continue using FindBoost.cmake module
endif()
set(Boost_NO_BOOST_CMAKE ON)  # Don't search for BoostConfig.cmake (slow)
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost REQUIRED COMPONENTS system thread)
set(BOOST_LIBRARIES ${Boost_LIBRARIES})
set(BOOST_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})
message(STATUS "  Found Boost ${Boost_VERSION}")

# Find xxHash
message(STATUS "Finding xxHash...")
pkg_check_modules(XXHASH REQUIRED libxxhash)
message(STATUS "  Found xxHash")

message(STATUS "Configuring build...")
# Add include directories
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${GRPC_INCLUDE_DIRS})
include_directories(${BOOST_INCLUDE_DIRS})
include_directories(${XXHASH_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
# Include shared proto generated files
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../proto)

# Build the shared proto library first
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../proto ${CMAKE_CURRENT_BINARY_DIR}/proto)

# Create the pybind11 module (include proto object files)
pybind11_add_module(_C
    pybind.cc
    rollout_cache_server.cc
    suffix_cache.cc
    suffix_tree.cc
    $<TARGET_OBJECTS:specrl_proto>
)

# Set compile options for position independent code
set_target_properties(_C PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Link libraries - proto dependencies are inherited from specrl_proto
target_link_libraries(_C PRIVATE
  ${Protobuf_LIBRARIES}
  ${GRPC_LIBRARIES}
  ${GRPC_REFLECTION_LIBS}
  ${BOOST_LIBRARIES}
  ${XXHASH_LIBRARIES}
  Threads::Threads
)

# On Linux, link rt library and OpenMP
if(UNIX AND NOT APPLE)
  target_link_libraries(_C PRIVATE rt)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(_C PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(_C PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
