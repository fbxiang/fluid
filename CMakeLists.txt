cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(SPHFluid LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "-o0 -g3 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-o3 -g0 -Wall")

find_package(CUDA REQUIRED)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_DEBUG})

set(
  CUDA_NVCC_FLAGS
  ${CUDA_NVCC_FLAGS};
  -O3 -std=c++11
  )

# if (NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE Debug)
# endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenGL)
find_package(GLEW)
find_package(glfw3)

file(GLOB_RECURSE RENDER_SRC "render/*.h" "render/*.cpp")

include_directories("/home/fx/NVIDIA-OptiX-SDK-6.0.0-linux64/include")
link_directories("/home/fx/NVIDIA-OptiX-SDK-6.0.0-linux64/lib64")

add_executable(run main_gpu.cpp sph_common.h sph_gpu.h sph_math.h gpu_render_util.h ${RENDER_SRC} sph.cuh sph.cu marching_cube_table.h profiler.h profiler.cpp)

target_link_libraries(run ${CUDA_LIBRARIES} ${OPENGL_LIBRARY} glfw ${GLEW_LIBRARY} assimp pthread optix)

file(GLOB_RECURSE PTX_SRC "cuda/*.cu")

cuda_compile_ptx(
  cuda_ptx_files
  ${PTX_SRC}
)

add_custom_target(ptx ALL
    DEPENDS ${cuda_ptx_files} ${PTX_SRC}
    SOURCES ${PTX_SRC})

add_dependencies(run ptx)
