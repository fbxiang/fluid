cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(SPHFluid LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG \"-o0 -g3 -Wall\")
set(CMAKE_CXX_FLAGS_RELEASE \"-o3 -g0 -Wall\")

find_package(CUDA REQUIRED)
set(CMAKE_BUILD_TYPE DEBUG)

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
find_package(Boost COMPONENTS unit_test_framework REQUIRED)

file(GLOB_RECURSE RENDER_SRC "render/*.h" "render/*.cpp")

# cuda_add_executable(run main.cpp profiler.h profiler.cpp fluid_domain.h fluid_system.h fluid_system.cpp
#   particle.h render_util.h utils.h render/camera.cpp render/input.cpp
#   render/mesh.cpp render/object.cpp render/objectLoader.cpp render/renderer.cpp
#   render/scene.cpp render/shader.cpp render/texture.cpp render/noise/perlin.cpp
#   render/camera.h render/debug.h render/lights.h render/mesh.h render/object.h
#   render/objectLoader.h render/renderer.h render/scene.h render/shader.h
#   render/stb_image.h render/stb_image_write.h render/texture.h
#   render/tiny_obj_loader.h render/noise/perlin.h )

# cuda_add_executable(run_df divergence_free.cpp sph_render_util.h sph_common.h
#   sph_gpu.h sph_gpu.cpp sph_math.h ${RENDER_SRC})

add_executable(run main_gpu.cpp sph_common.h sph_gpu.h sph_math.h gpu_render_util.h ${RENDER_SRC} sph.cuh sph.cu)


# add_executable(run_test test/entry.cpp test/cuda.cpp sph_common.h sph.cuh sph.cu
# sph_cpu.h sph_cpu.cpp)


target_link_libraries(run ${CUDA_LIBRARIES} ${OPENGL_LIBRARY} glfw ${GLEW_LIBRARY} assimp pthread)
# target_link_libraries(run_df ${CUDA_LIBRARIES} ${OPENGL_LIBRARY} glfw ${GLEW_LIBRARY} assimp
#   pthread)

# target_link_libraries(run_test ${CUDA_LIBRARIES} ${Boost_LIBRARIES})

