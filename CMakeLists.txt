cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(SPHFluid LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_DEBUG \"-o0 -g3 -Wall\")
set(CMAKE_CXX_FLAGS_RELEASE \"-o3 -g0\")

find_package(CUDA QUIET REQUIRED)
set(
  CUDA_NVCC_FLAGS
  ${CUDA_NVCC_FLAGS};
  -O3 -gencode
  )

# if (NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE Debug)
# endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_BUILD_TYPE Release)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenGL)
find_package(GLEW)
find_package(glfw3)

# add_executable(run main.cpp)

cuda_add_executable(run main.cpp fluid_domain.h fluid_system.h particle.h
  render_util.h utils.h render/camera.cpp render/input.cpp render/mesh.cpp
  render/object.cpp render/objectLoader.cpp render/renderer.cpp render/scene.cpp
  render/shader.cpp render/texture.cpp render/noise/perlin.cpp render/camera.h
  render/debug.h render/lights.h render/mesh.h render/object.h
  render/objectLoader.h render/renderer.h render/scene.h render/shader.h
  render/stb_image.h render/stb_image_write.h render/texture.h
  render/tiny_obj_loader.h render/noise/perlin.h )

target_link_libraries(run ${OPENGL_LIBRARY} glfw ${GLEW_LIBRARY} assimp pthread)
