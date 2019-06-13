#pragma once
#include "render/renderer.h"
#include "render/scene.h"
#include "scene_util.h"
#include "sph.cuh"
#include "sph_gpu.h"
#include <cuda_gl_interop.h>
#include <glm/gtx/compatibility.hpp>
#include <iomanip>
#include <iostream>
#include <sstream>

#define N_THREADS 1024
#define CUDA_CHECK_RETURN(value)                                               \
  {                                                                            \
    cudaError_t _m_cudaStat = value;                                           \
    if (_m_cudaStat != cudaSuccess) {                                          \
      fprintf(stderr, "Error %s at line %d in file %s\n",                      \
              cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);            \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }


class GPURenderUtil {
  static constexpr int MAX_FACES = 100000;

public:
  std::shared_ptr<Scene> scene;
  std::shared_ptr<Object> domain;
  std::vector<std::shared_ptr<Object>> particles;
  std::shared_ptr<Object> fluidObj;
  std::shared_ptr<Object> visualObj;
  cudaGraphicsResource_t resource = 0;
  cudaGraphicsResource_t visual_resource = 0;

  Renderer *renderer;
  SPH_GPU *fluid_system;

public:
  GPURenderUtil(SPH_GPU *system, uint32_t width, uint32_t height) {
    fluid_system = system;
    scene = std::make_shared<Scene>();
    setupEmpty(scene, width, height);
    // setupSponza(scene, width, height);

    // int mc_num_cells = system->get_mc_num_cells();

    // std::shared_ptr<DynamicMesh> mesh =
    //     std::make_shared<DynamicMesh>(std::min(mc_num_cells * 15, 3 * MAX_FACES));
    // CUDA_CHECK_RETURN(cudaGraphicsGLRegisterBuffer(
    //     &resource, mesh->getVBO(), cudaGraphicsRegisterFlagsNone));

    // fluidObj = NewObject<Object>(mesh);
    // fluidObj->name = "Fluid";
    // scene->addObject(fluidObj);

    auto mesh = std::make_shared<DynamicPointMesh>(100000);
    CUDA_CHECK_RETURN(cudaGraphicsGLRegisterBuffer(
        &resource, mesh->getVBO(), cudaGraphicsRegisterFlagsNone));

    auto visual_mesh = std::make_shared<DynamicPointMesh>(100000);
    CUDA_CHECK_RETURN(cudaGraphicsGLRegisterBuffer(
        &visual_resource, mesh->getVBO(), cudaGraphicsRegisterFlagsNone));

    fluidObj = NewObject<Object>(mesh);
    fluidObj->name = "Fluid";
    scene->addObject(fluidObj);

    auto pointShader = new Shader("/home/fx/glsl-files/point.vsh",
                                  "/home/fx/glsl-files/point.fsh");
    fluidObj->shader = pointShader;

    visualObj = NewObject<Object>(visual_mesh);
    scene->addObject(visualObj);
    visualObj->shader = pointShader;

    renderer = new Renderer(width, height);
    renderer->init();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glEnable(GL_FRAMEBUFFER_SRGB_EXT);
  }

  void render_debug() {
    update_debug_particles();
    renderer->renderScene(scene);
  }

  void render() {

    float *d_vertex_pointer;
    size_t size;
    CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &resource));
    CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer(
        (void **)&d_vertex_pointer, &size, resource));
    // int num_faces = 0;
    // fluid_system->update_mesh();
    // fluid_system->update_faces(d_vertex_pointer, &num_faces, MAX_FACES);
    // std::dynamic_pointer_cast<DynamicMesh>(fluidObj->getMesh())
    //     ->setVertexCount(num_faces * 3);

    std::dynamic_pointer_cast<DynamicPointMesh>(fluidObj->getMesh())
        ->setVertexCount(sph::get_num_particles());
    sph::update_debug_points(d_vertex_pointer);

    CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &resource));

    CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &visual_resource));
    CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer(
        (void **)&d_vertex_pointer, &size, visual_resource));

    sph::visual::update_debug_points(d_vertex_pointer);
    std::dynamic_pointer_cast<DynamicPointMesh>(visualObj->getMesh())
        ->setVertexCount(sph::visual::get_num_visual_particles());

    CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &visual_resource));
    renderer->renderScene(scene);
  }

  void renderToFile(const std::string& directory, uint32_t i = 0) {
    std::stringstream ss;
    ss << std::setw(10) << std::setfill('0') << i;
    std::string s = ss.str();
    renderer->renderSceneToFile(scene, directory + "/sph_" + s + ".png");
  }

  void init_debug_particles() {
    std::vector<glm::vec3> positions = fluid_system->get_particles();
    for (uint32_t i = 0; i < positions.size(); ++i) {
      auto p = NewSphere();
      p->scale = glm::vec3(fluid_system->solver_params.particle_size / 2);
      p->material.kd = {0, 1, 1};
      scene->addObject(p);
      particles.push_back(p);
    }
  }

  void update_debug_particles() {
    std::vector<glm::vec3> positions = fluid_system->get_particles();
    for (uint32_t i = 0; i < particles.size(); ++i) {
      particles[i]->position = positions[i];
    }
  }

  void invalidate_camera() {}
};
