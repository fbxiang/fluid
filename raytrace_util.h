#pragma once
#include "render/optix_renderer.h"
#include "render/scene.h"
#include "scene_util.h"
#include "sph.cuh"
#include "sph_gpu.h"
#include <cuda_gl_interop.h>
#include <glm/gtx/compatibility.hpp>
#include <iostream>

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


class RaytraceUtil {
  static constexpr int MAX_FACES = 1000000;

public:
  std::shared_ptr<Scene> scene;
  std::shared_ptr<Object> fluidObj;
  std::shared_ptr<Object> diffuseObj;

  cudaGraphicsResource_t resource = 0;
  cudaGraphicsResource_t diffuseResource = 0;

  OptixRenderer *renderer;
  SPH_GPU *fluid_system;

public:
  RaytraceUtil(SPH_GPU *system, uint32_t width, uint32_t height) {
    fluid_system = system;
    scene = std::make_shared<Scene>();
    setupSponza(scene, width, height);
    // glm::vec3 size = system->fluid_domain.size;
    // glm::vec3 center = system->fluid_domain.corner + size * .5f;

    int mc_num_cells = system->get_mc_num_cells();

    printf("%d\n", std::min(mc_num_cells * 15, MAX_FACES * 3));
    // TODO: maybe optimize using EBO?
    // each cell can have 5 faces (15 vertices)
    std::shared_ptr<DynamicMesh> mesh = std::make_shared<DynamicMesh>(
        std::min(mc_num_cells * 15, MAX_FACES * 3));
    CUDA_CHECK_RETURN(cudaGraphicsGLRegisterBuffer(
        &resource, mesh->getVBO(), cudaGraphicsRegisterFlagsNone));

    std::shared_ptr<DynamicMesh> diffuseMesh =
        std::make_shared<DynamicMesh>(100000 * 12);
    CUDA_CHECK_RETURN(cudaGraphicsGLRegisterBuffer(
        &diffuseResource, diffuseMesh->getVBO(), cudaGraphicsRegisterFlagsNone));

    fluidObj = NewObject<Object>(mesh);
    fluidObj->name = "Fluid";
    fluidObj->material.type = "transparent";
    fluidObj->material.kd = {0.8, 0.9, 1.0};
    scene->addObject(fluidObj);

    diffuseObj = NewObject<Object>(diffuseMesh);
    diffuseObj->name = "Diffuse";
    diffuseObj->material.type = "foam";
    diffuseObj->material.kd = { 1.0, 1.0, 1.0 };
    scene->addObject(diffuseObj);

    size_t free, used;
    cudaMemGetInfo(&free, &used);
    printf("free: %ld; used: %ld\n", free, used);

    scene->addDirectionalLight({glm::vec3(0, -1, -1), glm::vec3(1, 1, 0.5)});
    renderer = new OptixRenderer(width, height);
    renderer->init();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glEnable(GL_FRAMEBUFFER_SRGB_EXT);
  }

  void invalidate_camera() { renderer->invalidateCamera(); }

  void render() {
    renderer->renderScene(scene);

    float *d_vertex_pointer;
    size_t size;
    CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &resource));
    CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer(
        (void **)&d_vertex_pointer, &size, resource));
    int num_faces = 0;
    fluid_system->update_mesh();
    fluid_system->update_faces(d_vertex_pointer, &num_faces, MAX_FACES);
    std::dynamic_pointer_cast<DynamicMesh>(fluidObj->getMesh())
        ->setVertexCount(num_faces * 3);
    CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &resource));

    d_vertex_pointer = nullptr; size = 0;
    CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &diffuseResource));
    CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer(
        (void **)&d_vertex_pointer, &size, diffuseResource));

    sph::visual::update_visual_faces(d_vertex_pointer);
    std::dynamic_pointer_cast<DynamicMesh>(diffuseObj->getMesh())
        ->setVertexCount(sph::visual::get_num_visual_particles() * 12);

    CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &diffuseResource));
  }

  void renderToFile(const std::string &dir, uint32_t i = 0) {
    renderer->renderCurrentToFile(dir + "/sph_" + std::to_string(i) + ".png");
  }
};
