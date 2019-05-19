#pragma once
#include "sph_gpu.h"
#include "render/optix_renderer.h"
#include "render/scene.h"
#include <glm/gtx/compatibility.hpp>
#include <iostream>
#include "sph.cuh"
#include "scene_util.h"
#include <cuda_gl_interop.h>

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
public:
  std::shared_ptr<Scene> scene;
  std::shared_ptr<Object> fluidObj;
  cudaGraphicsResource_t resource = 0;

  OptixRenderer *renderer;
  SPH_GPU *fluid_system;

public:
  RaytraceUtil(SPH_GPU *system, uint32_t width, uint32_t height) {
    fluid_system = system;
    scene = std::make_shared<Scene>();
    setupSponza(scene, width, height);
    glm::vec3 size = system->fluid_domain.size;
    glm::vec3 center = system->fluid_domain.corner + size * .5f;

    // scene->setEnvironmentMap("../assets/ame_desert/desertsky_ft.tga",
    //                          "../assets/ame_desert/desertsky_bk.tga",
    //                          "../assets/ame_desert/desertsky_up.tga",
    //                          "../assets/ame_desert/desertsky_dn.tga",
    //                          "../assets/ame_desert/desertsky_lf.tga",
    //                          "../assets/ame_desert/desertsky_rt.tga");

    // auto test_obj = NewSphere();
    // test_obj->material.kd = {1,1,1};
    // scene->addObject(test_obj);
    // test_obj->position = system->fluid_domain.corner + system->fluid_domain.size / 2.f;
    // test_obj->scale = system->fluid_domain.size / 2.f;

    int mc_num_cells = system->get_mc_num_cells();

    // TODO: maybe optimize using EBO?
    // each cell can have 5 faces (15 vertices)
    std::shared_ptr<DynamicMesh> mesh = std::make_shared<DynamicMesh>(mc_num_cells * 15);
    CUDA_CHECK_RETURN(cudaGraphicsGLRegisterBuffer(&resource, mesh->getVBO(), cudaGraphicsRegisterFlagsNone));

    fluidObj = NewObject<Object>(mesh);
    fluidObj->name = "Fluid";
    scene->addObject(fluidObj);

    size_t free, used;
    cudaMemGetInfo(&free, &used);
    printf("free: %ld; used: %ld\n", free, used);

    fluidObj->material.kd = {0, 1, 1};

    scene->addDirectionalLight({glm::vec3(0, -1, -1), glm::vec3(1, 1, 0.5)});
    renderer = new OptixRenderer(width, height);
    renderer->init();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glEnable(GL_FRAMEBUFFER_SRGB_EXT);
  }

  void invalidate_camera() {
    renderer->invalidateCamera();
  }

  void render() {
    renderer->renderScene(scene);

    float *d_vertex_pointer;
    size_t size;
    CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &resource));
    CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer((void**)&d_vertex_pointer, &size, resource));
    int num_faces = 0;
    fluid_system->update_mesh();
    fluid_system->update_faces(d_vertex_pointer, &num_faces);
    std::dynamic_pointer_cast<DynamicMesh>(fluidObj->getMesh())->setVertexCount(num_faces * 3);
    CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &resource));
  }

  void renderToFile(uint32_t i = 0) {
    renderer->renderSceneToFile(scene, "/tmp/sph_" + std::to_string(i) + ".png");
  }
};
