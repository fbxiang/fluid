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
public:
  std::shared_ptr<Scene> scene;
  std::shared_ptr<Object> domain;
  std::vector<std::shared_ptr<Object>> particles;
  std::shared_ptr<Object> fluidObj;
  cudaGraphicsResource_t resource = 0;

  Renderer *renderer;
  SPH_GPU *fluid_system;

public:
  GPURenderUtil(SPH_GPU *system, uint32_t width, uint32_t height) {
    fluid_system = system;
    scene = std::make_shared<Scene>();
    // setupEmpty(scene, width, height);
    setupSponza(scene, width, height);
    glm::vec3 size = system->fluid_domain.size;
    glm::vec3 center = system->fluid_domain.corner + size * .5f;

    // scene->setEnvironmentMap("../assets/ame_desert/desertsky_ft.tga",
    //                          "../assets/ame_desert/desertsky_bk.tga",
    //                          "../assets/ame_desert/desertsky_up.tga",
    //                          "../assets/ame_desert/desertsky_dn.tga",
    //                          "../assets/ame_desert/desertsky_lf.tga",
    //                          "../assets/ame_desert/desertsky_rt.tga");

    // auto lineCube = NewLineCube();
    // lineCube->material.kd = {1, 1, 1};
    // scene->addObject(lineCube);
    // lineCube->position =
    //     system->fluid_domain.corner + system->fluid_domain.size / 2.f;
    // lineCube->scale = system->fluid_domain.size / 2.f;

    int mc_num_cells = system->get_mc_num_cells();

    // TODO: maybe optimize using EBO?
    // each cell can have 5 faces (15 vertices)
    std::shared_ptr<DynamicMesh> mesh =
        std::make_shared<DynamicMesh>(mc_num_cells * 15);
    CUDA_CHECK_RETURN(cudaGraphicsGLRegisterBuffer(
        &resource, mesh->getVBO(), cudaGraphicsRegisterFlagsNone));

    fluidObj = NewObject<Object>(mesh);
    fluidObj->name = "Fluid";
    scene->addObject(fluidObj);

    fluidObj->material.kd = {0, 1, 1};

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
    int num_faces = 0;
    fluid_system->update_mesh();
    fluid_system->update_faces(d_vertex_pointer, &num_faces);
    // printf("Num faces: %d\n", num_faces);
    std::dynamic_pointer_cast<DynamicMesh>(fluidObj->getMesh())
        ->setVertexCount(num_faces * 3);
    CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &resource));
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
    for (int i = 0; i < positions.size(); ++i) {
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
