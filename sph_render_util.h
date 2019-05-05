#pragma once
#include "sph_gpu.h"
#include "render/renderer.h"
#include "render/scene.h"
#include <glm/gtx/compatibility.hpp>
#include <iostream>

class RenderUtil {
public:
  std::shared_ptr<Scene> scene;
  std::shared_ptr<Object> domain;
  std::vector<std::shared_ptr<Object>> particles;

  SPH_GPU *sph;
  Renderer *renderer;

public:
  RenderUtil(SPH_GPU *system, uint32_t width, uint32_t height) {
    sph = system;
    scene = std::make_shared<Scene>();
    auto cam = NewObject<Camera>();
    glm::vec3 size = system->fluid_domain.size;
    glm::vec3 center = system->fluid_domain.corner + size * .5f;

    cam->position = {center.x, center.y, center.z + size.z * 5};
    cam->fovy = glm::radians(45.f);
    cam->aspect = width / (float)height;
    scene->addObject(cam);
    scene->setMainCamera(cam);

    scene->addDirectionalLight({glm::vec3(0, -1, -1), glm::vec3(1, 1, 0.5)});
    renderer = new Renderer(width, height);
    renderer->init();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glEnable(GL_FRAMEBUFFER_SRGB_EXT);
  }

  void render() {
    update_particles();
    renderer->renderScene(scene);
  }

  void renderToFile(uint32_t i = 0) {
    renderer->renderSceneToFile(scene, "/tmp/sph_" + std::to_string(i) + ".png");
  }

  void add_particles() {
    particles.resize(sph->positions.size());
    for (uint32_t i = 0; i < particles.size(); ++i) {
      particles[i] = NewSphere();
      glm::vec3 pos = sph->positions[i];
      particles[i]->position = {pos.x, pos.y, pos.z};
      scene->addObject(particles[i]);
      float scale = sph->solver_params.particle_size / 2.f;
      particles[i]->scale = {scale, scale, scale};
      glm::uvec3 size = sph->grid_size();
      particles[i]->material.kd = glm::vec3(0, 0.5, 0.75);
    }
  }

  void update_particles() {
    for (uint32_t i = 0; i < particles.size(); ++i) {
      particles[i]->position = sph->positions[i];
      glm::uvec3 size = sph->grid_size();
      float ratio = sph->rho[i] / sph->solver_params.rest_density;

      float l = glm::clamp((ratio - 1.f) / 3.f, 0.f, 1.f);
      auto color = glm::lerp(glm::vec3(0, 0.5, 0.75), glm::vec3(1, 0, 0), l);

      particles[i]->material.kd = color;
    }
  }
};
