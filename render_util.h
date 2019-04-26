#pragma once
#include "fluid_system.h"
#include "render/renderer.h"
#include "render/scene.h"
#include "utils.h"
#include <glm/gtx/compatibility.hpp>

class RenderUtil {
public:
  std::shared_ptr<Scene> scene;
  std::shared_ptr<Object> domain;
  std::vector<std::shared_ptr<Object>> particles;

  FluidSystem *fluid_system;
  Renderer *renderer;

public:
  RenderUtil(FluidSystem *system, uint32_t width, uint32_t height) {
    fluid_system = system;
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

  void add_fluid_domain_object() {
    domain = NewCube();
    auto size = fluid_system->fluid_domain.size;
    glm::vec3 center = fluid_system->fluid_domain.corner + size * 0.5f;
    domain->scale *= glm::vec3(size.x, size.y, size.z);
    domain->position = glm::vec3(center.x, center.y, center.z);
    scene->addObject(domain);
    domain->material.kd = glm::vec3(0, 0, 1);
  }

  void add_particles() {
    particles.resize(fluid_system->positions.size());
    for (uint32_t i = 0; i < particles.size(); ++i) {
      particles[i] = NewSphere();
      glm::vec3 pos = fluid_system->positions[i];
      particles[i]->position = {pos.x, pos.y, pos.z};
      scene->addObject(particles[i]);
      float scale = fluid_system->solver.particle_size / 2.f;
      particles[i]->scale = {scale, scale, scale};
      glm::uvec3 size = fluid_system->grid_size();
      particles[i]->material.kd = glm::vec3(0, 0.5, 0.75);
    }
  }

  void update_particles() {
    for (uint32_t i = 0; i < particles.size(); ++i) {
      particles[i]->position = fluid_system->positions[i];
      glm::uvec3 size = fluid_system->grid_size();
      float ratio = fluid_system->rhos[i] / fluid_system->solver.rest_density;

      float l = glm::clamp((ratio - 1.f) / 3.f, 0.f, 1.f);
      auto color = glm::lerp(glm::vec3(0, 0.5, 0.75), glm::vec3(1, 0, 0), l);
      
      particles[i]->material.kd = color;
    }
  }
};
