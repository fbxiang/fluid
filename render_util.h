#pragma once
#include "fluid_system.h"
#include "render/renderer.h"
#include "render/scene.h"
#include "utils.h"
#include <glm/gtx/compatibility.hpp>
#include <iostream>

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

  void renderToFile(uint32_t i = 0) {
    renderer->renderSceneToFile(scene, "/tmp/sph_" + std::to_string(i) + ".png");
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

  void render_grid() {
    scene->removeObjectsByName("grid_point");
    float grid_step = fluid_system->solver.particle_size / 2.f;
    glm::vec3 corner = fluid_system->fluid_domain.corner;
    glm::vec3 size = fluid_system->fluid_domain.size;
    glm::uvec3 grid = size / grid_step;

    int g = 0;
    for (uint32_t i = 0; i < grid.x; ++i) {
      for (uint32_t j = 0; j < grid.x; ++j) {
        for (uint32_t k = 0; k < grid.x; ++k) {
          glm::vec3 point = corner + glm::vec3(i, j, k) * grid_step;
          // if (fluid_system->check_neighbors_at(point)) {

          float density = fluid_system->compute_density_at(point);

          if (density > 800) {
            auto obj = NewSphere();
            obj->name = "grid_point";
            obj->position = point;
            float scale = grid_step / 2.f;
            obj->scale = {scale, scale, scale};
            obj->material.kd = {1, 0, 0};
            scene->addObject(obj);
            g++;
          }
        }
      }
    }
    cout << g << " Found" << endl;
  }
};
