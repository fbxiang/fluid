#pragma once
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

struct SolverParams {
  glm::vec3 g;        // = {0.f, -9.8f, 0.f};
  float rest_density; // = 1000.f;
  float k;            // = 7.f;
  float gamma;        // = 7.f;

  float viscosity;     // = 0.f;
  float particle_mass; // h * h * h * rest_density
  float time_step;     // = 0.001f;

  float particle_size; // h
  float cell_size;     // 2 * h

  float eps; // = 1e-6;
};

struct FluidDomain {
  glm::vec3 corner;
  glm::vec3 size;
};
