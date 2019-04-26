#pragma once
#include "fluid_domain.h"
#include "fluid_system.h"
#include "particle.h"
#include "utils.h"
#include <cstdint>

void random_fill_system(FluidSystem &system, uint32_t n) {
  std::vector<Particle> particles;
  for (uint32_t i = 0; i < n; ++i) {
    float x =
        rand_float(system.fluid_domain.corner.x, system.fluid_domain.size.x);
    float y =
        rand_float(system.fluid_domain.corner.y, system.fluid_domain.size.y);
    float z =
        rand_float(system.fluid_domain.corner.z, system.fluid_domain.size.z);
    particles.push_back({{x, y, z}, {0.f, 0.f, 0.f}});
  }
  system.set_particles(particles);
}

void fill_block(FluidSystem &system, glm::vec3 corner, glm::vec3 size) {
  float step = system.solver.particle_size;
  std::vector<Particle> particles;
  glm::uvec3 n = size / step;
  for (uint32_t i = 0; i < n.x; ++i) {
    for (uint32_t j = 0; j < n.y; ++j) {
      for (uint32_t k = 0; k < n.z; ++k) {
        glm::vec3 jitter = {rand_float(-step / 4.f, step / 4.f),
                            rand_float(-step / 4.f, step / 4.f),
                            rand_float(-step / 4.f, step / 4.f)};
        particles.push_back({{corner + glm::vec3(i, j, k) * step + jitter}, {0, 0, 0}});
      }
    }
  }
  system.set_particles(particles);
}


void fill_triangle(FluidSystem &system, glm::vec3 corner, glm::vec3 size) {
  float step = system.solver.particle_size;
  std::vector<Particle> particles;
  glm::uvec3 n = size / step;
  for (uint32_t i = 0; i < n.x; ++i) {
    for (uint32_t j = i; j < n.y; ++j) {
      for (uint32_t k = 0; k < n.z; ++k) {
        glm::vec3 jitter = {rand_float(-step / 4.f, step / 4.f),
                            rand_float(-step / 4.f, step / 4.f),
                            rand_float(-step / 4.f, step / 4.f)};
        particles.push_back(
            {{corner + glm::vec3(i, n.y - j, k) * step + jitter}, {0, 0, 0}});
      }
    }
  }
  system.set_particles(particles);
}

void log_system(const FluidSystem &system) {
  std::cout << "[Fluid System]" << std::endl;
  // std::cout << "cell indices: " << system.cell_indices << std::endl;
  // std::cout << "particle ids: " << system.particle_ids << std::endl;
  // std::cout << "[ ";
  // for (auto id : system.particle_ids) {
  //   std::cout << system.grid[system.cell_indices[id]] << " ";
  // }
  // std::cout << "]";
  std::cout << std::endl;
}
