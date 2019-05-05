#pragma once
#include "sph_math.h"
#include <algorithm>
#include <cstdint>
#include <glm/glm.hpp>
#include <iostream>
#include <vector>
#include "sph.cuh"

class SPH_GPU {
public:
  SolverParams solver_params;
  FluidDomain fluid_domain;

  SPH_GPU(float size=0.05f) {
    solver_params.g = { 0.f, -9.8f, 0.f };
    solver_params.rest_density = 1000.f;
    solver_params.k = 7.f;
    solver_params.gamma = 7.f;
    solver_params.viscosity = 0.001f;
    solver_params.particle_size = size;
    solver_params.cell_size = 2 * size;
    solver_params.time_step = 0.001f;  // not used
    solver_params.particle_mass = size * size * size * 1000.f;
    solver_params.eps = 1e-6;
  }

  void set_domain(glm::vec3 corner, glm::vec3 size) {
    fluid_domain.corner = corner;
    fluid_domain.size = size;
  }

  void cuda_init() {
    sph::init(solver_params, fluid_domain, 1000000);
  }

  void add_particles(std::vector<glm::vec3>& positions) {
    sph::add_particles(positions.data(), positions.size());
  }

  std::vector<glm::vec3> get_particles() {
    int size = sph::get_num_particles();
    std::vector<glm::vec3> positions(size);
    sph::get_particles(positions.data(), size);
    return positions;
  }

  void update_neighbors() {
    sph::update_neighbors();
  }

  float step_regular() {
    return sph::step_regular();
  }
  /* initialization stage */
  // void sim_init();

  /* loop stage */
  // void step();
};
