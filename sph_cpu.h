#pragma once
#include "sph_math.h"
#include <algorithm>
#include <cstdint>
#include <glm/glm.hpp>
#include <iostream>
#include <vector>
#include "sph_common.h"

class SPH_CPU {
public:
  SolverParams solver_params;
  FluidDomain fluid_domain;

  void set_particle_size_and_density(float h, float density) {
    solver_params.rest_density = density;
    solver_params.particle_size = h;
    solver_params.cell_size = h * 2;
    solver_params.particle_mass = h * h * h * density;
  }

  void set_domain(glm::vec3 corner, glm::vec3 size) {
    fluid_domain.corner = corner;
    fluid_domain.size = size;
  }

  void update_neighbors();
  void update_densities();
  void update_factors();
  void update_non_pressure_forces();
  void update_dt_by_CFL();
  void update_predicted_velocities();
  void correct_density_error();
  void update_positions();
  void correct_divergence_error();
  void update_velocities();

  void init(uint32_t size);
  void init(const std::vector<glm::vec3> ps);

  /* initialization stage */
  void sim_init();

  /* loop stage */
  void step();

  //tested
  void clear_grid();
  // tested
  void update_cell_idx();
  // tested
  void update_sorted_particle_idx();

  /* ======================= Helpers  ======================= */
  std::vector<glm::vec3> positions;
  std::vector<glm::vec3> velocities;
  std::vector<glm::vec3> velocities_pred;

  // cell_idx[i]: cell number for particle i
  std::vector<uint32_t> cell_idx;

  // sorted_particle_idx[i]: particle number that ranks at i
  std::vector<uint32_t> sorted_particle_idx;

  // grid[n]: starting rank for grid n
  std::vector<uint32_t> grid;
  std::vector<uint32_t> grid_start_idx;

  std::vector<float> rho;
  std::vector<float> rho_pred;
  std::vector<float> alpha;
  std::vector<float> pressure;
  std::vector<glm::vec3> force;

  std::vector<float> kappa;
  std::vector<float> kappav;

  float dt;

  // tested
  inline glm::uvec3 grid_size() const {
    uint32_t x = ceil(fluid_domain.size.x / solver_params.cell_size);
    uint32_t y = ceil(fluid_domain.size.y / solver_params.cell_size);
    uint32_t z = ceil(fluid_domain.size.z / solver_params.cell_size);
    return {x, y, z};
  }

  inline uint32_t num_cells() const {
    glm::vec3 size = grid_size();
    return size.x * size.y * size.z;
  }

  inline glm::uvec3 pos2grid_pos(glm::vec3 pos) const {
    return (pos - fluid_domain.corner) / solver_params.cell_size;
  }

  inline uint32_t grid_pos2cell_idx(glm::uvec3 gpos) const {
    glm::uvec3 gsize = grid_size();
    if (gpos.x < 0 || gpos.y < 0 || gpos.z < 0 || gpos.x >= gsize.x ||
        gpos.y >= gsize.y || gpos.z >= gsize.z) {
      return -1;
    }
    return gpos.x * gsize.y * gsize.z + gpos.y * gsize.z + gpos.z;
  }

  // tested
  inline uint32_t pos2cell_idx(glm::vec3 pos) const {
    glm::uvec3 gpos = pos2grid_pos(pos);
    return grid_pos2cell_idx(gpos);
  }

  // tested
  // TODO: neighbors use cell size
  inline std::vector<uint32_t> find_neighbors(uint32_t i) {
    float c = solver_params.cell_size;
    std::vector<uint32_t> out;
    glm::uvec3 gsize = grid_size();
    glm::uvec3 gpos = pos2grid_pos(positions[i]);

    std::vector<int> xs = {0};
    std::vector<int> ys = {0};
    std::vector<int> zs = {0};
    if (gpos.x > 0) {
      xs.push_back(-1);
    }
    if (gpos.y > 0) {
      ys.push_back(-1);
    }
    if (gpos.z > 0) {
      zs.push_back(-1);
    }
    if (gpos.x < gsize.x - 1) {
      xs.push_back(1);
    }
    if (gpos.y < gsize.y - 1) {
      ys.push_back(1);
    }
    if (gpos.z < gsize.z - 1) {
      zs.push_back(1);
    }
    for (int x : xs) {
      for (int y : ys) {
        for (int z : zs) {
          uint32_t cell_idx = grid_pos2cell_idx(glm::uvec3(gpos.x + x, gpos.y + y, gpos.z + z));
          for (uint32_t k = grid_start_idx[cell_idx], e = k + grid[cell_idx];
               k < e; ++k) {
            uint32_t j = sorted_particle_idx[k];
            glm::vec3 r = positions[i] - positions[j];
            if (glm::dot(r, r) < c * c) {
              out.push_back(j);
            }
          }
        }
      }
    }
    return out;
  }

  glm::vec3 gradient(const std::vector<float> &quantity, uint32_t i) {
    glm::vec3 value = {0.f, 0.f, 0.f};
    for (uint32_t j : find_neighbors(i)) {
      if (i == j)
        continue;
      value += solver_params.particle_mass *
               (quantity[i] / rho[i] / rho[i] + quantity[j] / rho[j] / rho[j]) *
               dw_ij(positions[i], positions[j], solver_params.particle_size);
    }

    return rho[i] * value;
  }

  float divergence(const std::vector<glm::vec3> &vec, uint32_t i) {
    float value = 0;
    for (uint32_t j: find_neighbors(i)) {
      if (i == j) {
        continue;
      }
      value += solver_params.particle_mass *
               glm::dot(vec[i] - vec[j], dw_ij(positions[i], positions[j], solver_params.particle_size));
    }
    value /= -rho[i];
    return value;
  }

  inline uint32_t size() const { return positions.size(); }
};
