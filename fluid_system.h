#pragma once
#include "fluid_domain.h"
#include "particle.h"
#include <algorithm>
#include <cstdint>
#include <glm/glm.hpp>
#include <iostream>
#include <vector>

using std::cout;
using std::endl;

class SolverParams {
public:
  glm::vec3 g = {0.f, -9.8f, 0.f};
  float rest_density = 1000.f;
  float k = 20.f;     // stiffness constant
  float gamma = 7.f; // stiffness exponential

  float viscosity = 0.000f;
  float particle_mass; // h * h * h * rest_density
  float time_step = 0.001f;

  float particle_size; // h
  float cell_size;

  float eps = 1e-6;
};

class FluidSystem {

public:
  SolverParams solver;
  FluidDomain fluid_domain;

  void set_particle_size(float size);

  inline glm::uvec3 grid_size() const {
    uint32_t x = ceil(fluid_domain.size.x / solver.cell_size);
    uint32_t y = ceil(fluid_domain.size.y / solver.cell_size);
    uint32_t z = ceil(fluid_domain.size.z / solver.cell_size);
    return {x, y, z};
  }

  inline uint32_t num_cells() const {
    uint32_t x = ceil(fluid_domain.size.x / solver.cell_size);
    uint32_t y = ceil(fluid_domain.size.y / solver.cell_size);
    uint32_t z = ceil(fluid_domain.size.z / solver.cell_size);
    uint32_t m = std::max(std::max(x, y), z);
    m = pow(2, ceilf(log2f(m)));
    return m * m * m;
  }

  inline glm::uvec3 pos2grid_pos(glm::vec3 pos) const {
    return (pos - fluid_domain.corner) / solver.cell_size;
  }

  // cell idx to start point in particle id
  std::vector<uint32_t> grid;

  // particle properties
  std::vector<glm::vec3> positions;
  std::vector<glm::vec3> velocities;

  // cell for each particle
  std::vector<uint32_t> cell_indices;

  // particle indices sorted by cell
  std::vector<uint32_t> particle_ids;

  inline uint32_t size() const { return positions.size(); }

  // cache for neighbors
  std::vector<std::vector<uint32_t>> neighbors;

  std::vector<float> rhos;
  std::vector<float> pressures;
  std::vector<glm::vec3> forces;

public:
  virtual void init() { grid = std::vector<uint32_t>(num_cells()); }

  virtual void set_particles(const std::vector<Particle> &ps);

  void update_cell_indices();

  void update_particle_ids();

  void sort_particles();

  // tested
  std::vector<uint32_t> find_ids_in_cell(uint32_t cell_idx) const;

  // thread method
  void _find_neighbors(uint32_t i);

  // cache neighbors
  void find_neighbors();

  std::vector<uint32_t> find_neighbors_at(glm::vec3 point) const;
  bool check_neighbors_at(glm::vec3 point) const;

  float kernel(float q) const {
    if (q < 1) {
      return 3.f / (2 * M_PIf32) * (2.f / 3.f - q * q + 0.5 * q * q * q);
    } else if (q < 2) {
      return 3.f / (2 * M_PIf32) * (1.f / 6.f * powf(2 - q, 3));
    }
    return 0.f;
  }

  float dkernel(float q) const {
    if (q < 1) {
      return 3.f / (2 * M_PIf32) * (-2 * q + 1.5 * q * q);
    } else if (q < 2) {
      return 3.f / (2 * M_PIf32) * (-3.f / 6.f * powf(2 - q, 2));
    }
    return 0.f;
  }

  float w_ij(uint32_t i, uint32_t j) const {
    return w_ij(positions[i], positions[j]);
  }

  float w_ij(glm::vec3 pos1, glm::vec3 pos2) const {
    float h = solver.particle_size;
    float r = glm::length(pos1 - pos2);
    float q = r / h;
    return 1.f / powf(h, 3) * kernel(q);
  }

  glm::vec3 dw_ij(uint32_t i, uint32_t j) const {
    return dw_ij(positions[i], positions[j]);
  }

  glm::vec3 dw_ij(glm::vec3 pos1, glm::vec3 pos2) const {
    float h = solver.particle_size;
    glm::vec3 d = pos1 - pos2;
    float r = glm::length(d);
    float q = r / h;
    if (r < 1e-10) { return glm::vec3(); }
    return d * ((1.f / powf(h, 4.f)) * dkernel(q) / r);
  }


  // average quantity at point
  template <typename T> T quantity(const std::vector<T> &values, uint32_t i) {
    float h = solver.particle_size;
    T output;
    for (uint32_t j : neighbors[i]) {
      output += (w_ij(i, j) * solver.particle_mass / rhos[j]) * values[j];
    }
    return output;
  }

  // gradient at point
  glm::vec3 gradient(const std::vector<float> &quantity, uint32_t i) {
    glm::vec3 value = {0.f, 0.f, 0.f};
    for (uint32_t j : neighbors[i]) {
      if (i == j)
        continue;
      value +=
          solver.particle_mass *
          (quantity[i] / rhos[i] / rhos[i] + quantity[j] / rhos[j] / rhos[j]) *
          dw_ij(i, j);
    }

    return rhos[i] * value;
  }

  // laplacian at point
  template <typename T>
  T laplacian(const std::vector<T> &quantity, uint32_t i) {
    float h = solver.particle_size;
    T value;
    for (uint32_t j : neighbors[i]) {
      glm::vec3 xij = positions[i] - positions[j];
      value += (quantity[i] - quantity[j]) *
               (solver.particle_mass / rhos[j] * glm::dot(xij, dw_ij(i, j)) /
                (glm::dot(xij, xij) + 0.01f * h * h));
    }
    return 2.f * value;
  }

  virtual void compute_density();
  float compute_density_at(glm::vec3 point) const;
  glm::vec3 compute_density_gradient_at(glm::vec3 point, float point_density) const;
  
  virtual void compute_pressure();

  float compute_adaptive_time_step() const; 

  // cache all forces
  void compute_forces();

  // compute integration
  void integrate();

  // find neighbors, compute density pressure, compute forces, integrate
  virtual void step();
};
