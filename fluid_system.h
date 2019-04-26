#pragma once
#include "fluid_domain.h"
#include "particle.h"
#include "utils.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <glm/glm.hpp>
#include <thread>
#include <vector>

static long l1 = 0;
static long l2 = 0;
static long l3 = 0;
static long l4 = 0;
static long l5 = 0;

using std::cout;
using std::endl;
using namespace std::chrono;

class SolverParams {
public:
  glm::vec3 g = {0.f, -9.8f, 0.f};
  float rest_density = 1000.f;
  float k = 1000.f;

  float viscosity = 0.f;
  float particle_mass;
  float time_step = 0.001f;

  float particle_size;

  float eps = 0.1e-5;
};

class FluidSystem {

public:
  SolverParams solver;
  FluidDomain fluid_domain;

  void set_particle_size(float size) {
    solver.particle_size = size;
    solver.particle_mass = size * size * size * solver.rest_density;
  }

  glm::uvec3 grid_size() {
    uint32_t x = ceil(fluid_domain.size.x / solver.particle_size);
    uint32_t y = ceil(fluid_domain.size.y / solver.particle_size);
    uint32_t z = ceil(fluid_domain.size.z / solver.particle_size);
    return {x, y, z};
  }

  uint32_t num_cells() {
    uint32_t x = ceil(fluid_domain.size.x / solver.particle_size);
    uint32_t y = ceil(fluid_domain.size.y / solver.particle_size);
    uint32_t z = ceil(fluid_domain.size.z / solver.particle_size);
    uint32_t m = std::max(std::max(x, y), z);
    m = pow(2, ceilf(log2f(m)));
    return m * m * m;
  }

  glm::uvec3 pos2grid_pos(glm::vec3 pos) {
    return (pos - fluid_domain.corner) / solver.particle_size;
  }

  // cell idx to start point in particle id
  std::vector<uint32_t> grid;

  std::vector<glm::vec3> positions;
  std::vector<glm::vec3> velocities;

  // cell for each particle
  std::vector<uint32_t> cell_indices;

  // particle indices sorted by cell
  std::vector<uint32_t> particle_ids;

  std::vector<std::vector<uint32_t>> neighbors;

  std::vector<float> rhos;
  std::vector<float> pressures;
  std::vector<glm::vec3> forces;

  void init() { grid = std::vector<uint32_t>(num_cells()); }

  // tested
  void set_particles(const std::vector<Particle> &ps) {
    positions.resize(ps.size());
    velocities.resize(ps.size());
    for (uint32_t i = 0; i < ps.size(); ++i) {
      positions[i] = ps[i].position;
      velocities[i] = ps[i].velocity;
    }
    
    cell_indices.resize(ps.size());
    update_cell_indices();

    particle_ids.resize(ps.size());
    for (uint32_t i = 0; i < particle_ids.size(); ++i) {
      particle_ids[i] = i;
    }
    update_particle_ids();
    rhos.resize(ps.size());
    pressures.resize(ps.size());
    forces.resize(ps.size());
    neighbors.resize(ps.size());
  }

  // tested
  void update_cell_indices() {
    for (uint32_t i = 0; i < positions.size(); ++i) {
      glm::uvec3 grid_pos = pos2grid_pos(positions[i]);
      cell_indices[i] = zorder2number(grid_pos);
    }
  }

  // tested
  // insertion sort
  void update_particle_ids() {
    for (uint32_t i = 1; i < particle_ids.size(); ++i) {
      uint32_t id = particle_ids[i];
      uint32_t j = i;
      while (cell_indices[id] < cell_indices[particle_ids[j - 1]]) {
        particle_ids[j] = particle_ids[j - 1];
        j -= 1;
        if (j == 0)
          break;
      }
      particle_ids[j] = id;
    }

    grid[cell_indices[particle_ids[0]]] = 0;
    for (uint32_t i = 1; i < particle_ids.size(); ++i) {
      if (cell_indices[particle_ids[i - 1]] != cell_indices[particle_ids[i]]) {
        grid[cell_indices[particle_ids[i]]] = i;
      }
    }
  }

  void sort_particles() {
    std::vector<glm::vec3> pos_clone = positions;
    std::vector<glm::vec3> vel_clone = velocities;
    for (uint32_t i = 0; i < particle_ids.size(); ++i) {
      positions[i] = pos_clone[particle_ids[i]];
      velocities[i] = vel_clone[particle_ids[i]];
    }
    update_cell_indices();
    update_particle_ids();
  }

  // tested
  void find_ids_in_cell(uint32_t cell_idx, std::vector<uint32_t> &out) {
    for (uint32_t i = grid[cell_idx];
         i < particle_ids.size() && cell_indices[particle_ids[i]] == cell_idx;
         ++i) {
      out.push_back(particle_ids[i]);
    }
  }

  void _find_neighbors(uint32_t start, uint32_t length) {
    float h = solver.particle_size;
    uint32_t end = start + length;
    if (end >= neighbors.size())
      end = neighbors.size();
    for (uint32_t i = start; i < end; ++i) {
      neighbors[i] = std::vector<uint32_t>();
      glm::vec3 position = positions[i];
      glm::uvec3 grid_pos = pos2grid_pos(position);
      glm::uvec3 size = grid_size();
      std::vector<int32_t> xs = {0};
      std::vector<int32_t> ys = {0};
      std::vector<int32_t> zs = {0};
      if (grid_pos.x > 0)
        xs.push_back(-1);
      if (grid_pos.y > 0)
        ys.push_back(-1);
      if (grid_pos.z > 0)
        zs.push_back(-1);
      if (grid_pos.x < size.x - 1)
        xs.push_back(1);
      if (grid_pos.y < size.y - 1)
        ys.push_back(1);
      if (grid_pos.z < size.z - 1)
        zs.push_back(1);

      for (int32_t x : xs) {
        for (int32_t y : ys) {
          for (int32_t z : zs) {
            uint32_t cell_idx =
                zorder2number({grid_pos.x + x, grid_pos.y + y, grid_pos.z + z});
            std::vector<uint32_t> ids;
            find_ids_in_cell(cell_idx, ids);
            // prune
            for (auto j : ids) {
              auto r = positions[i] - positions[j];
              if (glm::dot(r, r) < h * h) {
                neighbors[i].push_back(j);
              }
            }
          }
        }
      }
    }
  }

  // tested?
  void find_neighbors() {
    int n_threads = 8;
    std::vector<std::thread> threads;
    auto length = positions.size() / n_threads + 1;
    for (int i = 0; i < n_threads; i++) {
      threads.push_back(
          std::thread([=] { this->_find_neighbors(length * i, length); }));
    }
    for (int i = 0; i < n_threads; i++) {
      threads[i].join();
    }
  }

  float kernel(float q) {
    if (q < 1) {
      return 3.f / (2 * M_PIf32) * (2.f / 3.f - q * q + 0.5 * q * q * q);
    } else if (q < 2) {
      return 3.f / (2 * M_PIf32) * (1.f / 6.f * powf(2 - q, 3));
    }
    return 0.f;
  }

  float dkernel(float q) {
    if (q < 1) {
      return 3.f / (2 * M_PIf32) * (-2 * q + 1.5 * q * q);
    } else if (q < 2) {
      return 3.f / (2 * M_PIf32) * (-3.f / 6.f * powf(2 - q, 2));
    }
    return 0.f;
  }

  float w_ij(uint32_t i, uint32_t j) {
    float h = solver.particle_size;
    float r = glm::length(positions[i] - positions[j]);
    float q = r / h;
    return 1.f / powf(h, 3) * kernel(q);
  }

  glm::vec3 dw_ij(uint32_t i, uint32_t j) {
    float h = solver.particle_size;
    glm::vec3 d = positions[i] - positions[j];
    float r = glm::length(d);
    float q = r / h;
    if ( r < 1e-10 ) {
      return {0.f, 0.f, 0.f};
    }

    return d * (1.f / powf(h, 4.f) * dkernel(q) / r);
  }

  void compute_density_presure() {
    float h = solver.particle_size;
    for (uint32_t i = 0; i < positions.size(); ++i) {
      rhos[i] = 0.f;

      for (uint32_t j : neighbors[i]) {
        rhos[i] += solver.particle_mass * w_ij(i, j);
      }

      pressures[i] = solver.k * (pow(rhos[i] / solver.rest_density, 7) - 1);
    }
  }

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

  template<typename T>
  T laplacian(const std::vector<T> &quantity, uint32_t i) {
    float h = solver.particle_size;
    T value;
    for (uint32_t j : neighbors[i]) {
      glm::vec3 xij = positions[i] - positions[j];
      value += (quantity[i] - quantity[j]) * (solver.particle_mass / rhos[j]
               * glm::dot(xij, dw_ij(i, j)) / (glm::dot(xij, xij) + 0.01f * h * h));
    }
    return 2.f * value;
  }

  // TODO: test
  void compute_forces() {
    float h = solver.particle_size;
    for (uint32_t i = 0; i < positions.size(); ++i) {
      glm::vec3 dp = {0, 0, 0};
      glm::vec3 lv = {0, 0, 0};

      glm::vec3 f_pressure = -solver.particle_mass / rhos[i] * gradient(pressures, i);
      glm::vec3 f_viscosity = solver.particle_mass * solver.viscosity * laplacian(velocities, i);
      glm::vec3 f_other = solver.particle_mass * solver.g;
      forces[i] = f_pressure + f_viscosity + f_other;
    }
  }

  // TODO: test, improve
  void integrate() {
    for (uint32_t i = 0; i < positions.size(); ++i) {
      velocities[i] += solver.time_step * forces[i] / solver.particle_mass;
      positions[i] += solver.time_step * velocities[i];

      if (positions[i].x < fluid_domain.corner.x) {
        velocities[i].x *= -0.1;
        positions[i].x = fluid_domain.corner.x + solver.eps;
      }
      if (positions[i].y < fluid_domain.corner.y) {
        velocities[i].y *= -0.1;
        positions[i].y = fluid_domain.corner.y + solver.eps;
      }
      if (positions[i].z < fluid_domain.corner.z) {
        velocities[i].z *= -0.1;
        positions[i].z = fluid_domain.corner.z + solver.eps;
      }

      if (positions[i].x >= fluid_domain.corner.x + fluid_domain.size.x) {
        velocities[i].x *= -0.1;
        positions[i].x = fluid_domain.corner.x + fluid_domain.size.x - solver.eps;
      }
      if (positions[i].y >= fluid_domain.corner.y + fluid_domain.size.y) {
        velocities[i].y *= -0.1;
        positions[i].y = fluid_domain.corner.y + fluid_domain.size.y - solver.eps;
      }
      if (positions[i].z >= fluid_domain.corner.z + fluid_domain.size.z) {
        velocities[i].z *= -0.1;
        positions[i].z = fluid_domain.corner.z + fluid_domain.size.z - solver.eps;
      }
    }
  }

  void step() {
    auto t0 = high_resolution_clock::now();

    update_cell_indices();
    update_particle_ids();

    auto t1 = high_resolution_clock::now();

    find_neighbors();

    // NOTE: Neighbor Validity Check
    // for (int i = 0; i < particles.size(); ++i) {
    //   int k1 = 0;
    //   int k2 = 0;
    //   k1 = neighbors[i].size();
    //   // for (auto j : neighbors[i]) {
    //   //   if (glm::length(particles[i].position - particles[j].position) <
    //   //   solver.particle_size) {
    //   //     ++k1;
    //   //   }
    //   // }
    //   for (int j = 0; j < particles.size(); ++j) {
    //     if (glm::length(particles[i].position - particles[j].position) <
    //         solver.particle_size) {
    //       ++k2;
    //     }
    //   }
    //   if (k1 != k2) {
    //     cout << "Wrong at " << i << endl;
    //     cout << "k1 " << k1 << " k2 " << k2 << endl;
    //     scanf("%d");
    //   }
    //   if (k1 == 0) {
    //     cout << "No neighbor at " << i << endl;
    //     cout << particles[i].position << endl;
    //     scanf("%d");
    //   }
    // }
    // cout << "Check passed!" << endl;

    auto t2 = high_resolution_clock::now();

    compute_density_presure();
    compute_forces();
    integrate();

    auto t3 = high_resolution_clock::now();

    l1 = (t1 - t0).count();
    l2 = (t2 - t1).count();
    l3 = (t3 - t2).count();
  }

  void show_profile() {
    printf("%f %f %f %f %f\n", l1 / 1000.f, l2 / 1000.f, l3 / 1000.f,
           l4 / 1000.f, l5 / 1000.f);
  }
};
