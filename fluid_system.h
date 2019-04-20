#pragma once
#include "fluid_domain.h"
#include "particle.h"
#include "utils.h"
#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

using std::cout;
using std::endl;

class SolverParams {
public:
  glm::vec3 g = {0.f, 1000 * -9.8f, 0.f};
  float rest_density = 1000.f;
  float gas_const = 2000.f;

  float viscosity = 0.f;
  float particle_mass = 1.f;
  float time_step = 0.001f;

  float eps = 0.0001;
};

class FluidSystem {

public:
  SolverParams solver;
  FluidDomain fluid_domain;
  uint32_t step_size;

  // cell idx to start point in particle id
  std::vector<uint32_t> grid;

  std::vector<Particle> particles;

  // cell for each particle
  std::vector<uint32_t> cell_indices;

  // particle indices sorted by cell
  std::vector<uint32_t> particle_ids;

  std::vector<float> rhos;
  std::vector<float> pressures;
  std::vector<glm::vec3> forces;

  void init() { grid = std::vector<uint32_t>(fluid_domain.num_cells()); }

  // tested
  void set_particles(const std::vector<Particle> &ps) {
    particles = ps;
    cell_indices.resize(particles.size());
    update_cell_indices();

    particle_ids.resize(particles.size());
    for (uint32_t i = 0; i < particle_ids.size(); ++i) {
      particle_ids[i] = i;
    }
    update_particle_ids();
    rhos.resize(particles.size());
    pressures.resize(particles.size());
    forces.resize(particles.size());
  }

  // tested
  void update_cell_indices() {
    for (uint32_t i = 0; i < particles.size(); ++i) {
      glm::uvec3 grid_pos = fluid_domain.pos2grid_pos(particles[i].position);
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

  // tested
  // place particles based on particle_ids
  void sort_particles() {
    std::vector<Particle> particles_clone = particles;
    for (uint32_t i = 0; i < particle_ids.size(); ++i) {
      particles[i] = particles_clone[particle_ids[i]];
    }
    update_cell_indices();
    update_particle_ids();
  }

  // tested
  void find_ids_in_cell(uint32_t cell_idx, std::vector<uint32_t> &out) {
    if (cell_idx < 0) {
      printf("Huge error 0!");
      exit(1);
    }
    if (cell_idx >= grid.size()) {
      printf("Huge error!");
      exit(1);
    }
    if (grid[cell_idx] < 0) {
      printf("Huge error 2!");
      exit(1);
    }

    for (uint32_t i = grid[cell_idx];
         i < particle_ids.size() && cell_indices[particle_ids[i]] == cell_idx;
         ++i) {
      out.push_back(particle_ids[i]);
    }
  }

  // tested?
  void find_neighbors(uint32_t i, std::vector<uint32_t> &out) {
    glm::vec3 position = particles[i].position;
    glm::uvec3 grid_pos = fluid_domain.pos2grid_pos(position);
    glm::uvec3 size = fluid_domain.grid_size();
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
    for (int32_t i : xs) {
      for (int32_t j : ys) {
        for (int32_t k : zs) {
          uint32_t cell_idx =
              zorder2number({grid_pos.x + i, grid_pos.y + j, grid_pos.z + k});
          find_ids_in_cell(cell_idx, out);
        }
      }
    }
  }

  // TODO: test
  float kernel(float q) {
    if (q < 1) {
      return 3.f / (2 * M_PIf32) * (2.f / 3.f - q * q + 0.5 * q * q * q);
    } else if (q < 2) {
      return 3.f / (2 * M_PIf32) * (1.f / 6.f * powf(2 - q, 3));
    }
    return 0.f;
  }

  // TODO: test
  float dkernel(float q) {
    if (q < 1) {
      return 3.f / (2 * M_PIf32) * (2 * q + 1.5 * q * q);
    } else if (q < 2) {
      return 3.f / (2 * M_PIf32) * (-3.f / 6.f * powf(2 - q, 2));
    }
    return 0.f;
  }

  // TODO: test
  float w_ij(uint32_t i, uint32_t j) {
    float h = fluid_domain.step_size;
    float r = glm::length(particles[i].position - particles[j].position);
    float q = r / h;
    return 1 / powf(h, 3) * kernel(q);
  }

  // TODO: test
  glm::vec3 dw_ij(uint32_t i, uint32_t j) {
    float h = fluid_domain.step_size;
    glm::vec3 d = particles[i].position - particles[j].position;
    float r = glm::length(d);
    float q = r / h;

    return d * (1.f / powf(h, 4.f) * dkernel(q) / r);
  }

  // TODO: test
  void compute_density_presure() {
    float h = fluid_domain.step_size;
    for (uint32_t i = 0; i < particles.size(); ++i) {
      rhos[i] = 0.f;
      std::vector<uint32_t> neighbors;
      find_neighbors(i, neighbors);

      for (uint32_t j : neighbors) {
        // if (i == j)
        //   continue;
        glm::vec3 r = (particles[j].position - particles[i].position);
        float r2 = glm::dot(r, r);
        float h2 = powf(fluid_domain.step_size, 2.f);
        if (r2 < h2) {
          rhos[i] += solver.particle_mass * w_ij(i, j);
        }
      }
      pressures[i] = solver.gas_const * (rhos[i] - solver.rest_density);
    }
  }

  // TODO: test
  void compute_forces() {
    float h = fluid_domain.step_size;
    for (uint32_t i = 0; i < particles.size(); ++i) {
      std::vector<uint32_t> neighbors;
      find_neighbors(i, neighbors);

      glm::vec3 dp = {0, 0, 0};
      glm::vec3 lv = {0, 0, 0};

      for (uint32_t j : neighbors) {
        if (i == j)
          continue;
        glm::vec3 r = (particles[j].position - particles[i].position);
        float r2 = glm::dot(r, r);
        float h2 = powf(fluid_domain.step_size, 2.f);
        if (r2 < h2) {
          glm::vec3 vij = particles[i].velocity - particles[j].velocity;
          glm::vec3 xij = particles[i].position - particles[j].position;
          glm::vec3 dwij = dw_ij(i, j);

          dp = solver.particle_mass *
               (pressures[i] / (rhos[i] * rhos[i]) +
                pressures[j] / (rhos[j] * rhos[j])) *
               dwij;
          lv = solver.particle_mass / rhos[j] * vij * glm::dot(xij, dwij) /
               (glm::dot(xij, xij), 0.01f * h * h);
        }
      }
      dp *= rhos[i];
      lv *= 2;

      glm::vec3 f_pressure = -solver.particle_mass / rhos[i] * dp;
      glm::vec3 f_viscosity = solver.particle_mass * solver.viscosity * lv;
      glm::vec3 f_other = solver.particle_mass * solver.g;
      forces[i] = f_pressure + f_viscosity + f_other;
      // std::cout << forces[i] << "\n";
    }
  }

  // TODO: test, improve
  void integrate() {
    for (uint32_t i = 0; i < particles.size(); ++i) {
      Particle &p = particles[i];
      p.velocity += solver.time_step * forces[i] / rhos[i];
      p.position += solver.time_step * p.velocity;

      if (p.position.x < fluid_domain.corner.x) {
        p.velocity.x *= 0.1;
        p.position.x = fluid_domain.corner.x + solver.eps;
      }
      if (p.position.y < fluid_domain.corner.y) {
        p.velocity.y *= 0.1;
        p.position.y = fluid_domain.corner.y + solver.eps;
      }
      if (p.position.z < fluid_domain.corner.z) {
        p.velocity.z *= 0.1;
        p.position.z = fluid_domain.corner.z + solver.eps;
      }

      if (p.position.x >= fluid_domain.corner.x + fluid_domain.size.x) {
        p.velocity.x *= 0.1;
        p.position.x = fluid_domain.corner.x + fluid_domain.size.x - solver.eps;
      }
      if (p.position.y >= fluid_domain.corner.y + fluid_domain.size.y) {
        p.velocity.y *= 0.1;
        p.position.y = fluid_domain.corner.y + fluid_domain.size.y - solver.eps;
      }
      if (p.position.z >= fluid_domain.corner.z + fluid_domain.size.z) {
        p.velocity.z *= 0.1;
        p.position.z = fluid_domain.corner.z + fluid_domain.size.z - solver.eps;
      }
    }
  }

  void step() {
    update_cell_indices();
    update_particle_ids();
    compute_density_presure();
    compute_forces();
    integrate();
  }
};
