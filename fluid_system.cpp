#include "fluid_system.h"
#include "profiler.h"
#include "utils.h"
#include <thread>

void FluidSystem::set_particle_size(float size) {
  solver.particle_size = size;
  solver.particle_mass = size * size * size * solver.rest_density;
  solver.cell_size = 2 * size;
}

void FluidSystem::set_particles(const std::vector<Particle> &ps) {
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

void FluidSystem::update_cell_indices() {
  for (uint32_t i = 0; i < positions.size(); ++i) {
    glm::uvec3 grid_pos = pos2grid_pos(positions[i]);
    cell_indices[i] = zorder2number(grid_pos);
  }
}

void FluidSystem::update_particle_ids() {
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

void FluidSystem::sort_particles() {
  std::vector<glm::vec3> pos_clone = positions;
  std::vector<glm::vec3> vel_clone = velocities;
  for (uint32_t i = 0; i < particle_ids.size(); ++i) {
    positions[i] = pos_clone[particle_ids[i]];
    velocities[i] = vel_clone[particle_ids[i]];
  }
  update_cell_indices();
  update_particle_ids();
}

std::vector<uint32_t> FluidSystem::find_ids_in_cell(uint32_t cell_idx) const {
  std::vector<uint32_t> out;
  for (uint32_t i = grid[cell_idx];
       i < particle_ids.size() && cell_indices[particle_ids[i]] == cell_idx;
       ++i) {
    out.push_back(particle_ids[i]);
  }
  return out;
}

void FluidSystem::_find_neighbors(uint32_t i) {
  float c = solver.cell_size;
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
        std::vector<uint32_t> ids = find_ids_in_cell(cell_idx);
        // prune
        for (auto j : ids) {
          auto r = positions[i] - positions[j];
          if (glm::dot(r, r) < c * c) {
            neighbors[i].push_back(j);
          }
        }
      }
    }
  }
}

void FluidSystem::find_neighbors() {
  parallel_execution(size(), [=](uint32_t i) { _find_neighbors(i); });
}

void FluidSystem::compute_density() {
  parallel_execution(size(), [=](uint32_t i) {
    rhos[i] = 0.f;
    for (uint32_t j : neighbors[i]) {
      rhos[i] += solver.particle_mass * w_ij(i, j);
    }
  });
}

void FluidSystem::compute_pressure() {
  parallel_execution(size(), [=](uint32_t i) {
    pressures[i] = solver.k * solver.rest_density / solver.gamma *
                   (powf(rhos[i] / solver.rest_density, solver.gamma) - 1.f);
  });
}

void FluidSystem::compute_forces() {
  parallel_execution(size(), [=](uint32_t i) {
    glm::vec3 f_pressure =
        -solver.particle_mass / rhos[i] * gradient(pressures, i);
    glm::vec3 f_viscosity =
        solver.particle_mass * solver.viscosity * laplacian(velocities, i);
    glm::vec3 f_other = solver.particle_mass * solver.g;
    forces[i] = f_pressure + f_viscosity + f_other;
  });
}

float FluidSystem::compute_adaptive_time_step() const {
  std::vector<float> v2(size());
  std::transform(velocities.begin(), velocities.end(), std::back_inserter(v2),
                 [](glm::vec3 v) { return glm::dot(v, v); });
  float vmax = *std::max_element(v2.begin(), v2.end());
  return 0.4 * solver.particle_size / sqrtf(vmax);
}

// TODO: test, improve
void FluidSystem::integrate() {
  parallel_execution(size(), [=](uint32_t i) {
    velocities[i] += solver.time_step * forces[i] / solver.particle_mass;
    positions[i] += solver.time_step * velocities[i];

    if (positions[i].x < fluid_domain.corner.x) {
      velocities[i].x *= 0.f;
      positions[i].x = fluid_domain.corner.x + solver.eps;
    }
    if (positions[i].y < fluid_domain.corner.y) {
      velocities[i].y *= 0.f;
      positions[i].y = fluid_domain.corner.y + solver.eps;
    }
    if (positions[i].z < fluid_domain.corner.z) {
      velocities[i].z *= 0.f;
      positions[i].z = fluid_domain.corner.z + solver.eps;
    }

    if (positions[i].x >= fluid_domain.corner.x + fluid_domain.size.x) {
      velocities[i].x *= 0.f;
      positions[i].x = fluid_domain.corner.x + fluid_domain.size.x - solver.eps;
    }
    if (positions[i].y >= fluid_domain.corner.y + fluid_domain.size.y) {
      velocities[i].y *= 0.f;
      positions[i].y = fluid_domain.corner.y + fluid_domain.size.y - solver.eps;
    }
    if (positions[i].z >= fluid_domain.corner.z + fluid_domain.size.z) {
      velocities[i].z *= 0.f;
      positions[i].z = fluid_domain.corner.z + fluid_domain.size.z - solver.eps;
    }
  });
}

void FluidSystem::step() {
  profiler::start("neighbors");
  update_cell_indices();
  update_particle_ids();
  find_neighbors();
  profiler::stop("neighbors");
  profiler::start("properties");

  // NOTE: Neighbor Validity Check
  // for (int i = 0; i < size(); ++i) {
  //   int k1 = 0;
  //   int k2 = 0;
  //   k1 = neighbors[i].size();
  //   // for (auto j : neighbors[i]) {
  //   //   if (glm::length(particles[i].position - particles[j].position) <
  //   //   solver.particle_size) {
  //   //     ++k1;
  //   //   }
  //   // }
  //   for (int j = 0; j < size(); ++j) {
  //     if (glm::length(positions[i] - positions[j]) <
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
  //     cout << positions[i] << endl;
  //     scanf("%d");
  //   }
  // }
  // cout << "Check passed!" << endl;

  compute_density();
  compute_pressure();
  compute_forces();
  profiler::stop("properties");
  profiler::start("integration");
  integrate();
  profiler::stop("integration");
}

std::vector<uint32_t> FluidSystem::find_neighbors_at(glm::vec3 point) const {
  float h = solver.particle_size;
  glm::uvec3 size = grid_size();
  std::vector<uint32_t> neighbors;
  glm::uvec3 grid_pos = pos2grid_pos(point);
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
        std::vector<uint32_t> ids = find_ids_in_cell(cell_idx);
        for (auto j : ids) {
          auto r = point - positions[j];
          if (glm::dot(r, r) < h * h) {
            neighbors.push_back(j);
          }
        }
      }
    }
  }
  return neighbors;
}

bool FluidSystem::check_neighbors_at(glm::vec3 point) const {
  float h = solver.particle_size;
  glm::uvec3 size = grid_size();
  std::vector<uint32_t> neighbors;
  glm::uvec3 grid_pos = pos2grid_pos(point);
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
        std::vector<uint32_t> ids = find_ids_in_cell(cell_idx);

        for (uint32_t i = grid[cell_idx];
             i < particle_ids.size() &&
             cell_indices[particle_ids[i]] == cell_idx;
             ++i) {
          auto r = point - positions[particle_ids[i]];
          if (glm::dot(r, r) < h * h) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

float FluidSystem::compute_density_at(glm::vec3 point) const {
  float rho = 0.f;
  for (uint32_t j: find_neighbors_at(point)) {
    rho += w_ij(point, positions[j]) * solver.particle_mass;
  }
  return rho;
}

glm::vec3 FluidSystem::compute_density_gradient_at(glm::vec3 point, float point_density) const {
  glm::vec3 value = {0.f, 0.f, 0.f};
  for (uint32_t j : find_neighbors_at(point)) {
    value +=
        solver.particle_mass *
        (1 / point_density + 1 / rhos[j]) *
        dw_ij(point, positions[j]);
  }

  return point_density * value;
}
