#include "sph_cpu.h"
#include "utils.h"
#include <iostream>

using std::cout;
using std::endl;

std::vector<uint32_t> exclusive_scan(const std::vector<uint32_t> &array) {
  std::vector<uint32_t> output;
  output.push_back(0);
  uint32_t last = 0;
  for (uint32_t i = 0; i < array.size() - 1; ++i) {
    last += array[i];
    output.push_back(last);
  }
  return output;
}

void SPH_CPU::init(uint32_t size) {
  solver_params.g = {0.f, -9.8f, 0.f};
  solver_params.k = 7.f;
  solver_params.gamma = 7.f;
  solver_params.viscosity = 0.f;
  solver_params.eps = 1e-6;

  if (solver_params.particle_size < 1e-6) {
    set_particle_size_and_density(0.1, 1000);
  }
  if (fluid_domain.size.x < solver_params.particle_size ||
      fluid_domain.size.y < solver_params.particle_size ||
      fluid_domain.size.z < solver_params.particle_size) {
    float h = solver_params.particle_size;
    set_domain({0, 0, 0}, {10 * h, 10 * h, 10 * h});
  }

  grid.resize(num_cells());
  grid_start_idx.resize(num_cells());

  positions.resize(size);
  velocities.resize(size);
  velocities_pred.resize(size);
  cell_idx.resize(size);
  sorted_particle_idx.resize(size);
  rho.resize(size);
  rho_pred.resize(size);
  alpha.resize(size);
  pressure.resize(size);
  force.resize(size);

  kappa.resize(size);
  kappav.resize(size);
}

void SPH_CPU::init(const std::vector<glm::vec3> ps) {
  init(ps.size());
  positions = ps;
}

void SPH_CPU::sim_init() {
  update_neighbors();
  update_densities();
  update_factors();
}

void SPH_CPU::step() {
  update_non_pressure_forces();
  update_dt_by_CFL();
  update_predicted_velocities();
  
  correct_density_error();
  update_positions();

  update_neighbors();
  update_densities();
  update_factors();
  // correct_divergence_error();
  update_velocities();
}

void SPH_CPU::update_neighbors() {
  clear_grid();
  update_cell_idx();
  update_sorted_particle_idx();
}

void SPH_CPU::clear_grid() {
  for (uint32_t i = 0, e = num_cells(); i < e; ++i) {
    grid[i] = 0;
  }
}

void SPH_CPU::update_cell_idx() {
  for (uint32_t i = 0; i < size(); ++i) {
    cell_idx[i] = pos2cell_idx(positions[i]);
    grid[cell_idx[i]] += 1;
  }
}

void SPH_CPU::update_sorted_particle_idx() {
  grid_start_idx = exclusive_scan(grid);
  for (uint32_t i = 0; i < size(); ++i) {
    sorted_particle_idx[grid_start_idx[cell_idx[i]]++] = i;
  }
  for (uint32_t i = 0; i < size(); ++i) {
    grid_start_idx[cell_idx[i]] -= 1;
  }
}

void SPH_CPU::update_densities() {
  float h = solver_params.particle_size;
  for (uint32_t i = 0; i < size(); ++i) {
    rho[i] = 0;
    for (uint32_t j : find_neighbors(i)) {
      rho[i] += solver_params.particle_mass * w_ij(positions[i], positions[j], h);
    }
  }
}

void SPH_CPU::update_factors() {
  float h = solver_params.particle_size;
  for (uint32_t i = 0; i < size(); ++i) {
    glm::vec3 vacc = {0, 0, 0};
    float sacc = 0;

    for (uint32_t j : find_neighbors(i)) {
      if (i == j) continue;
      glm::vec3 dw = dw_ij(positions[i], positions[j], h);
      glm::vec3 s = solver_params.particle_mass * dw;
      vacc += s;
      sacc += glm::dot(s, s);
    }
    alpha[i] = rho[i] / (glm::dot(vacc, vacc) + sacc);
  }
}

void SPH_CPU::update_non_pressure_forces(){
  // TODO: add other forces
  for (uint32_t i = 0; i < size(); ++i) {
    force[i] = solver_params.particle_mass * solver_params.g;
  }
}

void SPH_CPU::update_dt_by_CFL(){
  float maxv2 = 0.01;
  for (uint32_t i = 0; i < size(); ++i) {
    float v2 = glm::dot(velocities[i], velocities[i]);
    maxv2 = std::max(v2, maxv2);
  }
  dt = 0.2f * solver_params.particle_size / std::sqrt(maxv2);
}

void SPH_CPU::update_predicted_velocities(){
  for (uint32_t i = 0; i < size(); ++i) {
    velocities_pred[i] = velocities[i] + dt * force[i] / solver_params.particle_mass;
  }
}

void SPH_CPU::correct_density_error(){
  float h = solver_params.particle_size;
  int iter = 0;

  // warm start
  for (uint32_t i = 0; i < size(); ++i) {
    glm::vec3 acc = {0, 0, 0};
    for (uint32_t j : find_neighbors(i)) {
      if (i == j)
        continue;
      acc += solver_params.particle_mass *
             (kappa[i] / rho[i] + kappa[j] / rho[j]) *
             dw_ij(positions[i], positions[j], h);
    }
    velocities_pred[i] -= dt * acc;
  }

  double avg = 0.;
  do {
    // predict density
    for (uint32_t i = 0; i < size(); ++i) {
      float DrhoDt = -rho[i] * divergence(velocities_pred, i);
      rho_pred[i] = rho[i] + dt * DrhoDt; // predict density
    }

    for (uint32_t i = 0; i < size(); ++i) {
      kappa[i] = (rho_pred[i] - solver_params.rest_density) / (dt * dt) * alpha[i];
    }

    for (uint32_t i = 0; i < size(); ++i) {
      glm::vec3 acc = {0, 0, 0};
      for (uint32_t j : find_neighbors(i)) {
        if (i == j) continue;
        acc += solver_params.particle_mass *
               (kappa[i] / rho[i] + kappa[j] / rho[j]) *
               dw_ij(positions[i], positions[j], h);
      }
      printf("dtacc: %f\n", glm::length(dt * acc));
      velocities_pred[i] -= dt * acc;
    }

    double rho_sum = 0;
    for (uint32_t i = 0; i < size(); ++i) {
      rho_sum += rho_pred[i];
    }
    avg = rho_sum / size();

    iter++;
    printf("Average density %f\n", avg);
  } while (iter < 2 || avg - solver_params.rest_density > 10);
}

void SPH_CPU::update_positions() {
  for (uint32_t i = 0; i < size(); ++i) {
    positions[i] += dt * velocities_pred[i];

    if (positions[i].x < fluid_domain.corner.x) {
      velocities_pred[i].x = 0.f;
      positions[i].x = fluid_domain.corner.x + solver_params.eps;
    }
    if (positions[i].y < fluid_domain.corner.y) {
      velocities_pred[i].y = 0.f;
      positions[i].y = fluid_domain.corner.y + solver_params.eps;
    }
    if (positions[i].z < fluid_domain.corner.z) {
      velocities_pred[i].z = 0.f;
      positions[i].z = fluid_domain.corner.z + solver_params.eps;
    }

    if (positions[i].x >= fluid_domain.corner.x + fluid_domain.size.x) {
      velocities_pred[i].x = 0.f;
      positions[i].x = fluid_domain.corner.x + fluid_domain.size.x - solver_params.eps;
    }
    if (positions[i].y >= fluid_domain.corner.y + fluid_domain.size.y) {
      velocities_pred[i].y = 0.f;
      positions[i].y = fluid_domain.corner.y + fluid_domain.size.y - solver_params.eps;
    }
    if (positions[i].z >= fluid_domain.corner.z + fluid_domain.size.z) {
      velocities_pred[i].z = 0.f;
      positions[i].z = fluid_domain.corner.z + fluid_domain.size.z - solver_params.eps;
    }
  }
}

// void SPH_CPU::correct_divergence_error() {
//   float h = solver_params.particle_size;
//   int iter = 0;

//   std::vector<float> kappa(size());
//   double avg = 0;
//   do {
//     double sum = 0;
//     for (uint32_t i = 0; i < size(); ++i) {
//       float DrhoDt = 0;
//       for (uint32_t j : find_neighbors(i)) {
//         if (i == j) continue;
//         DrhoDt += solver_params.particle_mass *
//                   glm::dot(velocities_pred[i] - velocities_pred[j],
//                            dw_ij(positions[i], positions[j], h));
//       }
//       kappa[i] = 1 / dt * DrhoDt * alpha[i];
//       sum += DrhoDt;
//     }

//     // adapt velocities
//     for (uint32_t i = 0; i < size(); ++i) {
//       glm::vec3 acc = {0, 0, 0};
//       for (uint32_t j : find_neighbors(i)) {
//         if (i == j) continue;
//         acc += solver_params.particle_mass *
//                (kappa[i] / rho[i] + kappa[j] / rho[j]) *
//                dw_ij(positions[i], positions[j], h);
//       }
//       velocities_pred[i] -= dt * acc;
//     }
//     avg = sum / size();
//     iter++;
//     printf("Average density change %f\n", avg);
//   } while(avg > 10 && iter < 10);

//   int ss;
//   scanf("%d", &ss);

//   // printf("correct divergence error runs for %d iterations\n", iter);
// }
void SPH_CPU::correct_divergence_error() {
  float h = solver_params.particle_size;
  int iter = 0;

  // warm start
  for (uint32_t i = 0; i < size(); ++i) {
    glm::vec3 acc = {0, 0, 0};
    for (uint32_t j : find_neighbors(i)) {
      if (i == j)
        continue;
      acc += solver_params.particle_mass *
             (kappav[i] / rho[i] + kappav[j] / rho[j]) *
             dw_ij(positions[i], positions[j], h);
    }
    velocities_pred[i] -= dt * acc;
  }

  double avg = 0;
  do {
    double sum = 0;
    for (uint32_t i = 0; i < size(); ++i) {
      float DrhoDt = -rho[i] * divergence(velocities_pred, i);
      kappav[i] = 1 / dt * DrhoDt * alpha[i];
      sum += DrhoDt;
    }

    // adapt velocities
    for (uint32_t i = 0; i < size(); ++i) {
      glm::vec3 acc = {0, 0, 0};
      for (uint32_t j : find_neighbors(i)) {
        if (i == j) continue;
        acc += solver_params.particle_mass *
               (kappav[i] / rho[i] + kappav[j] / rho[j]) *
               dw_ij(positions[i], positions[j], h);
      }
      velocities_pred[i] -= dt * acc;
    }
    avg = sum / size();
    iter++;
    printf("Average density change %f\n", avg);
  } while(avg > 10 && iter < 10);

  int ss;
  scanf("%d", &ss);

  // printf("correct divergence error runs for %d iterations\n", iter);
}

void SPH_CPU::update_velocities() {
  for (uint32_t i = 0; i < size(); ++i) {
    velocities[i] = velocities_pred[i];
  }
}
