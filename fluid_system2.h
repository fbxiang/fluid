#include "fluid_system.h"
#include "utils.h"

class FluidSystem2 : public FluidSystem {

public:
  std::vector<float> alphas;
  std::vector<float> Drho_Dt;
  std::vector<glm::vec3> velocities_intermediate;
  virtual void set_particles(const std::vector<Particle> &ps) override {
    FluidSystem::set_particles(ps);
    alphas.resize(ps.size());
    Drho_Dt.resize(ps.size());
    velocities_intermediate.resize(ps.size());
  }

  void compute_alphas() {
    for (uint32_t i = 0; i < size(); ++i) {
      glm::vec3 v_acc = {0, 0, 0};
      float s_acc = 0;
      for (uint32_t j : neighbors[i]) {
        glm::vec3 v = solver.particle_mass * dw_ij(i, j);
        v_acc += v;
        s_acc += glm::dot(v, v);
      }
      alphas[i] = glm::dot(v_acc, v_acc) + s_acc;
    }
  }

  void compute_Drho_Dt() {
    for (uint32_t i = 0; i < size(); ++i) {
      float acc = 0.f;
      for (uint32_t j : neighbors[i]) {
        acc += solver.particle_mass *
               glm::dot(velocities[i] - velocities[j], dw_ij(i, j));
      }
      Drho_Dt[i] = acc;
    }
  }

  virtual void compute_pressure() override {
    float h = solver.particle_size;
    compute_alphas();
    compute_Drho_Dt();
    for (uint32_t i = 0; i < size(); ++i) {
      float kv = 1.f / solver.time_step * Drho_Dt[i] * rhos[i] / alphas[i];
      pressures[i] = kv * solver.rest_density / solver.gamma *
                     (pow(rhos[i] / solver.rest_density, solver.gamma) - 1);
    }
  }

  virtual void init() override {
    grid = std::vector<uint32_t>(num_cells());

    update_cell_indices();
    update_particle_ids();
    find_neighbors();

    compute_density();
    compute_alphas();
  }

  virtual void step() override {
    compute_non_pressure_forces();
    float dt = compute_adaptive_time_step();
    // TODO: use dt
    compute_intermediate_velocities(solver.time_step);
  }

  void compute_non_pressure_forces() {
    float h = solver.particle_size;
    for (uint32_t i = 0; i < positions.size(); ++i) {
      glm::vec3 lv = {0, 0, 0};
      glm::vec3 f_viscosity =
          solver.particle_mass * solver.viscosity * laplacian(velocities, i);
      glm::vec3 f_other = solver.particle_mass * solver.g;
      forces[i] = f_viscosity + f_other;
    }
  }

  void compute_intermediate_velocities(float dt) {
    for (uint32_t i = 0; i < size(); ++i) {
      velocities_intermediate[i] =
          velocities[i] + dt * forces[i] / solver.particle_mass;
    }
  }
};
