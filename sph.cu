#include "sph.cuh"
#include "sph_math.h"
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include "marching_cube_table.h"
#include "random.h"
#include "profiler.h"

#define N_THREADS 1024
#define CUDA_CHECK_RETURN(value)                                               \
  {                                                                            \
    cudaError_t _m_cudaStat = value;                                           \
    if (_m_cudaStat != cudaSuccess) {                                          \
      fprintf(stderr, "Error %s at line %d in file %s\n",                      \
              cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);            \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

namespace sph {

__constant__ SolverParams params;
SolverParams h_params;
__constant__ FluidDomain domain;
FluidDomain h_domain;
__constant__ glm::ivec3 grid_size;
glm::ivec3 h_grid_size;

__device__ int *grid;
int *d_grid;

__device__ int *grid_start_idx;
int *d_grid_start_idx;

__device__ glm::vec3 *positions;
glm::vec3 *d_positions;

__device__ glm::vec3 *velocities;
glm::vec3 *d_velocities;

__device__ glm::vec3 *force;
glm::vec3 *d_force;

__device__ int *cell_idx;
int *d_cell_idx;

__device__ int *sorted_particle_idx;
int *d_sorted_particle_idx;

__device__ float *rho;
float *d_rho;

__device__ float* tmp_float;
float* d_tmp_float;

__device__ glm::vec3 *velocities_pred;
glm::vec3 *d_velocities_pred;

__device__ glm::vec3 *positions_pred;
glm::vec3 *d_positions_pred;

__device__ float* pressure;
float* d_pressure;

__device__ glm::vec3 *non_pressure_force;  // used in PCISPH
glm::vec3 *d_non_pressure_force;

__device__ glm::vec3 *pressure_force;  // used in PCISPH
glm::vec3 *d_pressure_force;

__device__ float dt;

__device__ glm::vec3 *color_grad;  // particle normal
glm::vec3 *d_color_grad;

__device__ float *air_potential;
float *d_air_potential;

__device__ float *air_energy;
float *d_air_energy;

__device__ glm::vec3 *visual_debug;
glm::vec3 *d_visual_debug;



int max_num_particles;
int num_particles;
int num_cells;

void cuda_clear(void* d_field, size_t elem_size) {
  cudaMemset(d_field, 0, num_particles * elem_size);
}

__device__ float psi(float I, float tau_min, float tau_max) {
  return (min(I, tau_max) - min(I, tau_min)) / (tau_max - tau_min);
}

#define FOR_NEIGHBORS(...)                                                     \
  {                                                                            \
    float c = params.cell_size;                                                \
    glm::ivec3 gpos = pos2grid_pos(positions[i]);                              \
    for (int x = -1; x <= 1; ++x) {                                            \
      if (x == -1 && gpos.x == 0 || x == 1 && gpos.x == grid_size.x - 1)       \
        continue;                                                              \
      for (int y = -1; y <= 1; ++y) {                                          \
        if (y == -1 && gpos.y == 0 || y == 1 && gpos.y == grid_size.y - 1)     \
          continue;                                                            \
        for (int z = -1; z <= 1; ++z) {                                        \
          if (z == -1 && gpos.z == 0 || z == 1 && gpos.z == grid_size.y - 1)   \
            continue;                                                          \
          int cell_idx = grid_pos2cell_idx(                                    \
              glm::ivec3(gpos.x + x, gpos.y + y, gpos.z + z));                 \
          for (int k = grid_start_idx[cell_idx], e = k + grid[cell_idx];       \
               k < e; ++k) {                                                   \
            int j = sorted_particle_idx[k];                                    \
            glm::vec3 r = positions[i] - positions[j];                         \
            if (glm::dot(r, r) < c * c) {                                      \
              __VA_ARGS__                                                      \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }


__device__ glm::ivec3 pos2grid_pos(glm::vec3 pos) {
  return (pos - domain.corner) / params.cell_size;
}

__device__ int grid_pos2cell_idx(glm::ivec3 gpos) {
  return gpos.x * grid_size.y * grid_size.z + gpos.y * grid_size.z + gpos.z;
}

__device__ int pos2cell_idx(glm::vec3 pos) {
  return grid_pos2cell_idx(pos2grid_pos(pos));
}

__device__ glm::vec3 gradient(float *quantity, int i) {
  glm::vec3 value = {0,0,0};

  FOR_NEIGHBORS(
      if (i == j) { continue; }
      value += params.particle_mass *
      (quantity[i] / rho[i] / rho[i] + quantity[j] / rho[j] / rho[j]) *
      dw_ij(positions[i], positions[j], params.particle_size););
  return rho[i] * value;
}


__device__ glm::vec3 laplacian(glm::vec3 *quantity, int i) {
  float h = params.particle_size;
  glm::vec3 value = {0,0,0};
  FOR_NEIGHBORS(
      if (i == j) {continue;}
      glm::vec3 xij = positions[i] - positions[j];
      value += (quantity[i] - quantity[j]) *
      (params.particle_mass / rho[j] * glm::dot(xij, dw_ij(positions[i], positions[j], h)) /
       (glm::dot(xij, xij) + 0.01f * h * h));
                );
  return 2.f * value;
}

__device__ float divergence(glm::vec3 *vec, int i) {
  float value = 0;
  FOR_NEIGHBORS(if (i == j) { continue; } value +=
                params.particle_mass *
                glm::dot(vec[i] - vec[j], dw_ij(positions[i], positions[j],
                                                params.particle_size)););
  value /= -rho[i];
  return value;
}


__global__ void _update_cell_idx(int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    cell_idx[i] = pos2cell_idx(positions[i]);
    atomicAdd(&grid[cell_idx[i]], 1);
  }
}
__global__ void _update_sorted_particle_idx(int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int idx = atomicAdd(&grid_start_idx[cell_idx[i]], 1);
    sorted_particle_idx[idx] = i;
  }
}
__global__ void _restore_start_idx(int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicSub(&grid_start_idx[cell_idx[i]], 1);
  }
}

__global__ void _velocity_to_speed(float* d_speed, glm::vec3 *d_vel, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    d_speed[i] = glm::length(d_vel[i]);
  }
}

bool add_particles(glm::vec3 *in_positions, int size) {
  if (num_particles + size > max_num_particles)
    return false;
  cudaMemcpy(&d_positions[num_particles], in_positions, size * sizeof(glm::vec3),
             cudaMemcpyHostToDevice);
  cudaMemset(&d_velocities[num_particles], 0, size * sizeof(glm::vec3));
  num_particles += size;
  return true;
}

bool get_particles(glm::vec3 *out_positions, int size) {
  if (size <= 0 || size > num_particles) {
    return false;
  }
  cudaMemcpy(out_positions, d_positions, size * sizeof(glm::vec3),
             cudaMemcpyDeviceToHost);
  return true;
}

// all solvers
void update_neighbors() {
  // clear grid
  CUDA_CHECK_RETURN(cudaMemset(d_grid, 0, num_cells * sizeof(int)));

  // update cell idx
  _update_cell_idx<<<(num_particles + N_THREADS - 1) / N_THREADS, N_THREADS>>>(
      num_particles);
  // update sorted particle idx
  // 1. scan
  thrust::device_ptr<int> grid_ptr = thrust::device_pointer_cast(d_grid);
  thrust::device_ptr<int> grid_start_idx_ptr =
      thrust::device_pointer_cast(d_grid_start_idx);
  thrust::exclusive_scan(grid_ptr, grid_ptr + num_cells, grid_start_idx_ptr);
  // 2. update
  _update_sorted_particle_idx<<<(num_particles + N_THREADS - 1) / N_THREADS,
                                N_THREADS>>>(num_particles);
  // 3. restore
  _restore_start_idx<<<(num_particles + N_THREADS - 1) / N_THREADS,
                       N_THREADS>>>(num_particles);
}

// update dt using velocities
float update_dt_by_CFL() {
  _velocity_to_speed<<<(num_particles + N_THREADS - 1) / N_THREADS, N_THREADS>>>(d_tmp_float, d_velocities, num_particles);
  thrust::device_ptr<float> speed_ptr = thrust::device_pointer_cast(d_tmp_float);
  float max_speed = *thrust::max_element(speed_ptr, speed_ptr + num_particles);
  float dt = 0.2 * h_params.particle_size / max_speed;
  if (dt > 0.003) dt = 0.003;
  if (dt < 0.00005) dt = 0.00005;
  cudaMemcpyToSymbol(sph::dt, &dt, sizeof(float));
  return dt;
}

// update dt using velocities_pred
float update_dt_by_CFL_pred() {
  _velocity_to_speed<<<(num_particles + N_THREADS - 1) / N_THREADS, N_THREADS>>>(d_tmp_float, d_velocities_pred, num_particles);
  thrust::device_ptr<float> speed_ptr = thrust::device_pointer_cast(d_tmp_float);
  float max_speed = *thrust::max_element(speed_ptr, speed_ptr + num_particles);
  float dt = 0.4 * h_params.particle_size / max_speed;
  if (dt > 0.01) dt = 0.01;
  if (dt < 0.00005) dt = 0.00005;
  cudaMemcpyToSymbol(sph::dt, &dt, sizeof(float));
  return dt;
}


__global__ void _update_density(int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float value = 0;
    FOR_NEIGHBORS(
        value += params.particle_mass * w_ij(positions[i], positions[j], params.particle_size);
                 );
    rho[i] = value;
  }
}
// update rho using positions
void update_density() {
  _update_density<<<(num_particles + N_THREADS - 1) / N_THREADS, N_THREADS>>>(num_particles);
}


__global__ void _update_density_increment_pressure(int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float rho_pred = 0;
    FOR_NEIGHBORS(
        rho_pred += params.particle_mass * w_ij(positions_pred[i], positions_pred[j], params.particle_size);
                 );

    // update pressure
    float rho_err = rho_pred - params.rest_density;
    float beta = dt*dt*params.particle_mass*params.particle_mass*2.f/(params.rest_density * params.rest_density);
    glm::vec3 vacc = {0, 0, 0};
    float sacc = 0;
    FOR_NEIGHBORS(
        if (i == j) continue;
        glm::vec3 dw = dw_ij(positions_pred[i], positions_pred[j], params.particle_size);
        vacc += dw;
        sacc += glm::dot(dw, dw);
                  );
    pressure[i] += rho_err / (beta * (glm::dot(vacc, vacc) + sacc));
    if (pressure[i] < 0) pressure[i] = 0;
  }
}
// increment pressure using rho_pred computed from positions_pred
void update_density_increment_pressure() {
  _update_density_increment_pressure<<<(num_particles + N_THREADS - 1) / N_THREADS, N_THREADS>>>(num_particles);
}

__global__ void _update_pressure_force(int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    pressure_force[i] = -params.particle_mass / rho[i] * gradient(pressure, i);
    force[i] = pressure_force[i] + non_pressure_force[i]; 
  }
}
// pci solver, update pressure force and net-force from current pressure
void update_pressure_force() {
  _update_pressure_force<<<(num_particles + N_THREADS - 1) / N_THREADS, N_THREADS>>>(num_particles);
}

__global__ void _update_pressure(int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    pressure[i] = params.k * params.rest_density / params.gamma *
                  (powf(rho[i] / params.rest_density, params.gamma) - 1.f);
  }
}
__global__ void _update_all_forces(int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    glm::vec3 f_pressure = -params.particle_mass / rho[i] * gradient(pressure, i);
    glm::vec3 f_viscosity =
        params.particle_mass * params.viscosity * laplacian(velocities, i);
    glm::vec3 f_gravity = params.particle_mass * params.g;
    force[i] = f_pressure + f_viscosity + f_gravity;
  }
}
// only in regular solver, compute force directly from rho
void update_all_forces() {
  _update_pressure<<<(num_particles + N_THREADS - 1) / N_THREADS, N_THREADS>>>(num_particles);
  _update_all_forces<<<(num_particles + N_THREADS - 1) / N_THREADS, N_THREADS>>>(num_particles);
}


// regular solver
__global__ void _update_velocity_position(int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    velocities[i] += sph::dt * force[i] / params.particle_mass;
    positions[i] += sph::dt * velocities[i];

    if (positions[i].x < domain.corner.x + params.eps) {
      velocities[i].x = 0.f;
      positions[i].x = domain.corner.x + params.eps;
    }
    if (positions[i].y < domain.corner.y + params.eps) {
      velocities[i].y = 0.f;
      positions[i].y = domain.corner.y + params.eps;
    }
    if (positions[i].z < domain.corner.z + params.eps) {
      velocities[i].z = 0.f;
      positions[i].z = domain.corner.z + params.eps;
    }

    if (positions[i].x >= domain.corner.x + domain.size.x - params.eps) {
      velocities[i].x = 0.f;
      positions[i].x = domain.corner.x + domain.size.x - params.eps;
    }
    if (positions[i].y >= domain.corner.y + domain.size.y - params.eps) {
      velocities[i].y = 0.f;
      positions[i].y = domain.corner.y + domain.size.y - params.eps;
    }
    if (positions[i].z >= domain.corner.z + domain.size.z - params.eps) {
      velocities[i].z = 0.f;
      positions[i].z = domain.corner.z + domain.size.z - params.eps;
    }
  }
}

__global__ void _update_color_grad(int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    glm::vec3 value;
    FOR_NEIGHBORS(
        if (i == j) continue;
        value += params.particle_mass / rho[j] * dw_ij(positions[i], positions[j], params.particle_size);
                  );
    color_grad[i] = value;
  }
}

__global__ void _update_visual(int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v_diff = 0;
    float curvature = 0;
    glm::vec3 ni = glm::normalize(color_grad[i]);
    FOR_NEIGHBORS(
        if (i == j) continue;
        // v_diff
        glm::vec3 vij = velocities[i]-velocities[j];
        glm::vec3 xij = positions[i]-positions[j];
        float kn = kernel2(glm::length(xij) / params.particle_size);
        if (glm::length(vij) > 1e-6 && glm::length(xij) > 1e-6) {
          v_diff += glm::length(vij) * (1 - glm::dot(glm::normalize(vij), glm::normalize(xij))) * kn;
        }

        glm::vec3 nj = glm::normalize(color_grad[j]);
        curvature += (1 - glm::dot(ni, nj)) * kn * (glm::dot(xij, ni) > 0);
                  );
    glm::vec3 normal = glm::normalize(color_grad[i]);
    glm::vec3 vel_dir = glm::normalize(velocities[i]);
    bool crest = glm::length(color_grad[i]) > 10 && glm::dot(normal, vel_dir) > 0.6;

    float energy = glm::dot(velocities[i], velocities[i]);

    // v_diff
    float Ita = psi(v_diff, 10, 40);
    // curvature * crest
    float Iwc = psi(curvature * crest, 2, 8);
    // energy
    float Ik = psi(energy, 0, 1);
    float potential = Ik * (3000 * Ita + 1000 * Iwc) * dt;
    air_potential[i] = potential;
    air_energy[i] = Ik;

    float test = potential / dt;
    visual_debug[i] = glm::vec3(1, 1-test, 1-test);
  }
}

void update_velocity_position() {
  _update_velocity_position<<<(num_particles + N_THREADS - 1) / N_THREADS, N_THREADS>>>(num_particles);
  _update_color_grad<<<(num_particles + N_THREADS - 1) / N_THREADS, N_THREADS>>>(num_particles);
  _update_visual<<<(num_particles + N_THREADS - 1) / N_THREADS, N_THREADS>>>(num_particles);
}

__global__ void _update_non_pressure_forces(int n) {
  float E = 0.01;
  float F = 0.1;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    glm::vec3 f_boundary = {0,0,0};

    if (positions[i].x < domain.corner.x + E) {
      f_boundary.x += F;
    }
    if (positions[i].y < domain.corner.y + E) {
      f_boundary.y += F;
    }
    if (positions[i].z < domain.corner.z + E) {
      f_boundary.z += F;
    }
    if (positions[i].x >= domain.corner.x + domain.size.x - E) {
      f_boundary.x -= F;
    }
    if (positions[i].y >= domain.corner.y + domain.size.y - E) {
      f_boundary.y -= F;
    }
    if (positions[i].z >= domain.corner.z + domain.size.z - E) {
      f_boundary.z -= F;
    }

    glm::vec3 f_gravity = params.particle_mass * params.g;

    glm::vec3 f_viscosity =
        params.particle_mass * params.viscosity * laplacian(velocities, i);

    non_pressure_force[i] = f_gravity + f_boundary + f_viscosity;
    force[i] = non_pressure_force[i] + pressure_force[i];
  }
}
void update_non_pressure_forces() {
  _update_non_pressure_forces<<<(num_particles + N_THREADS - 1) / N_THREADS, N_THREADS>>>(num_particles);
}

__global__ void _update_positions(int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    positions[i] += sph::dt * velocities_pred[i];

    if (positions[i].x < domain.corner.x + params.eps) {
      velocities[i].x = 0.f;
      positions[i].x = domain.corner.x + params.eps;
    }
    if (positions[i].y < domain.corner.y + params.eps) {
      velocities[i].y = 0.f;
      positions[i].y = domain.corner.y + params.eps;
    }
    if (positions[i].z < domain.corner.z + params.eps) {
      velocities[i].z = 0.f;
      positions[i].z = domain.corner.z + params.eps;
    }

    if (positions[i].x >= domain.corner.x + domain.size.x - params.eps) {
      velocities[i].x = 0.f;
      positions[i].x = domain.corner.x + domain.size.x - params.eps;
    }
    if (positions[i].y >= domain.corner.y + domain.size.y - params.eps) {
      velocities[i].y = 0.f;
      positions[i].y = domain.corner.y + domain.size.y - params.eps;
    }
    if (positions[i].z >= domain.corner.z + domain.size.z - params.eps) {
      velocities[i].z = 0.f;
      positions[i].z = domain.corner.z + domain.size.z - params.eps;
    }
  }
}
void update_positions() {
  _update_positions<<<(num_particles + N_THREADS - 1) / N_THREADS, N_THREADS>>>(num_particles);
}

__global__ void _update_velocity_pred(int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    velocities_pred[i] = velocities[i] + dt * non_pressure_force[i] / params.particle_mass;
  }
}
/* Predict velocity only */
void update_velocity_pred() {
  _update_velocity_pred<<<(num_particles + N_THREADS - 1) / N_THREADS, N_THREADS>>>(num_particles);
}

__global__ void _update_velocity_position_pred(int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    velocities_pred[i] = velocities[i] + sph::dt * force[i] / params.particle_mass;
    positions_pred[i] = positions[i] + sph::dt * velocities_pred[i];

    if (positions_pred[i].x < domain.corner.x + params.eps) {
      velocities_pred[i].x = 0.f;
      positions_pred[i].x = domain.corner.x + params.eps;
    }
    if (positions_pred[i].y < domain.corner.y + params.eps) {
      velocities_pred[i].y = 0.f;
      positions_pred[i].y = domain.corner.y + params.eps;
    }
    if (positions_pred[i].z < domain.corner.z + params.eps) {
      velocities_pred[i].z = 0.f;
      positions_pred[i].z = domain.corner.z + params.eps;
    }

    if (positions_pred[i].x >= domain.corner.x + domain.size.x - params.eps) {
      velocities_pred[i].x = 0.f;
      positions_pred[i].x = domain.corner.x + domain.size.x - params.eps;
    }
    if (positions_pred[i].y >= domain.corner.y + domain.size.y - params.eps) {
      velocities_pred[i].y = 0.f;
      positions_pred[i].y = domain.corner.y + domain.size.y - params.eps;
    }
    if (positions_pred[i].z >= domain.corner.z + domain.size.z - params.eps) {
      velocities_pred[i].z = 0.f;
      positions_pred[i].z = domain.corner.z + domain.size.z - params.eps;
    }
  }
}
/* Predict both velocity and position */
void update_velocity_position_pred() {
  _update_velocity_position_pred<<<(num_particles + N_THREADS - 1) / N_THREADS, N_THREADS>>>(num_particles);
}

void update_velocity() {
  cudaMemcpy(d_velocities, d_velocities_pred,
             num_particles * sizeof(glm::vec3),
             cudaMemcpyDeviceToDevice);
}

float step_regular() {
  profiler::start("neighbors");
  update_neighbors();
  profiler::stop("neighbors");

  profiler::start("density_pressure");
  float dt = update_dt_by_CFL();
  update_density();
  update_all_forces();
  profiler::stop("density_pressure");

  profiler::start("integration");
  update_velocity_position();
  profiler::stop("integration");

  return dt;
}

float pci_step() {
  update_neighbors();

  cuda_clear(d_pressure_force, sizeof(glm::vec3));
  cuda_clear(d_pressure, sizeof(float));
  update_density();
  update_non_pressure_forces();

  update_velocity_position_pred();
  float dt = update_dt_by_CFL_pred();

  for (int iter = 0; iter < 5; ++iter) {
    update_density_increment_pressure();
    update_pressure_force();
    update_velocity_position_pred();
  }
  update_velocity_position();

  return dt;
}

__global__ void _update_debug_points(float* vbo, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    vbo[6 * i    ] = positions[i].x;
    vbo[6 * i + 1] = positions[i].y;
    vbo[6 * i + 2] = positions[i].z;

    vbo[6 * i + 3] = visual_debug[i].x;
    vbo[6 * i + 4] = visual_debug[i].y;
    vbo[6 * i + 5] = visual_debug[i].z;
  }
}

void update_debug_points(float* vbo) {
  _update_debug_points<<<(num_particles + N_THREADS - 1) / N_THREADS, N_THREADS>>>(vbo, num_particles);
}


void init(const SolverParams &in_params, const FluidDomain &in_domain,
          const int max_num_particles) {
  cudaMemcpyToSymbol(sph::params, &in_params, sizeof(SolverParams));
  h_params = in_params;
  cudaMemcpyToSymbol(sph::domain, &in_domain, sizeof(FluidDomain));
  h_domain = in_domain;

  glm::ivec3 grid_size = glm::ivec3(ceil(in_domain.size.x / in_params.cell_size),
                                    ceil(in_domain.size.y / in_params.cell_size),
                                    ceil(in_domain.size.z / in_params.cell_size));
  h_grid_size = grid_size;
  cudaMemcpyToSymbol(sph::grid_size, &grid_size, sizeof(glm::ivec3));

  sph::num_cells = grid_size.x * grid_size.y * grid_size.z;
  sph::max_num_particles = max_num_particles;
  sph::num_particles = 0;

  // common quantities
  CUDA_CHECK_RETURN(cudaMalloc(&d_grid, num_cells * sizeof(int)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(sph::grid, &d_grid, sizeof(void *), 0,
                                       cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc(&d_grid_start_idx, num_cells * sizeof(int)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(sph::grid_start_idx, &d_grid_start_idx,
                                       sizeof(void *), 0,
                                       cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc(&d_positions, max_num_particles * sizeof(glm::vec3)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(sph::positions, &d_positions, sizeof(void *),
                                       0, cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc(&d_velocities, max_num_particles * sizeof(glm::vec3)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(
      sph::velocities, &d_velocities, sizeof(void *), 0, cudaMemcpyHostToDevice));


  CUDA_CHECK_RETURN(cudaMalloc(&d_force, max_num_particles * sizeof(glm::vec3)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(
      sph::force, &d_force, sizeof(void *), 0, cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc(&d_pressure_force, max_num_particles * sizeof(glm::vec3)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(
      sph::pressure_force, &d_pressure_force, sizeof(void *), 0, cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc(&d_non_pressure_force, max_num_particles * sizeof(glm::vec3)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(
      sph::non_pressure_force, &d_non_pressure_force, sizeof(void *), 0, cudaMemcpyHostToDevice));


  CUDA_CHECK_RETURN(cudaMalloc(&d_cell_idx, max_num_particles * sizeof(int)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(sph::cell_idx, &d_cell_idx, sizeof(void *),
                                       0, cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc(&d_sorted_particle_idx, max_num_particles * sizeof(int)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(sph::sorted_particle_idx, &d_sorted_particle_idx,
                                       sizeof(void *), 0,
                                       cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc(&d_rho, max_num_particles * sizeof(float)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(sph::rho, &d_rho, sizeof(void *), 0,
                                       cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc(&d_tmp_float, max_num_particles * sizeof(float)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(sph::tmp_float, &d_tmp_float, sizeof(void *), 0,
                                       cudaMemcpyHostToDevice));
  
  // regular solver
  CUDA_CHECK_RETURN(cudaMalloc(&d_pressure, max_num_particles * sizeof(float)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(sph::pressure, &d_pressure, sizeof(void *), 0,
                                       cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc(&d_velocities_pred, max_num_particles * sizeof(glm::vec3)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(sph::velocities_pred, &d_velocities_pred, sizeof(void *), 0,
                                       cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc(&d_positions_pred, max_num_particles * sizeof(glm::vec3)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(sph::positions_pred, &d_positions_pred, sizeof(void *), 0,
                                       cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc(&d_color_grad, max_num_particles * sizeof(glm::vec3)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(sph::color_grad, &d_color_grad, sizeof(void *), 0,
                                       cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc(&d_air_potential, max_num_particles * sizeof(float)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(sph::air_potential, &d_air_potential, sizeof(void *), 0,
                                       cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc(&d_air_energy, max_num_particles * sizeof(float)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(sph::air_energy, &d_air_energy, sizeof(void *), 0,
                                       cudaMemcpyHostToDevice));


  CUDA_CHECK_RETURN(cudaMalloc(&d_visual_debug, max_num_particles * sizeof(glm::vec3)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(sph::visual_debug, &d_visual_debug, sizeof(void *), 0,
                                       cudaMemcpyHostToDevice));

}

int get_num_particles() {
  return num_particles;
}

void log() {
  printf("\n");
}
void log(const char* key) {
  printf("%s\n", key);
}
void log(const char* key, int value) {
  printf("%-30s %d\n", key, value);
}
void log(const char* key, float value) {
  printf("%-30s %.3f\n", key, value);
}
void log(const char* key, glm::vec3 value) {
  printf("%-30s %f %f %f\n", key, value.x, value.y, value.z);
}
void log(const char* key, glm::ivec3 value) {
  printf("%-30s %d %d %d\n", key, value.x, value.y, value.z);
}
void log_array(const char* key, int* array, int size) {
  printf("%s\n", key);
  int i = 0;
  while (i < size) {
    for (int j = 0; j < 10 && i < size; ++j) {
      printf("%6d ", array[i++]);
    }
  }
  printf("\n");
}

void print_summary() {
  log("Simulation State");
  log("Max size", max_num_particles);
  log("Current size", num_particles);
  log();

  log("Parameters");
  SolverParams h_params;
  cudaMemcpyFromSymbol(&h_params, sph::params, sizeof(SolverParams));
  log("Particle size", h_params.particle_size);
  log("Cell size", h_params.cell_size);
  log("Rest density", h_params.rest_density);
  log();

  log("Domain");
  FluidDomain h_domain;
  cudaMemcpyFromSymbol(&h_domain, sph::domain, sizeof(FluidDomain));
  log("Corner", h_domain.corner);
  log("Size", h_domain.size);

  glm::ivec3 h_grid_size;
  cudaMemcpyFromSymbol(&h_grid_size, sph::grid_size, sizeof(glm::ivec3));
  log("Grid size", h_grid_size);
  log("Number of cells", num_cells);
  log();
}


namespace visual {

#define VISUAL_FOR_NEIGHBORS(...)                                              \
  {                                                                            \
    float c = params.cell_size;                                                \
    glm::ivec3 gpos = pos2grid_pos(visual::positions[i]);                      \
    for (int x = -1; x <= 1; ++x) {                                            \
      if (x == -1 && gpos.x == 0 || x == 1 && gpos.x == grid_size.x - 1)       \
        continue;                                                              \
      for (int y = -1; y <= 1; ++y) {                                          \
        if (y == -1 && gpos.y == 0 || y == 1 && gpos.y == grid_size.y - 1)     \
          continue;                                                            \
        for (int z = -1; z <= 1; ++z) {                                        \
          if (z == -1 && gpos.z == 0 || z == 1 && gpos.z == grid_size.y - 1)   \
            continue;                                                          \
          int cell_idx = grid_pos2cell_idx(                                    \
              glm::ivec3(gpos.x + x, gpos.y + y, gpos.z + z));                 \
          for (int k = grid_start_idx[cell_idx], e = k + grid[cell_idx];       \
               k < e; ++k) {                                                   \
            int j = sorted_particle_idx[k];                                    \
            glm::vec3 r = visual::positions[i] - sph::positions[j];            \
            if (glm::dot(r, r) < c * c) {                                      \
              __VA_ARGS__                                                      \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }


__device__ glm::vec3 *positions;
glm::vec3 *d_positions;

__device__ glm::vec3 *velocities;
glm::vec3 *d_velocities;

__device__ float *life;
float *d_life;

__device__ int *visual_types;
int *d_visual_types;

__device__ unsigned int *rand_seeds;
unsigned int *d_rand_seeds;

__constant__ int max_visual = 100000;
constexpr int h_max_visual = 100000;
__device__ int visual_count = 0;
int h_visual_count = 0;

int get_num_visual_particles() {
  return h_visual_count;
}

// Call add_particle on each fluid particle to add diffuse particle
__device__ void add_visual_particle(glm::vec3 position, glm::vec3 velocity, float life) {
  int idx = atomicAdd(&visual_count, 1);
  if (visual_count > max_visual) {
    atomicAdd(&visual_count, -1);
    return;
  }
  visual::positions[idx] = position;
  visual::velocities[idx] = velocity;
  visual::life[idx] = life * 5 + rnd(rand_seeds[idx]) * 2;
}
__global__ void _add_visual_particles(int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    if (rnd(rand_seeds[i]) < sph::air_potential[i]) {
      glm::vec3 p = sph::positions[i];
      glm::vec3 v = sph::velocities[i];

      glm::vec3 n = glm::normalize(v);
      glm::vec3 e1, e2;
      if (n.y < 0.9) {
        e1 = glm::normalize(glm::cross(glm::vec3(0,1,0), n));
        e2 = glm::cross(n, e1);
      } else {
        e1 = glm::normalize(glm::cross(glm::vec3(1,0,0), n));
        e2 = glm::cross(n, e1);
      }

      float r = rnd(rand_seeds[i]) * params.particle_size / 2.f;
      float h = rnd(rand_seeds[i]) * dt * glm::length(v);
      float theta = rnd(rand_seeds[i]) * M_PIf * 2.f;

      float dh = h;
      float dx = r * cos(theta);
      float dy = r * sin(theta);

      add_visual_particle(p + dx * e1 + dy * e2 + dh * n, v + dx * e1 + dy * e2, air_energy[i]);
    }
  }
}

void add_visual_particles() {
  // add visual particles
  _add_visual_particles<<<(num_particles + N_THREADS - 1) / N_THREADS, N_THREADS>>>(num_particles);

  // update host visual count
  CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&h_visual_count, visual_count, sizeof(int)));
}

__global__ void _update_visual_particles(int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    glm::vec3 vf(0.f);
    float weights = 0;
    int count = 0;
    VISUAL_FOR_NEIGHBORS(
        ++count;
        float kn = kernel(glm::length(visual::positions[i] - sph::positions[j])/params.particle_size);
        weights += kn;
        vf += sph::velocities[j] * kn;
                         );
    vf /= weights;
    float kb = 0.2;
    float kd = 0.8;

    if (count < 12) {
      // spray
      visual_types[i] = 0;
      velocities[i] += params.g * dt;
    } else if (count > 35) {
      // bubble
      visual_types[i] = 1;
      velocities[i] += (-kb * params.g + kd * (vf-velocities[i])/dt) * dt;
    } else {
      // foam
      visual_types[i] = 2;
      velocities[i] = vf;
      life[i] -= dt;
    }
    positions[i] += velocities[i] * dt;

    // force boundary
    if (positions[i].x < domain.corner.x + params.eps) {
      velocities[i].x = 0.f;
      positions[i].x = domain.corner.x + params.eps;
    }
    if (positions[i].y < domain.corner.y + params.eps) {
      velocities[i].y = 0.f;
      positions[i].y = domain.corner.y + params.eps;
    }
    if (positions[i].z < domain.corner.z + params.eps) {
      velocities[i].z = 0.f;
      positions[i].z = domain.corner.z + params.eps;
    }

    if (positions[i].x >= domain.corner.x + domain.size.x - params.eps) {
      velocities[i].x = 0.f;
      positions[i].x = domain.corner.x + domain.size.x - params.eps;
    }
    if (positions[i].y >= domain.corner.y + domain.size.y - params.eps) {
      velocities[i].y = 0.f;
      positions[i].y = domain.corner.y + domain.size.y - params.eps;
    }
    if (positions[i].z >= domain.corner.z + domain.size.z - params.eps) {
      velocities[i].z = 0.f;
      positions[i].z = domain.corner.z + domain.size.z - params.eps;
    }
  }
}
void update_visual_particles() {
  if (h_visual_count <= 0) return;
  _update_visual_particles<<<(h_visual_count + N_THREADS - 1) / N_THREADS, N_THREADS>>>(h_visual_count);
}

__global__ void _init_rand_seeds(int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    rand_seeds[i] = tea<16>(i, 0);
  }
}

struct _negative_pred : public thrust::unary_function<float,bool>
{
  __host__ __device__ float operator()(float x) { return x <= 0; }
};

void clear_visual_particles() {
  thrust::device_ptr<glm::vec3> positions_ptr = thrust::device_pointer_cast(d_positions);
  thrust::device_ptr<glm::vec3> velocities_ptr = thrust::device_pointer_cast(d_velocities);
  thrust::device_ptr<float> life_ptr = thrust::device_pointer_cast(d_life);
  if (h_visual_count > 0) {
    int new_count = thrust::remove_if(positions_ptr, positions_ptr + h_visual_count,
                                      life_ptr, _negative_pred()) - positions_ptr;
    thrust::remove_if(velocities_ptr, velocities_ptr + h_visual_count, life_ptr, _negative_pred());
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(visual::visual_count, &new_count, sizeof(int)));
    h_visual_count = new_count;
  }
}

void visual_step() {
  add_visual_particles();
  update_visual_particles();
  clear_visual_particles();
}

void init() {
  CUDA_CHECK_RETURN(cudaMalloc(&visual::d_positions, h_max_visual * sizeof(glm::vec3)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(visual::positions, &visual::d_positions,
                                       sizeof(void *), 0,
                                       cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc(&visual::d_velocities, h_max_visual * sizeof(glm::vec3)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(visual::velocities, &visual::d_velocities,
                                       sizeof(void *), 0,
                                       cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc(&visual::d_life, h_max_visual * sizeof(float)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(visual::life, &visual::d_life,
                                       sizeof(void *), 0,
                                       cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc(&visual::d_visual_types, h_max_visual * sizeof(int)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(visual::visual_types, &visual::d_visual_types,
                                       sizeof(void *), 0,
                                       cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc(&visual::d_rand_seeds, h_max_visual * sizeof(unsigned int)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(visual::rand_seeds, &visual::d_rand_seeds,
                                       sizeof(void *), 0,
                                       cudaMemcpyHostToDevice));
  _init_rand_seeds<<<(h_max_visual + N_THREADS - 1) / N_THREADS, N_THREADS>>>(h_max_visual);
}

__global__ void _update_debug_points(float* vbo, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    vbo[6 * i    ] = positions[i].x;
    vbo[6 * i + 1] = positions[i].y;
    vbo[6 * i + 2] = positions[i].z;

    if (visual_types[i] == 0) {
      vbo[6 * i + 3] = 1.f;
      vbo[6 * i + 4] = 1.f; 
      vbo[6 * i + 5] = 0.f;
    } else if (visual_types[i] == 1) {
      vbo[6 * i + 3] = 0.f;
      vbo[6 * i + 4] = 0.f; 
      vbo[6 * i + 5] = 1.f;
    } else if (visual_types[i] == 2) {
      vbo[6 * i + 3] = 1.f;
      vbo[6 * i + 4] = 0.f; 
      vbo[6 * i + 5] = 1.f;
    }
  }
}
void update_debug_points(float* vbo) {
  if (h_visual_count <= 0) return;
  _update_debug_points<<<(h_visual_count + N_THREADS - 1) / N_THREADS, N_THREADS>>>(vbo, h_visual_count);
}


__global__ void _update_visual_faces(float *vbo, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    float r = params.particle_size / 10;
    glm::vec3 ns[4] = {
      glm::normalize(glm::vec3(-1,1,-1)),
      glm::normalize(glm::vec3(-1,-1,1)),
      glm::normalize(glm::vec3(1,-1,-1)),
      glm::normalize(glm::vec3(1,1,1))
    };
    glm::vec3 vs[4] = {
      positions[i] + ns[1] * r,
      positions[i] + ns[2] * r,
      positions[i] + ns[3] * r,
      positions[i] + ns[4] * r
    };
    int arr[] = {0,2,1, 0,1,3, 0,3,2, 1,2,3};
    int j = 72 * i;
    for (int idx = 0; idx < 12; ++idx) {
      vbo[j++] = vs[arr[idx]].x;
      vbo[j++] = vs[arr[idx]].y;
      vbo[j++] = vs[arr[idx]].z;

      vbo[j++] = ns[arr[idx]].x;
      vbo[j++] = ns[arr[idx]].y;
      vbo[j++] = ns[arr[idx]].z;
    }
  }
}

void update_visual_faces(float* vbo) {
  if (h_visual_count <= 0) return;
  _update_visual_faces<<<(h_visual_count + N_THREADS - 1) / N_THREADS, N_THREADS>>>(vbo, h_visual_count);
}

}

namespace mc {
__constant__ glm::vec3 mc_corner;
glm::vec3 h_mc_corner;

__constant__ glm::vec3 mc_size;
glm::vec3 h_mc_size;

__constant__ glm::ivec3 grid_size;
glm::ivec3 h_grid_size;
__constant__ glm::ivec3 corner_size;
glm::ivec3 h_corner_size;
__constant__ float cell_size;
float h_cell_size;

__device__ int *grid_occupied;
int* d_grid_occupied;
__device__ float *corner_value;
float* d_corner_value;

__device__ glm::vec3 *faces;
glm::vec3 *d_faces;
__device__ glm::vec3 *face_normals;
glm::vec3 *d_face_normals;
__device__ int *num_faces;
int *d_num_faces;
__device__ int *grid_face_idx;  // used for stream compaction
int *d_grid_face_idx;


__device__ int get_corner_idx(glm::ivec3 v) {
  // TODO: get rid of this check?
  if (v.x < 0 || v.x >= corner_size.x ||
      v.y < 0 || v.y >= corner_size.y || 
      v.z < 0 || v.z >= corner_size.z) {
    return -1;
  }
  return v.x * corner_size.y * corner_size.z + v.y * corner_size.z + v.z;
}
__device__ int get_cell_idx(glm::ivec3 v) {
  return v.x * grid_size.y * grid_size.z + v.y * grid_size.z + v.z;
}
__device__ float get_corner_value(glm::ivec3 v) {
  int i = get_corner_idx(v);
  return i < 0 ? 0.f : corner_value[i];
}

__global__ void _update_grid_corners(int n, float radius) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    float r2 = radius * radius;
    glm::vec3 relative_position = (positions[i] - mc_corner);
    glm::vec3 start = relative_position - glm::vec3(radius);
    glm::vec3 end = relative_position + glm::vec3(radius);

    glm::ivec3 grid_start = glm::clamp(glm::ivec3(glm::ceil(start / cell_size)), glm::ivec3(0), corner_size);
    glm::ivec3 grid_end = glm::clamp(glm::ivec3(glm::ceil(end / cell_size)), glm::ivec3(0), corner_size);

    // TODO: look up table may be better
    for (int x = grid_start.x; x < grid_end.x; ++x) {
      for (int y = grid_start.y; y < grid_end.y; ++y) {
        for (int z = grid_start.z; z < grid_end.z; ++z) {
          glm::vec3 d = glm::vec3(x, y, z) * cell_size - relative_position;
          float d2 = glm::dot(d, d);
          if (d2 < r2) {
            int corner_idx = get_corner_idx(glm::ivec3(x,y,z));

            atomicAdd(&corner_value[corner_idx], kernel(sqrtf(d2) / params.particle_size));

            // TODO: the following may not be needed
            for (int x2 = -1; x2 <= 0; ++x2 ) {
              for (int y2 = -1; y2 <= 0; ++y2) {
                for (int z2 = -1; z2 <= 0; ++z2) {
                  glm::ivec3 cell_pos = glm::ivec3(x+x2, y+y2, z+z2);
                  if (cell_pos.x > 0 && cell_pos.x < grid_size.x &&
                      cell_pos.y > 0 && cell_pos.y < grid_size.y &&
                      cell_pos.z > 0 && cell_pos.z < grid_size.z) {
                    int cell_idx = get_cell_idx(cell_pos);
                    grid_occupied[cell_idx] = 1;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

__device__ glm::vec3 get_vert_normal(glm::ivec3 v) {
  return -glm::normalize(glm::vec3(
      get_corner_value(v + glm::ivec3(1, 0, 0)) - get_corner_value(v + glm::ivec3(-1, 0, 0)),
      get_corner_value(v + glm::ivec3(0, 1, 0)) - get_corner_value(v + glm::ivec3(0, -1, 0)),
      get_corner_value(v + glm::ivec3(0, 0, 1)) - get_corner_value(v + glm::ivec3(0, 0, -1))));
}

__device__ glm::vec3 vertex_interp(float isolevel, glm::vec3 p0, glm::vec3 p1, float v0, float v1) {
  return p0 + (isolevel - v0) / (v1 - v0) * (p1 - p0);
}
__device__ glm::vec3 normal_interp(float isolevel, glm::vec3 n0, glm::vec3 n1, float v0, float v1) {
  return glm::normalize(n0 + (isolevel - v0) / (v1 - v0) * (n1 - n0));
}

__device__ int generate_face(int i, float isolevel) {
  int cube_idx = 0;
  int idx = i;
  int x = idx / (grid_size.y * grid_size.z);
  idx %= grid_size.y * grid_size.z;
  int y = idx / grid_size.z;
  int z = idx % grid_size.z;

  glm::vec3 vertlist[12];
  glm::vec3 normlist[12];
  float val[8];
  val[0] = get_corner_value(glm::ivec3 (x,     y,     z     ) );
  val[1] = get_corner_value(glm::ivec3 (x + 1, y,     z     ) );
  val[2] = get_corner_value(glm::ivec3 (x + 1, y,     z + 1 ) );
  val[3] = get_corner_value(glm::ivec3 (x,     y,     z + 1 ) );
  val[4] = get_corner_value(glm::ivec3 (x,     y + 1, z     ) );
  val[5] = get_corner_value(glm::ivec3 (x + 1, y + 1, z     ) );
  val[6] = get_corner_value(glm::ivec3 (x + 1, y + 1, z + 1 ) );
  val[7] = get_corner_value(glm::ivec3 (x,     y + 1, z + 1 ) );

  glm::vec3 p[8];
  glm::vec3 pn[8];
  p[0] = glm::vec3(x,   y,   z   ) * cell_size + mc_corner;
  p[1] = glm::vec3(x+1, y,   z   ) * cell_size + mc_corner;
  p[2] = glm::vec3(x+1, y,   z+1 ) * cell_size + mc_corner;
  p[3] = glm::vec3(x,   y,   z+1 ) * cell_size + mc_corner;
  p[4] = glm::vec3(x,   y+1, z   ) * cell_size + mc_corner;
  p[5] = glm::vec3(x+1, y+1, z   ) * cell_size + mc_corner;
  p[6] = glm::vec3(x+1, y+1, z+1 ) * cell_size + mc_corner;
  p[7] = glm::vec3(x,   y+1, z+1 ) * cell_size + mc_corner;

  pn[0] = get_vert_normal(glm::vec3(x,   y,   z   ));
  pn[1] = get_vert_normal(glm::vec3(x+1, y,   z   ));
  pn[2] = get_vert_normal(glm::vec3(x+1, y,   z+1 ));
  pn[3] = get_vert_normal(glm::vec3(x,   y,   z+1 ));
  pn[4] = get_vert_normal(glm::vec3(x,   y+1, z   ));
  pn[5] = get_vert_normal(glm::vec3(x+1, y+1, z   ));
  pn[6] = get_vert_normal(glm::vec3(x+1, y+1, z+1 ));
  pn[7] = get_vert_normal(glm::vec3(x,   y+1, z+1 ));

  if (val[0] > isolevel) cube_idx |= 1;
  if (val[1] > isolevel) cube_idx |= 2;
  if (val[2] > isolevel) cube_idx |= 4;
  if (val[3] > isolevel) cube_idx |= 8;
  if (val[4] > isolevel) cube_idx |= 16;
  if (val[5] > isolevel) cube_idx |= 32;
  if (val[6] > isolevel) cube_idx |= 64;
  if (val[7] > isolevel) cube_idx |= 128;

  if (edgeTable[cube_idx] == 0)
    return 0;
  if (edgeTable[cube_idx] & 1) {
    vertlist[0] = vertex_interp(isolevel,  p[0], p[1], val[0], val[1]);
    normlist[0] = normal_interp(isolevel,  pn[0], pn[1], val[0], val[1]);
  }
  if (edgeTable[cube_idx] & 2) {
    vertlist[ 1] = vertex_interp(isolevel, p[1], p[2], val[1], val[2]);
    normlist[ 1] = normal_interp(isolevel, pn[1], pn[2], val[1], val[2]);
  }
  if (edgeTable[cube_idx] & 4) {
    vertlist[ 2] = vertex_interp(isolevel, p[2], p[3], val[2], val[3]);
    normlist[ 2] = normal_interp(isolevel, pn[2], pn[3], val[2], val[3]);
  }
  if (edgeTable[cube_idx] & 8) {
    vertlist [3] = vertex_interp(isolevel, p[3], p[0], val[3], val[0]);
    normlist [3] = normal_interp(isolevel, pn[3], pn[0], val[3], val[0]);
  }
  if (edgeTable[cube_idx] & 16) {
    vertlist[ 4] = vertex_interp(isolevel, p[4], p[5], val[4], val[5]);
    normlist[ 4] = normal_interp(isolevel, pn[4], pn[5], val[4], val[5]);
  }
  if (edgeTable[cube_idx] & 32) {
    vertlist[ 5] = vertex_interp(isolevel, p[5], p[6], val[5], val[6]);
    normlist[ 5] = normal_interp(isolevel, pn[5], pn[6], val[5], val[6]);
  }
  if (edgeTable[cube_idx] & 64) {
    vertlist[ 6] = vertex_interp(isolevel, p[6], p[7], val[6], val[7]);
    normlist[ 6] = normal_interp(isolevel, pn[6], pn[7], val[6], val[7]);
  }
  if (edgeTable[cube_idx] & 128) {
    vertlist[ 7] = vertex_interp(isolevel, p[7], p[4], val[7], val[4]);
    normlist[ 7] = normal_interp(isolevel, pn[7], pn[4], val[7], val[4]);
  }
  if (edgeTable[cube_idx] & 256) {
    vertlist[ 8] = vertex_interp(isolevel, p[0], p[4], val[0], val[4]);
    normlist[ 8] = normal_interp(isolevel, pn[0], pn[4], val[0], val[4]);
  }
  if (edgeTable[cube_idx] & 512) {
    vertlist[ 9] = vertex_interp(isolevel, p[1], p[5], val[1], val[5]);
    normlist[ 9] = normal_interp(isolevel, pn[1], pn[5], val[1], val[5]);
  }
  if (edgeTable[cube_idx] & 1024) {
    vertlist[10] = vertex_interp(isolevel, p[2], p[6], val[2], val[6]);
    normlist[10] = normal_interp(isolevel, pn[2], pn[6], val[2], val[6]);
  }
  if (edgeTable[cube_idx] & 2048) {
    vertlist[11] = vertex_interp(isolevel, p[3], p[7], val[3], val[7]);
    normlist[11] = normal_interp(isolevel, pn[3], pn[7], val[3], val[7]);
  }

  int vi;
  for (vi = 0; triTable[cube_idx][vi] != -1; vi += 1) {
    faces[15 * i + vi] = vertlist[triTable[cube_idx][vi]];
    face_normals[15 * i + vi] = normlist[triTable[cube_idx][vi]];
  }
  return vi / 3;
}

__global__ void _update_faces(int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    num_faces[i] = generate_face(i, 0.2f);
  }
}

__device__ int total_num_faces;
__global__ void _transfer_faces_to_vbo(int n, float* vbo, int max_num_faces) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int j = 0;
    if (grid_face_idx[i] + num_faces[i] >= max_num_faces) {
      if (grid_face_idx[i] < max_num_faces) {
        // printf("Warning too many faces!\n");
        total_num_faces = grid_face_idx[i];
      }
      return;
    }
    for (int t = 0; t < num_faces[i]; ++t) {
      glm::vec3 v1 = faces[15 * i + 3 * t    ];
      glm::vec3 v2 = faces[15 * i + 3 * t + 1];
      glm::vec3 v3 = faces[15 * i + 3 * t + 2];
      glm::vec3 vn1 = face_normals[15 * i + 3 * t    ];
      glm::vec3 vn2 = face_normals[15 * i + 3 * t + 1];
      glm::vec3 vn3 = face_normals[15 * i + 3 * t + 2];

      // vertex
      vbo[18 * grid_face_idx[i] + j++] = v1.x;
      vbo[18 * grid_face_idx[i] + j++] = v1.y;
      vbo[18 * grid_face_idx[i] + j++] = v1.z;

      // normal
      vbo[18 * grid_face_idx[i] + j++] = vn1.x;
      vbo[18 * grid_face_idx[i] + j++] = vn1.y;
      vbo[18 * grid_face_idx[i] + j++] = vn1.z;

      // vertex
      vbo[18 * grid_face_idx[i] + j++] = v2.x;
      vbo[18 * grid_face_idx[i] + j++] = v2.y;
      vbo[18 * grid_face_idx[i] + j++] = v2.z;

      // normal
      vbo[18 * grid_face_idx[i] + j++] = vn2.x;
      vbo[18 * grid_face_idx[i] + j++] = vn2.y;
      vbo[18 * grid_face_idx[i] + j++] = vn2.z;

      // vertex
      vbo[18 * grid_face_idx[i] + j++] = v3.x;
      vbo[18 * grid_face_idx[i] + j++] = v3.y;
      vbo[18 * grid_face_idx[i] + j++] = v3.z;

      // normal
      vbo[18 * grid_face_idx[i] + j++] = vn3.x;
      vbo[18 * grid_face_idx[i] + j++] = vn3.y;
      vbo[18 * grid_face_idx[i] + j++] = vn3.z;
    }
  }
  if (i == n - 1) {
    total_num_faces = grid_face_idx[i] + num_faces[i];
  }
}

void update_faces(float* vbo, int* h_total_num_faces, int max_num_faces) {
  int total_num_cells = h_grid_size.x * h_grid_size.y * h_grid_size.z;
  _update_faces<<<(total_num_cells + 512 - 1) / 512, 512>>>(total_num_cells);

  // stream compaction
  thrust::device_ptr<int> num_faces_ptr = thrust::device_pointer_cast(d_num_faces);
  thrust::device_ptr<int> grid_face_idx_ptr = thrust::device_pointer_cast(d_grid_face_idx);
  thrust::exclusive_scan(num_faces_ptr, num_faces_ptr + total_num_cells, grid_face_idx_ptr);

  _transfer_faces_to_vbo<<<(total_num_cells + N_THREADS - 1) / N_THREADS, N_THREADS>>>(total_num_cells, vbo, max_num_faces);
  CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(h_total_num_faces, mc::total_num_faces, sizeof(int)));
}

void update_grid_corners() {
  cudaMemset(d_grid_occupied, 0, h_grid_size.x * h_grid_size.y * h_grid_size.z * sizeof(int));
  cudaMemset(d_corner_value, 0, h_corner_size.x * h_corner_size.y * h_corner_size.z * sizeof(float));
  _update_grid_corners<<<(num_particles + N_THREADS - 1) / N_THREADS, N_THREADS>>>(num_particles, h_params.cell_size);
}


void init(float cell_size) {
  h_cell_size = cell_size;
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(mc::cell_size, &h_cell_size, sizeof(float)));

  h_mc_corner = h_domain.corner - 4 * cell_size;
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(mc::mc_corner, &h_mc_corner, sizeof(glm::vec3)));
  h_mc_size = h_domain.size + 8 * cell_size;
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(mc::mc_size, &h_mc_size, sizeof(glm::vec3)))

  h_grid_size = glm::ivec3(h_mc_size / cell_size) + glm::ivec3(1, 1, 1);

  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(mc::grid_size, &h_grid_size, sizeof(glm::ivec3)));

  h_corner_size = h_grid_size + glm::ivec3(1, 1, 1);
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(mc::corner_size, &h_corner_size, sizeof(glm::ivec3)));

  // mc grid for corners and cells
  int total_grid_size = h_grid_size.x * h_grid_size.y * h_grid_size.z;
  CUDA_CHECK_RETURN(cudaMalloc(&d_grid_occupied, total_grid_size * sizeof(int)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(mc::grid_occupied, &d_grid_occupied, sizeof(void *), 0,
                                       cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc(&d_corner_value, h_corner_size.x * h_corner_size.y * h_corner_size.z * sizeof(float)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(mc::corner_value, &d_corner_value, sizeof(void *), 0,
                                       cudaMemcpyHostToDevice));

  // initialize face storage
  CUDA_CHECK_RETURN(cudaMalloc(&d_faces, total_grid_size * 15 * sizeof(glm::vec3)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(mc::faces, &d_faces, sizeof(void *), 0, cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc(&d_face_normals, total_grid_size * 15 * sizeof(glm::vec3)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(mc::face_normals, &d_face_normals, sizeof(void *), 0, cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc(&d_num_faces, total_grid_size * sizeof(int)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(mc::num_faces, &d_num_faces, sizeof(void *), 0, cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc(&d_grid_face_idx, total_grid_size * sizeof(int)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(mc::grid_face_idx, &d_grid_face_idx, sizeof(void *), 0, cudaMemcpyHostToDevice));
}

/*
 * idx: idx in grid_array
 * Return: center of grid
 */
glm::vec3 get_grid_center(int idx) {
  int x = idx / (h_grid_size.y * h_grid_size.z);
  idx %= h_grid_size.y * h_grid_size.z;
  int y = idx / h_grid_size.z;
  int z = idx % h_grid_size.z;
  return h_mc_corner + mc::h_cell_size * glm::vec3(x,y,z) + mc::h_cell_size / 2;
}

void print_summary() {
  log("Marching Cube");
  log("Grid size", h_grid_size);
  log("Corner size", h_corner_size);
}

int get_num_cells() {
  return h_grid_size.x * h_grid_size.y * h_grid_size.z;
}

} // namespace mc
} // namespace sph
