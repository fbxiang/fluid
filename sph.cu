#include "sph.cuh"
#include "sph_math.h"
#include <thrust/scan.h>
#include <thrust/reduce.h>


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

__device__ float *rho_pred;
float *d_rho_pred;

__device__ float *alpha;
float *d_alpha;

__device__ float *kappa;
float *d_kappa;

__device__ float *kappav;
float *d_kappav;

__device__ float* tmp_float;
float* d_tmp_float;

// only for regular solver
__device__ float* pressure;
float* d_pressure;

__device__ float dt;

int max_num_particles;
int num_particles;
int num_cells;

// // Code from Jeroen Baert
// __device__ uint64_t splitBy3(unsigned int a) {
//   uint64_t x = a & 0x1fffff;
//   x = (x | x << 32) & 0x1f00000000ffff;
//   x = (x | x << 16) & 0x1f0000ff0000ff;
//   x = (x | x << 8) & 0x100f00f00f00f00f;
//   x = (x | x << 4) & 0x10c30c30c30c30c3;
//   x = (x | x << 2) & 0x1249249249249249;
//   return x;
// }

// __device__ uint64_t mortonEncode_magicbits(uint32_t x, uint32_t y, uint32_t
// z) {
//   uint64_t answer = 0;
//   answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
//   return answer;
// }

// __device__ uint64_t zorder2number(glm::uvec3 n) {
//   return mortonEncode_magicbits(n.x, n.y, n.z);
// }

// __device__ __forceinline__ glm::uvec3 space2grid(glm::vec3 point) {
//   return (point - domain.corner) / params.particle_size;
// }

// __device__ __forceinline__ uint32_t space2idx(glm::vec3 point) {
//   return zorder2number(space2grid(point));
// }

// /* store grid number for ith particle
//  * cell_idx[i]: cell number for particle i
//  * bin_counts[n]: number of particles in cell n
//  */
// __global__ void update_cell_idx(uint32_t *cell_idx, uint32_t *bin_counts,
//                                 glm::vec3 *positions, size_t n) {
//   uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
//   if (i < n) {
//     uint32_t grid = space2idx(positions[i]);
//     cell_idx[i] = grid;
//     atomicAdd(&bin_counts[grid], 1);
//   }
// }

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

__global__ void _velocity_to_speed(float* d_speed, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    d_speed[i] = glm::length(velocities[i]);
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

// all solvers
float update_dt_by_CFL() {
  _velocity_to_speed<<<(num_particles + N_THREADS - 1) / N_THREADS, N_THREADS>>>(d_tmp_float, num_particles);
  thrust::device_ptr<float> speed_ptr = thrust::device_pointer_cast(d_tmp_float);
  float max_speed = *thrust::max_element(speed_ptr, speed_ptr + num_particles);
  float dt = 0.2 * h_params.particle_size / max_speed;
  if (dt > 0.01) dt = 0.003;
  cudaMemcpyToSymbol(sph::dt, &dt, sizeof(float));
  return dt;
}

// all solvers
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
void update_density() {
  _update_density<<<(num_particles + N_THREADS - 1) / N_THREADS, N_THREADS>>>(num_particles);
}


// TODO: Check implementation
// DF solver
__global__ void _update_factor(int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    glm::vec3 vacc = {0, 0, 0};
    float sacc = 0;
    FOR_NEIGHBORS(
      if (i == j) continue;
      glm::vec3 dw = dw_ij(positions[i], positions[j], params.particle_size);
      glm::vec3 s = params.particle_mass * dw;
      vacc += s;
      sacc += glm::dot(s, s);
                 );
    alpha[i] = rho[i] / (glm::dot(vacc, vacc) + sacc);
  }
}
void update_factor() {
  _update_factor<<<(num_particles + N_THREADS - 1) / N_THREADS, N_THREADS>>>(num_particles);
}


// DF solver
__global__ void _update_non_pressure_forces(int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    force[i] = params.particle_mass * params.g;
  }
}
void update_non_pressure_forces(int n) {
  _update_non_pressure_forces<<<(num_particles + N_THREADS - 1) / N_THREADS, N_THREADS>>>(n);
}


__global__ void _update_pressure(int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    pressure[i] = params.k * params.rest_density / params.gamma *
                  (powf(rho[i] / params.rest_density, params.gamma) - 1.f);
  }
}
// regular solver
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
void update_velocity_position() {
  _update_velocity_position<<<(num_particles + N_THREADS - 1) / N_THREADS, N_THREADS>>>(num_particles);
}

float step_regular() {
  update_neighbors();
  float dt = update_dt_by_CFL();
  update_density();
  update_all_forces();
  update_velocity_position();
  return dt;
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

  CUDA_CHECK_RETURN(cudaMalloc(&d_grid, num_cells * sizeof(int)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(sph::grid, &d_grid, sizeof(void *), 0,
                                       cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc(&d_grid_start_idx, num_cells * sizeof(int)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(sph::grid_start_idx, &d_grid_start_idx,
                                       sizeof(void *), 0,
                                       cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(
      cudaMalloc(&d_positions, max_num_particles * sizeof(glm::vec3)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(sph::positions, &d_positions, sizeof(void *),
                                       0, cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(
      cudaMalloc(&d_velocities, max_num_particles * sizeof(glm::vec3)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(
      sph::velocities, &d_velocities, sizeof(void *), 0, cudaMemcpyHostToDevice));


  CUDA_CHECK_RETURN(
      cudaMalloc(&d_force, max_num_particles * sizeof(glm::vec3)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(
      sph::force, &d_force, sizeof(void *), 0, cudaMemcpyHostToDevice));

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

  CUDA_CHECK_RETURN(cudaMalloc(&d_rho_pred, max_num_particles * sizeof(float)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(sph::rho_pred, &d_rho_pred, sizeof(void *),
                                       0, cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc(&d_alpha, max_num_particles * sizeof(float)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(sph::alpha, &d_alpha, sizeof(void *), 0,
                                       cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc(&d_kappa, max_num_particles * sizeof(float)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(sph::kappa, &d_kappa, sizeof(void *), 0,
                                       cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc(&d_kappav, max_num_particles * sizeof(float)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(sph::kappav, &d_kappav, sizeof(void *), 0,
                                       cudaMemcpyHostToDevice));

  CUDA_CHECK_RETURN(cudaMalloc(&d_tmp_float, max_num_particles * sizeof(float)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(sph::tmp_float, &d_tmp_float, sizeof(void *), 0,
                                       cudaMemcpyHostToDevice));
  
  CUDA_CHECK_RETURN(cudaMalloc(&d_pressure, max_num_particles * sizeof(float)));
  CUDA_CHECK_RETURN(cudaMemcpyToSymbol(sph::pressure, &d_pressure, sizeof(void *), 0,
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

  // int h_cell_idx[num_particles];
  // cudaMemcpy(h_cell_idx, d_cell_idx, num_particles * sizeof(int), cudaMemcpyDeviceToHost);
  // log_array("cell_idx", h_cell_idx, num_particles);

  // int *h_grid = new int[num_cells];
  // cudaMemcpy(h_grid, d_grid, num_cells * sizeof(int), cudaMemcpyDeviceToHost);
  // log("cell particle count");
  // for (int i = 0; i < num_cells; ++i) {
  //   if (h_grid[i] != 0) {
  //     printf("%d %d\n", i, h_grid[i]);
  //   }
  // }
  // delete [] h_grid;

  // int *h_grid_start_idx = new int[num_cells];
  // cudaMemcpy(h_grid_start_idx, d_grid_start_idx, num_cells * sizeof(int), cudaMemcpyDeviceToHost);
  // log("cell start index");
  // for (int i = 0; i < num_cells; ++i) {
  //   if (i == 0 || h_grid_start_idx[i] != h_grid_start_idx[i-1] )
  //     printf("%d %d\n", i, h_grid_start_idx[i]);
  // }
  // delete [] h_grid_start_idx;

  // int h_sorted_particle_idx[num_particles];
  // cudaMemcpy(h_sorted_particle_idx, d_sorted_particle_idx, num_particles * sizeof(int), cudaMemcpyDeviceToHost);
  // for (int i = 0; i < num_particles; ++i) {
  //   printf("%6d ", h_cell_idx[h_sorted_particle_idx[i]]);
  // }
  // log();
}

__global__ void _debug_count_neighbors(int* d_out, int size) {
  int count = 0;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    FOR_NEIGHBORS(
        count++;
                  );
    d_out[i] = count;
  }
}
void debug_count_neighbors(int* h_out, int size) {
  int* d_out;
  cudaMalloc(&d_out, size * sizeof(int));
  _debug_count_neighbors<<<(num_particles + N_THREADS - 1) / N_THREADS, N_THREADS>>>(d_out, size);
  cudaMemcpy(h_out, d_out, size * sizeof(int), cudaMemcpyDeviceToHost);
}

} // namespace sph

