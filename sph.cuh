#ifndef _SPH_CUH
#define _SPH_CUH

#include <cuda.h>
#define GLM_FORCE_CUDA
#include "sph_common.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <thrust/device_vector.h>

namespace sph {

/* initialize global memory */
void init(const SolverParams &params, const FluidDomain &domain,
          const int max_num_particles);
bool add_particles(glm::vec3 *positions, int size);
bool get_particles(glm::vec3 *positions, int size);
int get_num_particles();
void print_summary();

void update_neighbors();
float step_regular();

void update_density();
void update_factor();

void debug_count_neighbors(int* h_out, int size);
}

#endif // _SPH_CUH
