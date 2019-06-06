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

float pci_step();

void update_density();
void update_factor();

void update_debug_points(float* vbo);

namespace mc {

int get_num_cells();

void init(float cell_size);
glm::vec3 get_grid_center(int idx);
void print_summary();

void update_grid_corners();

void update_faces(float* vbo, int* h_total_num_faces, int max_num_faces);
} // namespace mc
}
