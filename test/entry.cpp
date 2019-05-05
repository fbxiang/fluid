#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SPHTest
#include "../sph_cpu.h"
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <random>

using namespace std;

// BOOST_AUTO_TEST_CASE(Scan) {
//   std::mt19937 rng(0);
//   std::uniform_int_distribution<int> gen(0, 10);

//   int N = 512;
//   uint32_t h_array[N];
//   uint32_t array[N];
//   for (int i = 0; i < N; ++i) {
//     array[i] = h_array[i] = gen(rng);
//   }

//   thrust::device_vector<uint32_t> d_array(h_array, h_array + N);
//   thrust::device_vector<uint32_t> d_out(d_array.size());
//   thrust::exclusive_scan(d_array.begin(), d_array.end(), d_out.begin());

//   scan(array, N);

//   for (int i = 0; i < N; ++i) {
//     BOOST_CHECK_EQUAL(array[i], h_array[i]);
//   }
// }

BOOST_AUTO_TEST_CASE(Fluid_Domain) {
  SPH_CPU sph;
  sph.set_domain({0.1, 0.2, 0.3}, {1.05, 2.05, 3.05});
  sph.set_particle_size_and_density(0.05, 1000);
  uint32_t cells = sph.num_cells();
  BOOST_CHECK_EQUAL(cells, 11 * 21 * 31);
}

glm::vec3 position_from_grid(glm::vec3 corner, glm::uvec3 grid_size, float cell_size,
                             uint32_t n) {
  uint32_t x = n / (grid_size.y * grid_size.z);
  n %= grid_size.y * grid_size.z;
  uint32_t y = n / grid_size.z;
  uint32_t z = n % grid_size.z;
  return corner + glm::vec3(x * cell_size + 0.01, y * cell_size + 0.01, z * cell_size + 0.01);
}

BOOST_AUTO_TEST_CASE(Cell_idx) {
  SPH_CPU sph;
  sph.set_domain({0.1, 0.2, 0.3}, {1.05, 2.05, 3.05});
  sph.set_particle_size_and_density(0.05, 1000);
  sph.grid.resize(sph.num_cells());
  sph.clear_grid();

  std::vector<uint32_t> v = {51, 70, 61, 60, 71, 50, 62, 61, 50, 70};

  for (uint32_t i : v) {
    sph.positions.push_back(
        position_from_grid(sph.fluid_domain.corner, sph.grid_size(),
                           sph.solver_params.cell_size, i));
  }
  sph.cell_idx.resize(sph.positions.size());
  sph.sorted_particle_idx.resize(sph.positions.size());
  sph.update_cell_idx();

  for (uint32_t i = 0; i < sph.size(); ++i) {
    BOOST_CHECK_EQUAL(v[i], sph.cell_idx[i]);
  }
  for (uint32_t n = 0; n < sph.num_cells(); ++n) {
    if (sph.grid[n]) {
      printf("%d: %d ", n, sph.grid[n]);
    }
  }
  printf("\n");
  sph.update_sorted_particle_idx();

  for (uint32_t n = 0; n < sph.num_cells(); ++n) {
    if (sph.grid[n]) {
      printf("%d: %d ", n, sph.grid_start_idx[n]);
    }
  }
  printf("\n");

  for (uint32_t i = 0; i < sph.size(); ++i) {
    printf("%d ", sph.sorted_particle_idx[i]);
  }
  printf("\n");
}

inline float rand_float(float start, float range) {
  float random = ((float)rand()) / (float)RAND_MAX;
  float r = random * range;
  return start + r;
}

void fill_block(SPH_CPU &sph, glm::vec3 corner, glm::vec3 size) {
  float step = sph.solver_params.particle_size;
  std::vector<glm::vec3> positions;
  glm::uvec3 n = size / step;
  for (uint32_t i = 0; i < n.x; ++i) {
    for (uint32_t j = 0; j < n.y; ++j) {
      for (uint32_t k = 0; k < n.z; ++k) {
        glm::vec3 jitter = {rand_float(-step / 4.f, step / 4.f),
                            rand_float(-step / 4.f, step / 4.f),
                            rand_float(-step / 4.f, step / 4.f)};
        positions.push_back({{corner + glm::vec3(i, j, k) * step + jitter}});
      }
    }
  }
  sph.init(positions);
}

BOOST_AUTO_TEST_CASE(Neighbor) {
  SPH_CPU sph;
  sph.set_domain({0, 0, 0}, {9.99, 9.99, 9.99});
  sph.set_particle_size_and_density(0.1, 1000);
  fill_block(sph, {0.01, 0.01, 0.01}, {1, 0.8, 1.2});
  sph.sim_init();

  for (uint32_t i = 0; i < sph.size(); ++i) {
    // printf("Particle %d\n", i);
    std::vector<uint32_t> real_neighbors;
    for (uint32_t j = 0; j < sph.size(); ++j) {
      if (glm::length(sph.positions[i] - sph.positions[j]) <
          sph.solver_params.cell_size) {
        real_neighbors.push_back(j);
      }
    }

    std::vector<uint32_t> neighbors;
    for (uint32_t j : sph.find_neighbors(i)) {
      if (glm::length(sph.positions[i] - sph.positions[j]) <
          sph.solver_params.cell_size) {
        neighbors.push_back(j);
      }
    }

    std::sort(real_neighbors.begin(), real_neighbors.end());
    std::sort(neighbors.begin(), neighbors.end());
    BOOST_CHECK_EQUAL(real_neighbors.size(), neighbors.size());
    for (uint32_t i = 0; i < neighbors.size(); ++i) {
      BOOST_CHECK_EQUAL(real_neighbors[i], neighbors[i]);
    }
  }
}
