#define BOOST_TEST_MODULE SPHTest
#include "../sph_gpu.h"
#include "../utils.h"
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <random>
#include <vector>

static glm::vec3 position_from_grid(glm::vec3 corner, glm::uvec3 grid_size,
                             float cell_size, uint32_t n) {
  int x = n / (grid_size.y * grid_size.z);
  n %= grid_size.y * grid_size.z;
  int y = n / grid_size.z;
  int z = n % grid_size.z;
  return corner + glm::vec3(x * cell_size + 0.01, y * cell_size + 0.01,
                            z * cell_size + 0.01);
}

std::vector<glm::vec3> fill_block(glm::vec3 corner, glm::vec3 size, float step) {
  std::vector<glm::vec3> positions;
  glm::ivec3 n = size / step;
  for (int i = 0; i < n.x; ++i) {
    for (int j = 0; j < n.y; ++j) {
      for (int k = 0; k < n.z; ++k) {
        glm::vec3 jitter = {rand_float(-step / 4.f, step / 4.f),
                            rand_float(-step / 4.f, step / 4.f),
                            rand_float(-step / 4.f, step / 4.f)};
        positions.push_back({{corner + glm::vec3(i, j, k) * step + jitter}});
      }
    }
  }
  return positions;
}

std::vector<int> count_neighrbos_brute_force(std::vector<glm::vec3> positions, float cell_size) {
  std::vector<int> count(positions.size());
  for (int i = 0; i < positions.size(); ++i) {
    for (int j = 0; j < positions.size(); ++j) {
      if (glm::length(positions[i] - positions[j]) < cell_size) {
        count[i]++;
      }
    }
  }
  return count;
}

BOOST_AUTO_TEST_CASE(NeighborCount) {
  SPH_GPU system(0.05f);
  system.set_domain({0, 0, 0}, {10, 20, 30});
  system.cuda_init();

  std::vector<glm::vec3> positions = fill_block({1,1,1}, {0.4, 0.5, 0.6}, 0.05f);
  std::cout << positions.size() << std::endl;
  system.add_particles(positions);
  system.update_neighbors();

  sph::print_summary();
  std::vector<int> count(positions.size());
  sph::debug_count_neighbors(count.data(), count.size());
  std::vector<int> true_count = count_neighrbos_brute_force(positions, 0.1f);
  for (int i = 0; i < count.size(); ++i) {
    BOOST_CHECK_EQUAL(count[i], true_count[i]);
  }
}


BOOST_AUTO_TEST_CASE(MarchingCube) {
  SPH_GPU system(0.05f);
  system.set_domain({0, 0, 0}, {10, 20, 30});
  system.cuda_init();
  system.marching_cube_init();

  std::vector<glm::vec3> positions =
      fill_block({1, 1, 1}, {0.4, 0.5, 0.6}, 0.05f);
  system.add_particles(positions);
  system.update_neighbors();

  sph::print_summary();
  sph::mc::print_summary();
  sph::mc::update_grid_corners();
  sph::mc::print_summary();
}
