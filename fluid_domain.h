#pragma once
#include <cmath>
#include <cstdio>
#include <glm/vec3.hpp>
#include <algorithm>

struct FluidDomain {
  glm::vec3 corner;
  glm::vec3 size;
  float step_size;

  glm::uvec3 grid_size() {
    uint32_t x = ceil(size.x / step_size);
    uint32_t y = ceil(size.y / step_size);
    uint32_t z = ceil(size.z / step_size);
    return { x, y, z };
  }

  uint32_t num_cells() {
    uint32_t x = ceil(size.x / step_size);
    uint32_t y = ceil(size.y / step_size);
    uint32_t z = ceil(size.z / step_size);
    uint32_t m = std::max(std::max(x, y), z);
    m = pow(2, ceilf(log2f(m)));
    return m * m * m;
  }

  glm::uvec3 pos2grid_pos(glm::vec3 pos) {
    return {(pos.x - corner.x) / step_size, (pos.y - corner.y) / step_size,
            (pos.z - corner.z) / step_size};
  }
};
