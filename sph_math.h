#pragma once
#include <cmath>
#include <cuda_runtime.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

__device__ __host__ __forceinline__ float kernel(float q) {
  if (q < 1) {
    return 3.f / (2 * M_PIf32) * (2.f / 3.f - q * q + 0.5 * q * q * q);
  } else if (q < 2) {
    return 3.f / (2 * M_PIf32) * (1.f / 6.f * powf(2.f - q, 3.f));
  }
  return 0.f;
}

__device__ __host__ __forceinline__ float dkernel(float q) {
  if (q < 1) {
    return 3.f / (2.f * M_PIf32) * (-2.f * q + 1.5f * q * q);
  } else if (q < 2) {
    return 3.f / (2.f * M_PIf32) * (-3.f / 6.f * powf(2.f - q, 2.f));
  }
  return 0.f;
}

__device__ __host__ __forceinline__ float w_ij(glm::vec3 pos1, glm::vec3 pos2, float h) {
  float r = glm::length(pos1 - pos2);
  float q = r / h;
  return 1.f / (h * h * h) * kernel(q);
}

__device__ __host__ __forceinline__ glm::vec3 dw_ij(glm::vec3 pos1, glm::vec3 pos2, float h) {
  glm::vec3 d = pos1 - pos2;
  float r = glm::length(d);
  float q = r / h;
  if (r < 1e-10) {
    return glm::vec3(0,0,0);
  }
  return d * (1.f / (h * h * h * h) * dkernel(q) / r);
}
