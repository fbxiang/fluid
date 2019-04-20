#pragma once
#include <cmath>
#include <iostream>
#include <limits.h>
#include <random>
#include <stdint.h>
#include "particle.h"
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

bool less_msb(uint32_t n1, uint32_t n2) {
  return n1 < n2 && n1 < (n1 ^ n2);
}

bool less_zorder(glm::uvec3 p1, glm::uvec3 p2) {
  uint msd_p1 = p1.x; uint msd_p2 = p2.x;
  if (less_msb(msd_p1 ^ msd_p2, p1.y ^ p2.y)) {
    msd_p1 = p1.y; msd_p2 = p2.y;
  }
  if (less_msb(msd_p1 ^ msd_p2, p1.z ^ p2.z)) {
    msd_p1 = p1.z; msd_p2 = p2.z;
  }
  return msd_p1 < msd_p2;
}

// Code from Jeroen Baert
inline uint64_t splitBy3(unsigned int a){
  uint64_t x = a & 0x1fffff; // we only look at the first 21 bits
  x = (x | x << 32) & 0x1f00000000ffff; // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
  x = (x | x << 16) & 0x1f0000ff0000ff; // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
  x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
  x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
  x = (x | x << 2) & 0x1249249249249249;
  return x;
}

inline uint64_t mortonEncode_magicbits(unsigned int x, unsigned int y, unsigned int z){
  uint64_t answer = 0;
  answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
  return answer;
}

uint64_t zorder2number(glm::uvec3 n) {
  return mortonEncode_magicbits(n.x, n.y, n.z);
}

// debug only
glm::uvec3 number2zorder(uint32_t n) {
  if (n == 0) return { 0, 0, 0 };
  glm::uvec3 m = number2zorder(n >> 3);
  m.x <<= 1; m.y <<= 1; m.z <<= 1;
  m.x += n & 1;
  m.y += (n >> 1) & 1;
  m.z += (n >> 2) & 1;
  return m;
}

std::ostream &operator<<(std::ostream &out, glm::uvec2 v) {
  out << "[" << v.x << ", " << v.y << "]";
  return out;
}

std::ostream& operator<<(std::ostream& out, glm::uvec3 v) {
  out << "[" << v.x << ", " << v.y << ", " << v.z << "]";
  return out;
}

std::ostream &operator<<(std::ostream &out, glm::vec3 v) {
  out << "[" << v.x << ", " << v.y << ", " << v.z << "]";
  return out;
}

float rand_float(float start, float range) {
  float random = ((float)rand()) / (float)RAND_MAX; 
  float r = random * range;
  return start + r;
}


std::ostream &operator<<(std::ostream &out, const std::vector<uint32_t> &ps) {
  out << "[ ";
  for (auto i : ps) {
    out << i << " ";
  }
  out << "]";
  return out;
}
