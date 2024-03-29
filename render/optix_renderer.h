#pragma once
#include "scene.h"
#include "shader.h"
#include <GL/glew.h>
#include <cstdint>
#include <map>
#include <optixu/optixpp.h>
#include <optixu/optixu_math_namespace.h>
#include <stdint.h>
#include <iostream>

class OptixRenderer {
 public:
  OptixRenderer(uint32_t w, uint32_t h);
  void init();
  void exit();

  void invalidateCamera() { iterations = 0; };
  uint32_t max_iterations = 64;

private:
  std::map<std::shared_ptr<Object>, optix::Transform> _object_transform;
  std::map<std::shared_ptr<TriangleMesh>, optix::Geometry> _mesh_geometry;
  std::map<std::shared_ptr<DynamicMesh>, optix::Geometry> _dmesh_geometry;
  std::map<std::shared_ptr<Object>, optix::Acceleration> _object_accel;
  std::map<std::shared_ptr<Texture>, optix::TextureSampler> _texture_sampler;
  optix::TextureSampler _empty_sampler = 0;

  optix::Program _dmesh_intersect = 0;
  optix::Program _dmesh_bounds = 0;

  optix::Program _mesh_intersect = 0;
  optix::Program _mesh_bounds = 0;
  optix::Program _material_closest_hit = 0;
  optix::Program _material_any_hit = 0;
  optix::Program _material_shadow_any_hit = 0;
  optix::Program _material_fluid_closest_hit = 0;
  optix::Program _material_fluid_any_hit = 0;
  optix::Program _material_fluid_shadow_any_hit = 0;
  optix::Program _material_mirror_closest_hit = 0;
  optix::Program _material_mirror_any_hit = 0;
  optix::Program _material_mirror_shadow_any_hit = 0;
  optix::Program _material_foam_closest_hit = 0;
  optix::Program _material_foam_any_hit = 0;
  optix::Program _material_foam_shadow_any_hit = 0;

  optix::Transform getObjectTransform(std::shared_ptr<Object> obj);
  optix::Geometry getMeshGeometry(std::shared_ptr<TriangleMesh> mesh);
  optix::Geometry getMeshGeometry(std::shared_ptr<DynamicMesh> mesh);
  optix::Acceleration getObjectAccel(std::shared_ptr<Object> obj);
  optix::TextureSampler getTextureSampler(std::shared_ptr<Texture> tex);
  optix::TextureSampler getEmptySampler();

  uint32_t iterations = 0;

 private:
  bool sceneInitialized = false;
  void initSceneGeometry(std::shared_ptr<Scene> scene);
  void initSceneLights(std::shared_ptr<Scene> scene);

 public:
  uint32_t numRays = 1;
  bool useShadow = 1;

 public:
  inline uint32_t getWidth() const { return width; }
  inline uint32_t getHeight() const { return height; }
  inline void resize(int width, int height) {
    std::cerr << "resize not implemented" << std::endl;
  }

 private:
  uint32_t width, height;
  bool initialized = false;

  GLuint outputTex;

  optix::Context context = 0;
  GLuint outputVBO = 0;

  uint32_t nSamplesSqrt = 1;

 public:
  void renderScene(std::shared_ptr<Scene> scene);
  void renderSceneToFile(std::shared_ptr<Scene> scene, std::string filename);
  void renderCurrentToFile(std::string filename);
};
