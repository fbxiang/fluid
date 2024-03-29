#pragma once
#include "object.h"
#include <vector>
#include "camera.h"
#include "lights.h"

class Scene : public std::enable_shared_from_this<Scene> {
 public:
  Scene() {};
  ~Scene() {};

 private:
  std::vector<std::shared_ptr<Object> > objects;
  std::shared_ptr<Camera> mainCamera;

  std::vector<PointLight> pointLights;
  std::vector<DirectionalLight> directionalLights;
  std::vector<ParallelogramLight> parallelogramLights;
  std::shared_ptr<CubeMapTexture> environmentMap;

 public:
  bool contains(const std::shared_ptr<Object> obj) const;
  void addObject(std::shared_ptr<Object> obj);
  void removeObject(std::shared_ptr<Object> obj);
  void removeObjectsByName(std::string name);
  const std::vector<std::shared_ptr<Object>>& getObjects() const;

  void setMainCamera(const std::shared_ptr<Camera> cam);
  std::shared_ptr<Camera> getMainCamera() const;

  void addPointLight(PointLight light);
  void addDirectionalLight(DirectionalLight light);
  void addParalleloGramLight(ParallelogramLight light);

  void setEnvironmentMap(const std::string &front, const std::string &back,
                         const std::string &top, const std::string &bottom,
                         const std::string &left, const std::string &right,
                         int wrapping=GL_CLAMP_TO_EDGE, int filtering=GL_LINEAR);

  inline const std::shared_ptr<CubeMapTexture> &getEnvironmentMap() const { return environmentMap; }

  const std::vector<PointLight> &getPointLights() const { return pointLights; }
  const std::vector<DirectionalLight>& getDirectionalLights() const { return directionalLights; }
  const std::vector<ParallelogramLight>& getParallelogramLights() const { return parallelogramLights; }
};

