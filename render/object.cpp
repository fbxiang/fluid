#include "object.h"

glm::mat4 Object::getModelMat() const {
  glm::mat4 t = glm::toMat4(rotation);
  t[0] *= scale.x;
  t[1] *= scale.y;
  t[2] *= scale.z;
  t[3][0] = position.x; 
  t[3][1] = position.y; 
  t[3][2] = position.z; 

  return t;
}


void Object::setScene(const std::shared_ptr<Scene> inScene) {
  scene = inScene;
}


std::shared_ptr<Scene> Object::getScene() const {
  if (!scene.expired())
    return scene.lock();
  return std::shared_ptr<Scene>();
}


std::shared_ptr<TriangleMesh> Object::getMesh() const {
  return mesh;
}


std::shared_ptr<Object> NewNoisePlane(unsigned int res) {

  PerlinNoise noise;
  noise.addNoise(0.1, glm::vec2(0), 2, 0);
  noise.addNoise(0.01, glm::vec2(3, 4), 20, 7);
  noise.addNoise(0.05, glm::vec2(1, -1), 4, 2);

  std::vector<Vertex> vertices;
  float d = 1.f / res;
  for (int i = 0; i < res+1; i++) {
    for (int j = 0; j < res+1; j++) {
      float x = -0.5 + i*d;
      float z = -0.5 + j*d;
      vertices.push_back(Vertex(glm::vec3(x, noise(x, z), z),
                                glm::vec3(0),
                                glm::vec2(i*d, j*d)));
    }
  }

  std::vector<unsigned int> indices;
  for (unsigned int i = 0; i < res; i++) {
    for (unsigned int j = 0; j < res; j++) {
      unsigned int v1 = i * (res+1) + j;
      unsigned int v2 = i * (res+1) + j + 1;
      unsigned int v3 = (i+1) * (res+1) + j;
      unsigned int v4 = (i+1) * (res+1) + j + 1;
      indices.push_back(v1);
      indices.push_back(v2);
      indices.push_back(v3);
      indices.push_back(v3);
      indices.push_back(v2);
      indices.push_back(v4);
    }
  }

  auto mesh = std::make_shared<TriangleMesh>(vertices, indices);
  auto obj = NewObject<Object>(mesh);
  return obj;
}

std::shared_ptr<Object> NewDebugObject() {
  std::vector<Vertex> vertices;
  vertices.push_back(Vertex(glm::vec3(-1, 1, 0), glm::vec3(0, 0, 1), glm::vec2(0, 1)));
  vertices.push_back(Vertex(glm::vec3(-1, -1, 0), glm::vec3(0, 0, 1), glm::vec2(0, 0)));
  vertices.push_back(Vertex(glm::vec3(1, -1, 0), glm::vec3(0, 0, 1), glm::vec2(1, 0)));
  vertices.push_back(Vertex(glm::vec3(1, 1, 0), glm::vec3(0, 0, 1), glm::vec2(1, 1)));

  std::vector<uint32_t> indices = { 0, 1, 2, 0, 2, 3 };

  auto obj = NewObject<Object>(std::make_shared<TriangleMesh>(vertices, indices));
  obj->name = "Debug";
  return obj;
}

std::shared_ptr<Object> NewPlane() {
  std::vector<Vertex> vertices;
  vertices.push_back(Vertex(glm::vec3(-1, 1, 0), glm::vec3(0, 0, 1), glm::vec2(0, 1),
                            glm::vec3(1, 0, 0), glm::vec3(0, 1, 0)));
  vertices.push_back(Vertex(glm::vec3(-1, -1, 0), glm::vec3(0, 0, 1), glm::vec2(0, 0),
                            glm::vec3(1, 0, 0), glm::vec3(0, 1, 0)));
  vertices.push_back(Vertex(glm::vec3(1, -1, 0), glm::vec3(0, 0, 1), glm::vec2(1, 0),
                            glm::vec3(1, 0, 0), glm::vec3(0, 1, 0)));
  vertices.push_back(Vertex(glm::vec3(1, 1, 0), glm::vec3(0, 0, 1), glm::vec2(1, 1),
                            glm::vec3(1, 0, 0), glm::vec3(0, 1, 0)));

  std::vector<uint32_t> indices = { 0, 1, 2, 0, 2, 3 };
  auto obj = NewObject<Object>(std::make_shared<TriangleMesh>(vertices, indices));
  obj->name = "Plane";
  return obj;
}

std::shared_ptr<Object> NewCube() {
  auto obj = NewObject<Object>(NewCubeMesh());
  obj->name = "cube";
  return obj;
}
