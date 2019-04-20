#pragma once
#include <GL/glew.h>
#include <memory>
#include <vector>
#include <glm/glm.hpp>
#include "texture.h"

struct Vertex {
  glm::vec3 position;
  glm::vec3 normal;
  glm::vec2 texCoord;
  glm::vec3 tangent;
  glm::vec3 bitangent;

  Vertex(glm::vec3 p=glm::vec3(0), glm::vec3 n=glm::vec3(0), glm::vec2 t=glm::vec2(0),
         glm::vec3 tan=glm::vec3(0), glm::vec3 bitan=glm::vec3(0)):
      position(p), normal(n), texCoord(t), tangent(tan), bitangent(bitan) {}
};


class TriangleMesh : public std::enable_shared_from_this<TriangleMesh> {
 protected:
  GLuint vao;
  GLuint vbo;
  GLuint ebo;

  std::vector<Vertex> vertices;
  std::vector<GLuint> indices;

 public:
  TriangleMesh();
  TriangleMesh(const std::vector<Vertex>& inVertices, const std::vector<GLuint>& inIndices, bool recalcNormal = false);
  ~TriangleMesh();

  uint64_t size() const { return indices.size() / 3; }

  GLuint getVAO() const;
  GLuint getVBO() const;
  GLuint getEBO() const;
  const std::vector<Vertex>& getVertices() const;
  const std::vector<GLuint>& getIndices() const;

 private:
  void recalculateNormals();
};



std::shared_ptr<TriangleMesh> NewCubeMesh();
