#include "mesh.h"

TriangleMesh::TriangleMesh() : vao(0), vbo(0), ebo(0) {}

TriangleMesh::~TriangleMesh() {
  if (vbo)
    glDeleteBuffers(1, &vbo);
  if (ebo)
    glDeleteBuffers(1, &ebo);
  if (vao)
    glDeleteVertexArrays(1, &vao);
}

TriangleMesh::TriangleMesh(const std::vector<Vertex> &inVertices,
                           const std::vector<GLuint> &inIndices,
                           bool recalcNormal) {
  vertices = inVertices;
  indices = inIndices;

  if (recalcNormal)
    recalculateNormals();

  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);

  glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0],
               GL_STATIC_DRAW);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *)0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                        (void *)(3 * sizeof(float)));

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                        (void *)(6 * sizeof(float)));

  glEnableVertexAttribArray(3);
  glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                        (void *)(8 * sizeof(float)));

  glEnableVertexAttribArray(4);
  glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                        (void *)(11 * sizeof(float)));

  glGenBuffers(1, &ebo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint),
               &indices[0], GL_STATIC_DRAW);
}


void TriangleMesh::recalculateNormals() {
  for (auto &v : vertices) {
    v.normal = glm::vec3(0);
  }
  for (size_t i = 0; i < indices.size(); i += 3) {
    unsigned int i1 = indices[i];
    unsigned int i2 = indices[i + 1];
    unsigned int i3 = indices[i + 2];
    Vertex &v1 = vertices[i1];
    Vertex &v2 = vertices[i2];
    Vertex &v3 = vertices[i3];

    glm::vec3 normal = glm::normalize(
        glm::cross(v2.position - v1.position, v3.position - v1.position));
    if (std::isnan(normal.x)) {
      continue;
    }
    v1.normal += normal;
    v2.normal += normal;
    v3.normal += normal;
  }

  for (size_t i = 0; i < vertices.size(); i++) {
    if (vertices[i].normal == glm::vec3(0))
      continue;
    vertices[i].normal = glm::normalize(vertices[i].normal);
  }
}

GLuint TriangleMesh::getVAO() const { return vao; }
GLuint TriangleMesh::getVBO() const { return vbo; }
GLuint TriangleMesh::getEBO() const { return ebo; }

const std::vector<Vertex> &TriangleMesh::getVertices() const {
  return vertices;
}

const std::vector<GLuint> &TriangleMesh::getIndices() const { return indices; }

std::shared_ptr<TriangleMesh> NewCubeMesh() {
  std::vector<Vertex> vertices = { Vertex(glm::vec3(-1.0, -1.0, 1.0)), 
                                   Vertex(glm::vec3(1.0,  -1.0, 1.0)),
                                   Vertex(glm::vec3(1.0,  1.0,  1.0)), 
                                   Vertex(glm::vec3(-1.0, 1.0,  1.0)), 
                                   Vertex(glm::vec3(-1.0, -1.0, -1.0)), 
                                   Vertex(glm::vec3(1.0,  -1.0, -1.0)), 
                                   Vertex(glm::vec3(1.0,  1.0,  -1.0)), 
                                   Vertex(glm::vec3(-1.0, 1.0,  -1.0))};
  std::vector<GLuint> indices = {0, 1, 2, 2, 3, 0,
                                 1, 5, 6, 6, 2, 1,
                                 7, 6, 5, 5, 4, 7,
                                 4, 0, 3, 3, 7, 4,
                                 4, 5, 1, 1, 0, 4,
                                 3, 2, 6, 6, 7, 3};

  // std::vector<GLuint> indices = {4, 5, 1, 1, 0, 4,
  //                                3, 2, 6, 6, 7, 3};

  return std::make_shared<TriangleMesh>(vertices, indices, true);
}