#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <memory>

class Texture {
 private:
  GLuint id;

 public:
  Texture(): id(0) {}
  virtual ~Texture() { destroy(); }

  void load(const std::string& filename, int mipmap=0, int wrapping=GL_REPEAT, int minFilter=GL_NEAREST_MIPMAP_LINEAR, int magFilter=GL_LINEAR);
  void destroy();
  GLuint getId() const;

 public:
  static const Texture Empty;

  // TODO: convert texture to Optix texture
};

std::shared_ptr<Texture> LoadTexture(const std::string& filename, int mipmap=0, int wrapping=GL_REPEAT, int minFilter=GL_NEAREST_MIPMAP_LINEAR, int magFilter=GL_LINEAR);

void writeToFile(GLuint textureId, GLuint width, GLuint height, std::string filename);
