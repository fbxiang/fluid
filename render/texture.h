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

class CubeMapTexture {
 private:
  GLuint id;
  int width = 0;
  int height = 0;

 public:
  CubeMapTexture(): id(0) {}
  virtual ~CubeMapTexture() { destroy(); }

  void destroy() {
    glDeleteTextures(1, &id);
    id = 0;
  }


  void load(const std::string &front, const std::string &back,
            const std::string &top, const std::string &bottom,
            const std::string &left, const std::string &right, int wrapping,
            int filtering);

  inline GLuint getId() const { return id; } 
  inline int getWidth() const { return width; }
  inline int getHeight() const { return height; }

 private:
  void loadCubeMapSide(GLuint texture, GLenum side, const std::string &filename, bool texStorage=false);
};

std::shared_ptr<CubeMapTexture>
LoadCubeMapTexture(const std::string &front, const std::string &back,
                   const std::string &top, const std::string &bottom,
                   const std::string &left, const std::string &right,
                   int wrapping, int filtering);
