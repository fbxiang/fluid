#include "texture.h"
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const Texture Texture::Empty;

void Texture::load(const std::string &filename, int mipmap, int wrapping,
                   int minFilter, int magFilter) {
  if (id)
    destroy();

  glActiveTexture(GL_TEXTURE0);
  glGenTextures(1, &id);
  glBindTexture(GL_TEXTURE_2D, id);

  int width, height, nrChannels;
  unsigned char *data =
      stbi_load(filename.c_str(), &width, &height, &nrChannels, STBI_rgb_alpha);
  printf("%d channels loaded\n", nrChannels);

  glTexStorage2D(GL_TEXTURE_2D, 4, GL_RGBA8, width, height);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA,
                  GL_UNSIGNED_BYTE, data);
  glGenerateMipmap(GL_TEXTURE_2D);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapping);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapping);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);

  stbi_image_free(data);
}

void Texture::destroy() {
  glDeleteTextures(1, &id);
  id = 0;
}

std::shared_ptr<Texture> LoadTexture(const std::string &filename, int mipmap,
                                     int wrapping, int minFilter,
                                     int maxFilter) {
  auto tex = std::make_shared<Texture>();
  tex->load(filename, mipmap, wrapping, minFilter, maxFilter);
  return tex;
}

GLuint Texture::getId() const { return id; }

void writeToFile(GLuint textureId, GLuint width, GLuint height,
                 std::string filename) {
  uint8_t data[width * height * 4];
  glBindTexture(GL_TEXTURE_2D, textureId);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

  for (uint32_t h1 = 0; h1 < height / 2; ++h1) {
    uint32_t h2 = height - 1 - h1;
    for (uint32_t i = 0; i < 4 * width; ++i) {
      std::swap(data[h1 * width * 4 + i], data[h2 * width * 4 + i]);
    }
  }

  stbi_write_png(filename.c_str(), width, height, 4, data, width * 4);
}

void CubeMapTexture::loadCubeMapSide(GLuint texture,
                                     GLenum side,
                                     const std::string &filename,
                                     bool texStorage) {
  int width, height, nrChannels;
  unsigned char *data =
      stbi_load(filename.c_str(), &width, &height, &nrChannels, 4);

  if (this->width != 0 && (width != this->width || height != this->height)) {
    std::cerr << "Cubemap is broken" << std::endl;
    return;
  }
  this->width = width;
  this->height = height;

  if (texStorage) {
    glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, GL_RGBA8, width, height);
  }
  glTexSubImage2D(side, 0, 0, 0, width, height, GL_RGBA,
                  GL_UNSIGNED_BYTE, data);

  stbi_image_free(data);
}

void CubeMapTexture::load(const std::string &front, const std::string &back,
                          const std::string &top, const std::string &bottom,
                          const std::string &left, const std::string &right,
                          int wrapping, int filtering) {
  if (id)
    destroy();

  glGenTextures(1, &id);
  printf("Cube map generated: %d\n", id);
  glBindTexture(GL_TEXTURE_CUBE_MAP, id);

  loadCubeMapSide(id, GL_TEXTURE_CUBE_MAP_POSITIVE_X, right, true);
  loadCubeMapSide(id, GL_TEXTURE_CUBE_MAP_NEGATIVE_X, left);
  loadCubeMapSide(id, GL_TEXTURE_CUBE_MAP_POSITIVE_Y, top);
  loadCubeMapSide(id, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, bottom);
  loadCubeMapSide(id, GL_TEXTURE_CUBE_MAP_POSITIVE_Z, back);
  loadCubeMapSide(id, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, front);


  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, filtering);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, filtering);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, wrapping);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, wrapping);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, wrapping);
  glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
}

std::shared_ptr<CubeMapTexture>
LoadCubeMapTexture(const std::string &front, const std::string &back,
                   const std::string &top, const std::string &bottom,
                   const std::string &left, const std::string &right,
                   int wrapping, int filtering) {
  auto tex = std::make_shared<CubeMapTexture>();
  tex->load(front, back, top, bottom, left, right, wrapping, filtering);
  printf("Cube map loaded\n");
  return tex;
}
