#include "texture.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const Texture Texture::Empty;

void Texture::load(const std::string& filename, int mipmap, int wrapping, int minFilter, int magFilter) {
  if (id) destroy();

  glActiveTexture(GL_TEXTURE0);
  glGenTextures(1, &id);
  glBindTexture(GL_TEXTURE_2D, id);

  int width, height, nrChannels;
  unsigned char *data = stbi_load(filename.c_str(), &width, &height, &nrChannels, STBI_rgb_alpha); 
  printf("%d channels loaded\n", nrChannels);

  glTexStorage2D(GL_TEXTURE_2D, 4, GL_RGBA8, width, height);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, data);
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

std::shared_ptr<Texture> LoadTexture(const std::string& filename, int mipmap, int wrapping, int minFilter, int maxFilter) {
  auto tex = std::make_shared<Texture>();
  tex->load(filename, mipmap, wrapping, minFilter, maxFilter);
  return tex;
}

GLuint Texture::getId() const {
  return id;
}


void writeToFile(GLuint textureId, GLuint width, GLuint height, std::string filename) {
  uint8_t data[width * height * 4];
  glBindTexture(GL_TEXTURE_2D, textureId);
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
  stbi_write_png(filename.c_str(), width, height, 4, data, width * 4);
}
