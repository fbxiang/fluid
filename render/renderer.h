#pragma once
#include <stdint.h>
#include <GL/glew.h>
#include "scene.h"
#include "shader.h"
#include <map>

#define N_COLOR_ATTACHMENTS 3

class Renderer {
 public:
  int debug = 0;

 public:
  Renderer(GLuint w, GLuint h);
  void init();
  void exit();
  void resize(GLuint w, GLuint h);

 private:
  bool initialized;

 private:
  Shader* gbufferShader;
  Shader* deferredShader;
  Shader* skyboxShader;

 protected:
  GLuint width, height;

 public:
  inline GLuint getWidth() const { return width; }
  inline GLuint getHeight() const { return height; }

private:
  GLuint g_fbo;
  GLuint colortex[N_COLOR_ATTACHMENTS];
  GLuint depthtex;

  void initGbufferFramebuffer();
  void initColortex();
  void initDepthtex();
  void bindAttachments();

  void deleteGbufferFramebuffer();
  void deleteColortex();
  void deleteDepthtex();

 private:
  GLuint quadVAO, quadVBO;
  void initDeferredQuad();
  void deleteDeferredQuad();

 private:
  GLuint composite_fbo;
  GLuint compositeTex;
  void initCompositeFramebuffer();
  void initCompositeTex();
  void deleteCompositeFramebuffer();
  void deleteCompositeTex();

private:
  void gbufferPass(std::shared_ptr<Scene> scene);
  void gbufferPass(std::shared_ptr<Scene> scene, GLuint fbo);
  void deferredPass(std::shared_ptr<Scene> scene);
  void deferredPass(std::shared_ptr<Scene> scene, GLuint fbo);

 public:
  void renderScene(std::shared_ptr<Scene> scene);
  void renderSceneToFile(std::shared_ptr<Scene> scene, std::string filename);
  void reloadShaders();
};
