#include "fluid_system.h"
#include "particle_generator.h"
#include "render/input.h"
#include "render_util.h"
#include "utils.h"
#include <iostream>

using std::cout;
using std::endl;

GLFWwindow *window;

Input input;
void keyCallback(GLFWwindow *window, int key, int scancode, int action,
                 int mods) {
  if (key == GLFW_KEY_Q) {
    glfwSetWindowShouldClose(window, 1);
  }
  input.keyCallback(key, scancode, action, mods);
}

bool updateScene(std::shared_ptr<Scene> scene, double dt) {
  dt *= 0.5;

  bool rt = false;
  if (!scene->getMainCamera())
    return rt;

  if (input.getKeyState(GLFW_KEY_W)) {
    scene->getMainCamera()->move(0, 0, 2 * dt);
    rt = true;
  } else if (input.getKeyState(GLFW_KEY_S)) {
    scene->getMainCamera()->move(0, 0, -dt);
    rt = true;
  }
  if (input.getKeyState(GLFW_KEY_A)) {
    scene->getMainCamera()->move(0, -dt, 0);
    rt = true;
  } else if (input.getKeyState(GLFW_KEY_D)) {
    scene->getMainCamera()->move(0, dt, 0);
    rt = true;
  }

  if (input.getMouseButton(GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
    double dx, dy;
    input.getCursor(dx, dy);
    scene->getMainCamera()->rotateYaw(-dx / 1000.f);
    scene->getMainCamera()->rotatePitch(-dy / 1000.f);
    if (abs(dx) > 0.1 || abs(dy) > 0.1)
      rt = true;
  }

  return rt;
}

void init(uint32_t width, uint32_t height) {
  if (!glfwInit()) {
    fprintf(stderr, "error: Could not initialize GLFW\n");
    exit(1);
  }
  window = glfwCreateWindow(width, height, "opengl", NULL, NULL);
  glfwMakeContextCurrent(window);

  glewExperimental = GL_TRUE;
  glewInit();

  glfwSetKeyCallback(window, keyCallback);

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);

  const GLubyte *glrenderer = glGetString(GL_RENDERER);
  const GLubyte *version = glGetString(GL_VERSION);
  fprintf(stdout, "Renderer: %s\n", glrenderer);
  fprintf(stdout, "OpenGL Version: %s\n", version);
}

int main() {
  FluidSystem fs;
  fs.fluid_domain.corner = {0, 0, 0};
  fs.fluid_domain.size = {0.4, 0.8, 0.4};
  fs.set_particle_size(0.03);
  fs.init();

  fill_triangle(fs, {0.01, 0.01, 0.01}, {0.32, 0.32, 0.32});

  uint32_t w = 1200, h = 900;
  init(w, h);

  RenderUtil ru(&fs, w, h);
  ru.renderer->debug = 0;
  ru.add_particles();
  // ru.add_fluid_domain_object();

  int iter = 0;
  double time = glfwGetTime();
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    input.cursorPosCallback(xpos, ypos);
    input.mouseCallback(GLFW_MOUSE_BUTTON_RIGHT,
                        glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT));

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    if (width != ru.renderer->getWidth() ||
        height != ru.renderer->getHeight()) {
      cout << "Resizing to " << width << " " << height << endl;
      ru.renderer->resize(width, height);
      ru.scene->getMainCamera()->aspect = width / (float)height;
    }

    double newTime = glfwGetTime();
    double dt = newTime - time;
    if (dt > 0.1)
      dt = 0.1;
    updateScene(ru.scene, dt);
    time = newTime;

    // TODO: fix nan
    ru.render();
    glfwSwapBuffers(window);
    // fs.show_profile();

    for (int i = 0; i < 16; ++i) {
      fs.step();
      iter++;
      if (iter % 100 == 0) {
        fs.sort_particles();
      }
    }
  }
  return 0;
}
