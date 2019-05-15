#include "sph_gpu.h"
#include "render/input.h"
#include "gpu_render_util.h"
#include "utils.h"
#include <iostream>
#include "profiler.h"

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

std::vector<glm::vec3> fill_block(glm::vec3 corner, glm::vec3 size,
                                  float step) {
  std::vector<glm::vec3> positions;
  glm::ivec3 n = size / step;
  for (int i = 0; i < n.x; ++i) {
    for (int j = 0; j < n.y; ++j) {
      for (int k = 0; k < n.z; ++k) {
        glm::vec3 jitter = {rand_float(-step / 4.f, step / 4.f),
                            rand_float(-step / 4.f, step / 4.f),
                            rand_float(-step / 4.f, step / 4.f)};
        positions.push_back({{corner + glm::vec3(i, j, k) * step + jitter}});
      }
    }
  }
  return positions;
}

int main() {
  float h = 0.015;
  SPH_GPU fs(h);
  fs.set_domain({0, 0, 0}, {0.5, 1.0, 0.5});
  fs.cuda_init();
  fs.marching_cube_init();
  std::vector<glm::vec3> positions = fill_block({0.1, 0.1, 0.1}, {0.3, 0.3, 0.3}, h);
  fs.add_particles(positions);
  sph::print_summary();

  uint32_t W = 1200, H = 900;
  init(W, H);

  GPURenderUtil ru(&fs, W, H);
  ru.renderer->debug = 0;

  int frame = 0;
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
    if (width != (int)ru.renderer->getWidth() ||
        height != (int)ru.renderer->getHeight()) {
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

    // ru.render_debug();
    profiler::start("render");
    ru.render();
    profiler::stop("render");
    glfwSwapBuffers(window);

    float time = 0;
    profiler::start("simulate");
    while (time < 1.f / 60.f) {
      float dt = fs.step_regular();
      fs.update_mesh();
      // printf("Time step: %f\n", dt);
      time += dt;
    }
    profiler::stop("simulate");

    profiler::show();
  }
  return 0;
}
