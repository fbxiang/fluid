#include "sph_gpu.h"
#include "render/input.h"
#include "sph_render_util.h"
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

inline float rand_float(float start, float range) {
  float random = ((float)rand()) / (float)RAND_MAX;
  float r = random * range;
  return start + r;
}

void fill_block(SPH_GPU &sph, glm::vec3 corner, glm::vec3 size, float jitter_size=0.f) {
  float step = sph.solver_params.particle_size;
  std::vector<glm::vec3> positions;
  glm::uvec3 n = size / step;
  for (uint32_t i = 0; i < n.x; ++i) {
    for (uint32_t j = 0; j < n.y; ++j) {
      for (uint32_t k = 0; k < n.z; ++k) {
        glm::vec3 jitter = {rand_float(-step / 4.f, step / 4.f),
                            rand_float(-step / 4.f, step / 4.f),
                            rand_float(-step / 4.f, step / 4.f)};
        positions.push_back({{corner + glm::vec3(i, j, k) * step + jitter * jitter_size}});
      }
    }
  }
  sph.init(positions);
}

int main() {
  SPH_GPU fs;
  fs.fluid_domain.corner = {0, 0, 0};
  fs.fluid_domain.size = {0.4, 0.8, 0.4};
  fs.set_particle_size_and_density(0.03, 1000);
  fill_block(fs, {0.1, 0.01, 0.1}, {0.12, 0.7, 0.12});
  fs.sim_init();

  uint32_t w = 1200, h = 900;
  init(w, h);

  RenderUtil ru(&fs, w, h);
  ru.renderer->debug = 0;
  ru.add_particles();

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

    ru.render();
    glfwSwapBuffers(window);

    for (int i = 0; i < 16; ++i) {
      fs.step();
      iter++;
    }
  }
  return 0;
}
