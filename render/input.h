#include <map>

class Input {
  std::map<int, int> keyState;
  std::map<int, int> mouseState;

  double lastX = -1, lastY = -1, dx, dy;

  bool firstTime = true;

 public:
  void keyCallback(int key, int scancode, int action, int mods);
  void cursorPosCallback(double x, double y);
  void mouseCallback(int button, int state);

 public:
  int getKeyState(int key) const;
  void getCursor(double& dx, double& dy);
  int getMouseButton(int button);
};
