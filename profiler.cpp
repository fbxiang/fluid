#include "profiler.h"

static std::map<std::string, long> current_time;
static std::map<std::string, long> total_time;
static std::map<std::string, long> times;

void profiler::start(std::string name) {
  long value_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::time_point_cast<std::chrono::milliseconds>(
                          std::chrono::high_resolution_clock::now())
                          .time_since_epoch())
                      .count();

  current_time[name] = value_ms;
}

void profiler::stop(std::string name) {
  long value_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::time_point_cast<std::chrono::milliseconds>(
                          std::chrono::high_resolution_clock::now())
                          .time_since_epoch())
                      .count();

  if (current_time.find(name) == current_time.end()) {
    return;
  }

  if (total_time.find(name) == total_time.end()) {
    total_time[name] = 0l;
    times[name] = 0l;
  }

  total_time[name] += value_ms - current_time[name];
  ++times[name];
}

void profiler::show() {
  for (auto kp : total_time) {
    std::cout << kp.first << ": " << kp.second / (double)times[kp.first] << "ms" << std::endl;
  }
}
