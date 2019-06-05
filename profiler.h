#pragma once
#include <chrono>
#include <functional>
#include <map>
#include <iostream>

namespace profiler {

void start(std::string name); 

void stop(std::string name);

void show(); 

double get(std::string name);

} // namespace profiler
