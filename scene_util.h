#pragma once
#include "render/objectLoader.h"
#include "render/scene.h"

void setupSponza(std::shared_ptr<Scene> scene, uint32_t width,
                 uint32_t height) {
  auto objects = LoadObj("/home/fx/Scenes/sponza/sponza.obj");
  for (auto obj : objects) {
    obj->scale = glm::vec3(0.003f);
    obj->position *= 0.003f;
    scene->addObject(obj);
  }

  auto cam = NewObject<Camera>();

  cam->position = {1.5, 0.5, -0.5};
  cam->rotatePitch(-0.2);
  cam->rotateYaw(0.2);
  cam->rotateYaw(1.6);
  cam->rotatePitch(0.06);
  cam->fovy = glm::radians(45.f);
  cam->aspect = width / (float)height;
  scene->addObject(cam);
  scene->setMainCamera(cam);
  // scene->addDirectionalLight({glm::vec3(0, -1, 0.1), glm::vec3(1, 1, 1)});
  scene->addPointLight({glm::vec3(0, 1, 0), glm::vec3(0.5, 0.5, 0.5)});
}

void setupEmpty(std::shared_ptr<Scene> scene, uint32_t width, uint32_t height) {
  auto cam = NewObject<Camera>();

  cam->position = {0.25, 0.5, 2};
  cam->fovy = glm::radians(45.f);
  cam->aspect = width / (float)height;
  scene->addObject(cam);
  scene->setMainCamera(cam);
  // scene->addDirectionalLight({glm::vec3(0, -1, 0.1), glm::vec3(1, 1, 1)});
  scene->addPointLight({glm::vec3(1, 1, 0), glm::vec3(1, 1, 1)});
  scene->addPointLight({glm::vec3(0, 1, 1), glm::vec3(1, 1, 1)});
  // scene->addPointLight({glm::vec3(0, 0, 2), glm::vec3(1, 1, 1)});
}

void setupTest(std::shared_ptr<Scene> scene, uint32_t width, uint32_t height) {
  auto objects = LoadObj("/home/fx/Downloads/002_master_chef_can.obj");
  for (auto obj : objects) {
    obj->material.kd = {1, 1, 1};
    scene->addObject(obj);
  }
  auto cam = NewObject<Camera>();

  cam->position = {0, 0, 2};
  cam->fovy = glm::radians(45.f);
  cam->aspect = width / (float)height;
  scene->addObject(cam);
  scene->setMainCamera(cam);
  scene->addDirectionalLight({glm::vec3(0, -1, 0.1), glm::vec3(1, 1, 1)});
  scene->addPointLight({glm::vec3(1, 1, 0), glm::vec3(1, 1, 1)});
  scene->addPointLight({glm::vec3(0, 1, 1), glm::vec3(1, 1, 1)});
}
