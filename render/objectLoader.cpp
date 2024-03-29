#include "objectLoader.h"
#include <string>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "mesh.h"

std::vector<std::shared_ptr<Object> > LoadObj(const std::string file) {
  auto objects = std::vector<std::shared_ptr<Object> >();

  Assimp::Importer importer;
  const aiScene* scene = importer.ReadFile(file,
                                           aiProcess_CalcTangentSpace |
                                           aiProcess_Triangulate |
                                           aiProcess_GenNormals |
                                           aiProcess_PreTransformVertices |
                                           aiProcess_FlipUVs);

  if (!scene) {
    fprintf(stderr, "%s\n", importer.GetErrorString());
    return objects;
  }

  printf("Loaded %d meshes, %d materials, %d textures\n", scene->mNumMeshes, scene->mNumMaterials, scene->mNumTextures);

  std::vector<Material> mats(scene->mNumMaterials);
  for (int i = 0; i < scene->mNumMaterials; i++) {
    auto* m = scene->mMaterials[i];
    aiColor3D color = aiColor3D(0,0,0);
    m->Get(AI_MATKEY_COLOR_AMBIENT, color);
    mats[i].ka = glm::vec3(color.r, color.g, color.b);
    m->Get(AI_MATKEY_COLOR_DIFFUSE, color);
    mats[i].kd = glm::vec3(color.r, color.g, color.b);
    m->Get(AI_MATKEY_COLOR_SPECULAR, color);
    mats[i].ks = glm::vec3(color.r, color.g, color.b);

    std::string parentdir = file.substr(0, file.find_last_of('/')) + "/";

    aiString path;
    if (m->GetTextureCount(aiTextureType_DIFFUSE) > 0 &&
        m->GetTexture(aiTextureType_DIFFUSE, 0, &path) == AI_SUCCESS)
    {
      std::string p = std::string(path.C_Str());
      std::string fullPath = parentdir + p;
      
      auto tex = LoadTexture(fullPath, 0);
      mats[i].kd_map = tex;
      printf("Diffuse texture found at %s\n", fullPath.c_str());
    }

    if (m->GetTextureCount(aiTextureType_SPECULAR) > 0 &&
        m->GetTexture(aiTextureType_SPECULAR, 0, &path) == AI_SUCCESS)
    {
      std::string p = std::string(path.C_Str());
      std::string fullPath = parentdir + p;
      
      auto tex = LoadTexture(fullPath, 0);
      mats[i].ks_map = tex;
      printf("Specular texture found at %s\n", fullPath.c_str());
    }

    if (m->GetTextureCount(aiTextureType_HEIGHT) > 0 &&
        m->GetTexture(aiTextureType_HEIGHT, 0, &path) == AI_SUCCESS)
    {
      std::string p = std::string(path.C_Str());
      std::string fullPath = parentdir + p;
      
      auto tex = LoadTexture(fullPath, 0);
      mats[i].height_map = tex;
      printf("Height texture found at %s\n", fullPath.c_str());
    }

    if (m->GetTextureCount(aiTextureType_NORMALS) > 0 &&
        m->GetTexture(aiTextureType_NORMALS, 0, &path) == AI_SUCCESS)
    {
      std::string p = std::string(path.C_Str());
      std::string fullPath = parentdir + p;
      
      auto tex = LoadTexture(fullPath, 0);
      mats[i].normal_map = tex;
      printf("Normal texture found at %s\n", fullPath.c_str());
    }
  }

  for (int i = 0; i < scene->mNumMeshes; i++) {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    auto mesh = scene->mMeshes[i];
    if (!mesh->HasFaces()) continue;

    for (int v = 0; v < mesh->mNumVertices; v++) {
      glm::vec3 normal = glm::vec3(0);
      glm::vec2 texcoord = glm::vec2(0);
      glm::vec3 position = { mesh->mVertices[v].x, mesh->mVertices[v].y, mesh->mVertices[v].z };
      glm::vec3 tangent = glm::vec3(0);
      glm::vec3 bitangent = glm::vec3(0);
      if (mesh->HasNormals()) {
        normal = { mesh->mNormals[v].x, mesh->mNormals[v].y, mesh->mNormals[v].z };
      }
      if (mesh->HasTextureCoords(0)) {
        texcoord = { mesh->mTextureCoords[0][v].x, mesh->mTextureCoords[0][v].y };
      }
      if (mesh->HasTangentsAndBitangents()) {
        tangent =  { mesh->mTangents[v].x, mesh->mTangents[v].y, mesh->mTangents[v].z };
        bitangent = { mesh->mBitangents[v].x, mesh->mBitangents[v].y, mesh->mBitangents[v].z };
      }
      vertices.push_back(Vertex(position, normal, texcoord, tangent, bitangent));
    }
    for (int f = 0; f < mesh->mNumFaces; f++) {
      auto face = mesh->mFaces[f];
      if (face.mNumIndices != 3) {
        fprintf(stderr, "A face with %d indices is found and ignored.", face.mNumIndices);
        continue;
      }
      indices.push_back(face.mIndices[0]);
      indices.push_back(face.mIndices[1]);
      indices.push_back(face.mIndices[2]);
    }
    auto m = std::make_shared<TriangleMesh>(vertices, indices);
    auto obj = NewObject<Object>(m);
    objects.push_back(obj);

    obj->material = mats[mesh->mMaterialIndex];
  }
  return objects;
}
