#include "light.h"
#include "prd.h"
#include "random.h"
#include <optix.h>
#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>


rtDeclareVariable(rtObject,          top_object,                ,                           );
rtDeclareVariable(rtObject,          top_shadower,              ,                           );
rtDeclareVariable(float,             scene_epsilon,             ,                           );
rtDeclareVariable(unsigned int,      pathtrace_shadow_ray_type, ,                           );

rtBuffer<DirectionalLight>    directional_lights;
rtBuffer<PointLight>          point_lights;
rtBuffer<ParallelogramLight>  parallelogram_lights;

rtDeclareVariable(optix::Ray,        ray,                       rtCurrentRay,               );
rtDeclareVariable(float3,            hit_point,                 attribute hit_point,        ); 

rtDeclareVariable(int,               use_shadow,                ,                           );

rtDeclareVariable(float3,            kd,                        ,                           );

rtDeclareVariable(PerRayData,        current_prd,               rtPayload,                  );
rtDeclareVariable(PerRayData_shadow, prd_shadow,                rtPayload,                  );
rtDeclareVariable(float3,            geometric_normal,          attribute geometric_normal, ); 
rtDeclareVariable(float3,            shading_normal,            attribute shading_normal,   ); 
rtDeclareVariable(float,             t_hit,                     rtIntersectionDistance,     );


RT_PROGRAM void closest_hit() {
  float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
  float3 ffnormal               = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );
  float3 hitpoint               = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_point);

  float3 kd_val = kd;

  current_prd.origin = hitpoint;
  float z1 = rnd(current_prd.seed);
  float z2 = rnd(current_prd.seed);
  float3 p;
  cosine_sample_hemisphere(z1, z2, p);
  Onb onb(ffnormal);
  onb.inverse_transform(p);

  // hack, we actually sample sphere
  if (rnd(current_prd.seed) < 0.5) {
    current_prd.direction = p;
  } else {
    current_prd.direction = -p;
  }

  float3 result = make_float3(0.f);
  for (int i = 0; i < directional_lights.size(); i++) {
    DirectionalLight light = directional_lights[i];
    const float3 L = -light.direction;
    const float nDl = dot(ffnormal, L);
    if (nDl > 0.f) {
      PerRayData_shadow shadow_prd;
      shadow_prd.attenuation = make_float3(1.0f);
      shadow_prd.inShadow = false;
      Ray shadow_ray = make_Ray(hitpoint, L, pathtrace_shadow_ray_type, scene_epsilon, RT_DEFAULT_MAX);
      rtTrace(top_shadower, shadow_ray, shadow_prd);

      if (!shadow_prd.inShadow) {
        result += nDl * light.emission * kd_val * shadow_prd.attenuation;
      }
    }
  }

  for (int i = 0; i < point_lights.size(); i++) {
    PointLight light = point_lights[i];
    const float  Ldist = length(light.position - hitpoint);
    const float3 L = normalize(light.position - hitpoint);
    const float nDl = dot(ffnormal, L);
    if (nDl > 0.f) {
      PerRayData_shadow shadow_prd;
      shadow_prd.attenuation = make_float3(1.0f);
      shadow_prd.inShadow = false;
      Ray shadow_ray = make_Ray(hitpoint, L, pathtrace_shadow_ray_type, scene_epsilon, Ldist - scene_epsilon);
      rtTrace(top_shadower, shadow_ray, shadow_prd);

      if (!shadow_prd.inShadow) {
        result += kd_val * nDl * light.emission / Ldist / Ldist * shadow_prd.attenuation;
      }
    }
  }

  current_prd.radiance = result;
}

RT_PROGRAM void any_hit() {
}

RT_PROGRAM void shadow_any_hit() {
  rtIgnoreIntersection();
  // if (!use_shadow) {
  //   rtTerminateRay();
  //   return;
  // }

  // prd_shadow.attenuation = tint;
}
