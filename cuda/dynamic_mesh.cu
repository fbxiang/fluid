#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optix_device.h>

using namespace optix;

struct Vertex {
  float3 position;
  float3 normal;
};

rtBuffer<Vertex,  1>  vertex_buffer;

rtDeclareVariable(float3,     texcoord,         attribute texcoord,         ); 
rtDeclareVariable(float3,     geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3,     shading_normal,   attribute shading_normal,   ); 

rtDeclareVariable(float3,     hit_point,        attribute hit_point,        ); 
rtDeclareVariable(float3,     back_hit_point,   attribute back_hit_point,   ); 
rtDeclareVariable(float3,     front_hit_point,  attribute front_hit_point,  ); 

rtDeclareVariable(optix::Ray, ray,              rtCurrentRay,               );

static __device__ void meshIntersect(int primIdx) {
  const Vertex v0   = vertex_buffer[ 3 * primIdx     ];
  const Vertex v1   = vertex_buffer[ 3 * primIdx + 1 ];
  const Vertex v2   = vertex_buffer[ 3 * primIdx + 2 ];

  float3 n;
  float t, beta, gamma;
  if (intersect_triangle(ray, v0.position, v1.position, v2.position, n, t, beta, gamma)) {
    if(  rtPotentialIntersection( t ) ) {
      geometric_normal = normalize( n );
      float3 n0        = v0.normal;
      float3 n1        = v1.normal;
      float3 n2        = v2.normal;
      shading_normal   = normalize( n1*beta + n2*gamma + n0*(1.0f-beta-gamma) );

      // float2 t0        = v0.texCoord;
      // float2 t1        = v1.texCoord;
      // float2 t2        = v2.texCoord;
      texcoord         = make_float3(0.f, 0.f, 0.f);

      hit_point        = ray.origin + ray.direction * t;

      rtReportIntersection(0);
    }
  }
}


RT_PROGRAM void mesh_intersect( int primIdx ) {
  meshIntersect( primIdx );
}


RT_PROGRAM void mesh_bounds(int primIdx, float result[6]) {
  const float3 v0   = vertex_buffer[ 3 * primIdx     ].position;
  const float3 v1   = vertex_buffer[ 3 * primIdx + 1 ].position;
  const float3 v2   = vertex_buffer[ 3 * primIdx + 2 ].position;
  const float  area = length(cross(v1-v0, v2-v0));

  optix::Aabb* aabb = (optix::Aabb*)result;
  
  if(area > 0.0f && !isinf(area)) {
    aabb->m_min = fminf( fminf( v0, v1), v2 );
    aabb->m_max = fmaxf( fmaxf( v0, v1), v2 );
    // printf("%f %f %f %f %f %f\n", aabb->m_min.x, aabb->m_min.y, aabb->m_min.z, aabb->m_max.x, aabb->m_max.y, aabb->m_max.z);
  } else {
    // printf("invalidated.\n");
    aabb->invalidate();
  }
}
