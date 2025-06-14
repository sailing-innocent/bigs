/**
 * @file featmark_impl.h
 * @brief The featmark impl 
 * @author sailing-innocent
 * @date 2025-02-24
 */

#ifndef FEATMARK_IMPL_H
#define FEATMARK_IMPL_H

#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace sail {

void gs_project(
    int P,
    const float* means3D,
    const glm::vec3* scales,
    const glm::vec4* rotations,
    const float* opacities,
    const float* viewmatrix,
    const float* projmatrix,
    const int W, int H,
    const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy,
    int* radii,
    float2* means2D,
    float* depths,
    float* cov3Ds,
    float4* conic_opacity,
    const dim3 grid,
    uint32_t* tiles_touched,
    bool prefiltered);

void feat_mark_impl(
    const dim3 grid, dim3 block,
    const uint2* ranges,
    const uint32_t* point_list,
    int F,
    int W, int H,
    const float2* means2D,
    const float4* conic_opacity,
    float* feat_img,
    float* out_feat,
    float* out_color);

void feat_mark_var_impl(
    const dim3 grid, dim3 block,
    const uint2* ranges,
    const uint32_t* point_list,
    int F,
    int W, int H,
    const float2* means2D,
    const float4* conic_opacity,
    float* feat_img,
    float* out_feat,
    float* out_feat_var,
    float* out_color);

void feat_mark_debug_impl(
    const dim3 grid, dim3 block,
    const uint2* ranges,
    const uint32_t* point_list,
    int F,
    int W, int H,
    const float2* means2D,
    const float4* conic_opacity,
    const float* feat,
    float* out_feat_img);


}// namespace sail

#endif //FEATMARK_IMPL_H