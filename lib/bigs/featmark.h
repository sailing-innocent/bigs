/**
 * @file featmark.h
 * @brief The featmakr header file 
 * @author sailing-innocent
 * @date 2025-02-24
 */

#ifndef FEATMARK_GS_H
#define FEATMARK_GS_H

#include <functional>

namespace sail {

int gs_feat_mark(
    std::function<char*(size_t)> geometryBuffer,
    std::function<char*(size_t)> binningBuffer,
    std::function<char*(size_t)> tileBuffer,
    const int P, int F,
    const int width, const int height,
    const float* means3D,
    const float* opacities,
    const float* scales,
    const float* rotations,
    float* feat_img,
    const float* viewmatrix,
    const float* projmatrix,
    const float tan_fovx, float tan_fovy,
    float* out_color,
    float* out_feat,
    int* radii,
    bool debug);

int gs_feat_mark_var(
    std::function<char*(size_t)> geometryBuffer,
    std::function<char*(size_t)> binningBuffer,
    std::function<char*(size_t)> tileBuffer,
    const int P, int F,
    const int width, const int height,
    const float* means3D,
    const float* opacities,
    const float* scales,
    const float* rotations,
    float* feat_img,
    const float* viewmatrix,
    const float* projmatrix,
    const float tan_fovx, float tan_fovy,
    float* out_color,
    float* out_feat,
    float* out_feat_var,
    int* radii,
    bool debug);

int gs_feat_mark_debug(
    std::function<char*(size_t)> geometryBuffer,
    std::function<char*(size_t)> binningBuffer,
    std::function<char*(size_t)> tileBuffer,
    const int P, int F,
    const int width, const int height,
    const float* means3D,
    const float* opacities,
    const float* scales,
    const float* rotations,
    const float* feat,
    const float* viewmatrix,
    const float* projmatrix,
    const float tan_fovx, float tan_fovy,
    float* out_feat_img,
    int* radii,
    bool debug);

} // namespace sail

#endif // FEATMARK_GS_H