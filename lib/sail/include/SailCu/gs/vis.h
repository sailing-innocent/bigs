#pragma once
/**
 * @file vis.h
 * @brief Gaussian Visualize Helper
 * @author sailing-innocent
 * @date 2025-01-01
 */

namespace sail {

void point_to_bb_verts(const float* d_pos, float* d_verts, int* d_faces, int num_points) noexcept;

void expand_color(const float* d_color, float* d_expanded_color, int num_points) noexcept;

void point_bb_verts_transform(const float* d_scales, const float* d_rotations, const float* d_pos, float* d_verts, int num_points) noexcept;

}// namespace sail::cu
