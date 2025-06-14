/**
 * @file debug.cuh
 * @brief The Debug Kernel for Feature Marking
 * @author sailing-innocent
 * @date 2024-12-08
 */

#ifndef FEATMARK_GS_DEBUG_CUH
#define FEATMARK_GS_DEBUG_CUH

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "cov.cuh"
#include "auxiliary.cuh"

namespace sail::FORWARD {

namespace cg = cooperative_groups;

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void gs_project_CUDA(
	int P,
	const float* orig_points,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float* opacities,
	const float* viewmatrix,
	const float* projmatrix,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched) {
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P) { return; }
	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;
	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view)) {
		return;
	}
	// Transform point by projecting
	float3 p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]};
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};
	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters.
	const float* cov3D;
	FORWARD::computeCov3D(scales[idx], 1.0f, rotations[idx], cov3Ds + idx * 6);
	cov3D = cov3Ds + idx * 6;
	// Compute 2D screen-space covariance matrix
	float3 cov = FORWARD::computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f) {
		return;
	}
	float det_inv = 1.f / det;
	float3 conic = {cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv};
	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles.
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = {ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H)};
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0) {
		return;
	}
	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = {conic.x, conic.y, conic.z, opacities[idx]};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

}// namespace sail::FORWARD

#endif // FEATMARK_GS_DEBUG_CUH