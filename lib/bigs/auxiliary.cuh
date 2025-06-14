/**
 * @file auxiliary.h
 * @brief The auxiliary functions
 * @author sailing-innocent
 * @date 2024-12-10
 */

#ifndef FEATMARK_GS_AUX_H
#define FEATMARK_GS_AUX_H

#include "config.h"
#include <cmath>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector_types.h>

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE / 32)

namespace sail {

__forceinline__ __host__ __device__ float ndc2pix_impl(float v, int S) {
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __host__ __device__ unsigned int get_rect_impl_min(const float p, int max_radius, unsigned int grid, unsigned int BLOCK) {
	return min(grid, max(0u, (unsigned int)((p - max_radius) / BLOCK)));
}

__forceinline__ __host__ __device__ unsigned int get_rect_impl_max(const float p, int max_radius, unsigned int grid, unsigned int BLOCK) {
	return min(grid, max(0u, (unsigned int)((p + max_radius + BLOCK - 1) / BLOCK)));
}

__forceinline__ __host__ __device__ void transform_point_4x3_impl(const float px, const float py, const float pz, const float* matrix, float& x, float& y, float& z) {
	x = matrix[0] * px + matrix[4] * py + matrix[8] * pz + matrix[12];
	y = matrix[1] * px + matrix[5] * py + matrix[9] * pz + matrix[13];
	z = matrix[2] * px + matrix[6] * py + matrix[10] * pz + matrix[14];
}

__forceinline__ __host__ __device__ void transform_point_4x4_impl(
	const float px, const float py, const float pz, const float* matrix, float& x, float& y, float& z, float& w) {
	x = matrix[0] * px + matrix[4] * py + matrix[8] * pz + matrix[12];
	y = matrix[1] * px + matrix[5] * py + matrix[9] * pz + matrix[13];
	z = matrix[2] * px + matrix[6] * py + matrix[10] * pz + matrix[14];
	w = matrix[3] * px + matrix[7] * py + matrix[11] * pz + matrix[15];
}

__forceinline__ __host__ __device__ void transform_vec_4x3_impl(const float& px, const float& py, const float& pz, const float* matrix, float& x, float& y, float& z) {
	x = matrix[0] * px + matrix[4] * py + matrix[8] * pz;
	y = matrix[1] * px + matrix[5] * py + matrix[9] * pz;
	z = matrix[2] * px + matrix[6] * py + matrix[10] * pz;
}

__forceinline__ __host__ __device__ void transform_vec_4x3_transpose_impl(
	const float& px, const float& py, const float& pz, const float* matrix, float& x, float& y, float& z) {
	x = matrix[0] * px + matrix[1] * py + matrix[2] * pz;
	y = matrix[4] * px + matrix[5] * py + matrix[6] * pz;
	z = matrix[8] * px + matrix[9] * py + matrix[10] * pz;
}

__forceinline__ __host__ __device__ void dnormvdv_impl_3(float v_x, float v_y, float v_z, float dv_x, float dv_y, float dv_z, float& dnormvdv_x, float& dnormvdv_y, float& dnormvdv_z) {
	float sum2 = v_x * v_x + v_y * v_y + v_z * v_z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	dnormvdv_x = ((+sum2 - v_x * v_x) * dv_x - v_y * v_x * dv_y - v_z * v_x * dv_z) * invsum32;
	dnormvdv_y = (-v_x * v_y * dv_x + (sum2 - v_y * v_y) * dv_y - v_z * v_y * dv_z) * invsum32;
	dnormvdv_z = (-v_x * v_z * dv_x - v_y * v_z * dv_y + (sum2 - v_z * v_z) * dv_z) * invsum32;
}

__forceinline__ __host__ __device__ void dnormvdv_impl_4(float v_x, float v_y, float v_z, float v_w, float dv_x, float dv_y, float dv_z, float dv_w, float& dnormvdv_x, float& dnormvdv_y, float& dnormvdv_z, float& dnormvdv_w) {
	float sum2 = v_x * v_x + v_y * v_y + v_z * v_z + v_w * v_w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float vdv_x = v_x * dv_x;
	float vdv_y = v_y * dv_y;
	float vdv_z = v_z * dv_z;
	float vdv_w = v_w * dv_w;
	float vdv_sum = vdv_x + vdv_y + vdv_z + vdv_w;

	dnormvdv_x = ((sum2 - v_x * v_x) * dv_x - v_x * (vdv_sum - vdv_x)) * invsum32;
	dnormvdv_y = ((sum2 - v_y * v_y) * dv_y - v_y * (vdv_sum - vdv_y)) * invsum32;
	dnormvdv_z = ((sum2 - v_z * v_z) * dv_z - v_z * (vdv_sum - vdv_z)) * invsum32;
	dnormvdv_w = ((sum2 - v_w * v_w) * dv_w - v_w * (vdv_sum - vdv_w)) * invsum32;
}

__forceinline__ __host__ __device__ float sigmoid_impl(float x) {
	return 1.0f / (1.0f + expf(-x));
}

}// namespace sail

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f};

__forceinline__ __device__ float ndc2Pix(float v, int S) {
	return sail::ndc2pix_impl(v, S);
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid) {
	rect_min = {
		sail::get_rect_impl_min(p.x, max_radius, grid.x, BLOCK_X),
		sail::get_rect_impl_min(p.y, max_radius, grid.y, BLOCK_Y)};
	rect_max = {
		sail::get_rect_impl_max(p.x, max_radius, grid.x, BLOCK_X),
		sail::get_rect_impl_max(p.y, max_radius, grid.y, BLOCK_Y)};
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix) {
	float3 t;
	sail::transform_point_4x3_impl(p.x, p.y, p.z, matrix, t.x, t.y, t.z);
	return t;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix) {
	float4 t;
	sail::transform_point_4x4_impl(p.x, p.y, p.z, matrix, t.x, t.y, t.z, t.w);
	return t;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix) {
	float3 t;
	sail::transform_vec_4x3_impl(p.x, p.y, p.z, matrix, t.x, t.y, t.z);
	return t;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix) {
	float3 t;
	sail::transform_vec_4x3_transpose_impl(p.x, p.y, p.z, matrix, t.x, t.y, t.z);
	return t;
}

__forceinline__ __device__ __host__ float3 dnormvdv(float3 v, float3 dv) {
	float3 dnormvdv;
	sail::dnormvdv_impl_3(v.x, v.y, v.z, dv.x, dv.y, dv.z, dnormvdv.x, dnormvdv.y, dnormvdv.z);
	return dnormvdv;
}

__forceinline__ __device__ __host__ float4 dnormvdv(float4 v, float4 dv) {
	float4 dnormvdv;
	sail::dnormvdv_impl_4(v.x, v.y, v.z, v.w, dv.x, dv.y, dv.z, dv.w, dnormvdv.x, dnormvdv.y, dnormvdv.z, dnormvdv.w);

	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x) {
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(int idx,
										   const float* orig_points,
										   const float* viewmatrix,
										   const float* projmatrix,
										   bool prefiltered,
										   float3& p_view) {
	float3 p_orig = {orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]};

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = {p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w};
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered) {
			// printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

#endif //FEATMARK_GS_AUX_H