/**
 * @file vis.cu
 * @brief Gaussian Visualization
 * @author sailing-innocent
 * @date 2025-01-01
 */

#include "SailCu/gs/vis.h"
#include <cuda_runtime.h>
#include <glm/glm.hpp>

namespace sail {

__global__ void point_to_bb_verts_kernel(const float* d_pos, float* d_verts, int* d_faces, int num_points) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_points) { return; }

	const float3 pos = reinterpret_cast<const float3*>(d_pos)[idx];
	float3 verts[8] = {
		make_float3(-0.5f, -0.5f, -0.5f),
		make_float3(0.5f, -0.5f, -0.5f),
		make_float3(0.5f, 0.5f, -0.5f),
		make_float3(-0.5f, 0.5f, -0.5f),
		make_float3(-0.5f, -0.5f, 0.5f),
		make_float3(0.5f, -0.5f, 0.5f),
		make_float3(0.5f, 0.5f, 0.5f),
		make_float3(-0.5f, 0.5f, 0.5f),
	};

	int3 faces[12] = {
		make_int3(0, 2, 1),
		make_int3(0, 3, 2),
		make_int3(4, 5, 6),
		make_int3(4, 6, 7),
		make_int3(0, 4, 7),
		make_int3(0, 7, 3),
		make_int3(1, 6, 5),
		make_int3(1, 2, 6),
		make_int3(0, 1, 5),
		make_int3(0, 5, 4),
		make_int3(2, 7, 6),
		make_int3(2, 3, 7),
	};

	for (int i = 0; i < 8; i++) {
		float3 result = make_float3(
			pos.x + verts[i].x,
			pos.y + verts[i].y,
			pos.z + verts[i].z);
		reinterpret_cast<float3*>(d_verts)[idx * 8 + i] = result;
	}

	for (int i = 0; i < 12; i++) {
		int3 result = make_int3(
			idx * 8 + faces[i].x,
			idx * 8 + faces[i].y,
			idx * 8 + faces[i].z);
		reinterpret_cast<int3*>(d_faces)[idx * 12 + i] = result;
	}
}

void point_to_bb_verts(const float* d_pos, float* d_verts, int* d_faces, int num_points) noexcept {
	constexpr int block_size = 256;
	int num_blocks = (num_points + block_size - 1) / block_size;
	point_to_bb_verts_kernel<<<num_blocks, block_size>>>(d_pos, d_verts, d_faces, num_points);
}

__global__ void expand_color_kernel(const float* d_color, float* d_expanded_color, int num_points) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_points) { return; }

	const float3 color = reinterpret_cast<const float3*>(d_color)[idx];

	float scale = 0.05f;
	float3 diff_color;
	for (int i = 0; i < 8; i++) {
		// use a slightly different color for each vertex
		diff_color = make_float3(
			scale * i + color.x,
			scale * i + color.y,
			scale * i + color.z);
		reinterpret_cast<float3*>(d_expanded_color)[idx * 8 + i] = diff_color;
	}
}

void expand_color(const float* d_color, float* d_expanded_color, int num_points) noexcept {
	constexpr int block_size = 256;
	int num_blocks = (num_points + block_size - 1) / block_size;
	expand_color_kernel<<<num_blocks, block_size>>>(d_color, d_expanded_color, num_points);
}

__global__ void point_bb_transform_kernel(const float* d_scales, const float* d_rotations, const float* d_pos, float* d_verts, int num_points) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_points) { return; }

	const float3 scale = reinterpret_cast<const float3*>(d_scales)[idx];
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = scale.x;
	S[1][1] = scale.y;
	S[2][2] = scale.z;
	const float4 rot = reinterpret_cast<const float4*>(d_rotations)[idx];
	glm::vec4 q = glm::vec4(rot.x, rot.y, rot.z, rot.w);// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));
	glm::mat3 M = S * R;// M=S^TR^T
	// glm::mat3 T = glm::transpose(M) * M;
	glm::mat3 T = glm::transpose(M) * R;

	const float3 pos = reinterpret_cast<const float3*>(d_pos)[idx];

	float3 verts[8] = {
		make_float3(-0.5f, -0.5f, -0.5f),
		make_float3(0.5f, -0.5f, -0.5f),
		make_float3(0.5f, 0.5f, -0.5f),
		make_float3(-0.5f, 0.5f, -0.5f),
		make_float3(-0.5f, -0.5f, 0.5f),
		make_float3(0.5f, -0.5f, 0.5f),
		make_float3(0.5f, 0.5f, 0.5f),
		make_float3(-0.5f, 0.5f, 0.5f),
	};

	float s_ = 2.0f;
	// float s_ = 4.0f;
	for (int i = 0; i < 8; i++) {

		float3 vert_i = make_float3(s_ * verts[i].x, s_ * verts[i].y, s_ * verts[i].z);

		float3 result = make_float3(
			T[0][0] * vert_i.x + T[0][1] * vert_i.y + T[0][2] * vert_i.z,
			T[1][0] * vert_i.x + T[1][1] * vert_i.y + T[1][2] * vert_i.z,
			T[2][0] * vert_i.x + T[2][1] * vert_i.y + T[2][2] * vert_i.z);
		result.x += pos.x;
		result.y += pos.y;
		result.z += pos.z;
		reinterpret_cast<float3*>(d_verts)[idx * 8 + i] = result;
	}
}

void point_bb_verts_transform(const float* d_scales, const float* d_rotations, const float* d_pos, float* d_verts, int num_points) noexcept {
	constexpr int block_size = 256;
	int num_blocks = (num_points + block_size - 1) / block_size;
	point_bb_transform_kernel<<<num_blocks, block_size>>>(d_scales, d_rotations, d_pos, d_verts, num_points);
}

}// namespace sail