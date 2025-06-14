/**
 * @file impl.cu
 * @brief FeatMark Implementation
 * @author sailing-innocent
 * @date 2024-12-10
 */

#include <cuda.h>
#include "featmark_impl.h"
#include <cooperative_groups.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "auxiliary.cuh"
#include "project.cuh"

namespace sail {
 
namespace cg = cooperative_groups;

template<uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
    featMarkCUDA(
        const uint2* __restrict__ ranges,
        const uint32_t* __restrict__ point_list,
        int F,
        int W, int H,
        const float2* __restrict__ points_xy_image,
        const float4* __restrict__ conic_opacity,
        const float* __restrict__ feat_img,
        float* __restrict__ out_feat,
        float* __restrict__ out_color) {
    // Identify current tile and associated min/max pixel range.
    auto block = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;

    uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
    uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
    uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
    uint32_t pix_id = W * pix.y + pix.x;
    float2 pixf = {(float)pix.x, (float)pix.y};

    // Check if this thread is associated with a valid pixel or outside.
    bool inside = pix.x < W && pix.y < H;
    // Done threads can help with fetching, but don't rasterize
    bool done = !inside;

    // Load start/end range of IDs to process in bit sorted list.
    uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;

    // Allocate storage for batches of collectively fetched data.
    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];

    // Initialize helper variables
    float T = 1.0f;
    uint32_t contributor = 0;
    float C[CHANNELS] = {0};

    // Iterate over batches until all done or range is complete
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
        // End if entire block votes that it is done rasterizing
        int num_done = __syncthreads_count(done);
        if (num_done == BLOCK_SIZE) { break; }

        // Collectively fetch per-Gaussian data from global to shared
        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y) {
            int coll_id = point_list[range.x + progress];
            collected_id[block.thread_rank()] = coll_id;
            collected_xy[block.thread_rank()] = points_xy_image[coll_id];
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
        }
        block.sync();

        // Iterate over current batch
        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
            // Keep track of current position in range
            contributor++;

            // Resample using conic matrix (cf. "Surface
            // Splatting" by Zwicker et al., 2001)
            float2 xy = collected_xy[j];
            float2 d = {xy.x - pixf.x, xy.y - pixf.y};
            float4 con_o = collected_conic_opacity[j];
            float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
            if (power > 0.0f) {
                continue;
            }

            // Eq. (2) from 3D Gaussian splatting paper.
            // Obtain alpha by multiplying with Gaussian opacity
            // and its exponential falloff from mean.
            // Avoid numerical instabilities (see paper appendix).
            float alpha = min(0.99f, con_o.w * exp(power));
            if (alpha < 1.0f / 255.0f) {
                continue;
            }

            float test_T = T * (1 - alpha);
            if (test_T < 0.0001f) {
                done = true;
                continue;
            }

            int global_id = collected_id[j];
            // debug image
            for (int ch = 0; ch < CHANNELS; ch++) {
                C[ch] += 0.5f * alpha * T;
            }
            float w = alpha * T;
            atomicAdd(&(out_feat[global_id * (F + 1) + 0]), w);
            for (int ch = 0; ch < F; ch++) {
                float feat = feat_img[ch * H * W + pix_id];
                atomicAdd(&(out_feat[global_id * (F + 1) + ch + 1]), w * feat);
            }
            T = test_T;
        }
    }

    // All threads that treat valid pixel write out their final
    // rendering data to the frame and auxiliary buffers.
    if (inside) {
        for (int ch = 0; ch < CHANNELS; ch++) {
            out_color[ch * H * W + pix_id] = C[ch];
        }
    }
}

template<uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
    featMarkVARCUDA(
        const uint2* __restrict__ ranges,
        const uint32_t* __restrict__ point_list,
        int F,
        int W, int H,
        const float2* __restrict__ points_xy_image,
        const float4* __restrict__ conic_opacity,
        const float* __restrict__ feat_img,
        float* __restrict__ out_feat,
        float* __restrict__ out_feat_var,
        float* __restrict__ out_color) {
    // Identify current tile and associated min/max pixel range.
    auto block = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;

    uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
    uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
    uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
    uint32_t pix_id = W * pix.y + pix.x;
    float2 pixf = {(float)pix.x, (float)pix.y};

    // Check if this thread is associated with a valid pixel or outside.
    bool inside = pix.x < W && pix.y < H;
    // Done threads can help with fetching, but don't rasterize
    bool done = !inside;

    // Load start/end range of IDs to process in bit sorted list.
    uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;

    // Allocate storage for batches of collectively fetched data.
    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];

    // Initialize helper variables
    float T = 1.0f;
    uint32_t contributor = 0;
    float C[CHANNELS] = {0};

    // Iterate over batches until all done or range is complete
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
        // End if entire block votes that it is done rasterizing
        int num_done = __syncthreads_count(done);
        if (num_done == BLOCK_SIZE) { break; }

        // Collectively fetch per-Gaussian data from global to shared
        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y) {
            int coll_id = point_list[range.x + progress];
            collected_id[block.thread_rank()] = coll_id;
            collected_xy[block.thread_rank()] = points_xy_image[coll_id];
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
        }
        block.sync();

        // Iterate over current batch
        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
            // Keep track of current position in range
            contributor++;

            // Resample using conic matrix (cf. "Surface
            // Splatting" by Zwicker et al., 2001)
            float2 xy = collected_xy[j];
            float2 d = {xy.x - pixf.x, xy.y - pixf.y};
            float4 con_o = collected_conic_opacity[j];
            float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
            if (power > 0.0f) {
                continue;
            }

            // Eq. (2) from 3D Gaussian splatting paper.
            // Obtain alpha by multiplying with Gaussian opacity
            // and its exponential falloff from mean.
            // Avoid numerical instabilities (see paper appendix).
            float alpha = min(0.99f, con_o.w * exp(power));
            if (alpha < 1.0f / 255.0f) {
                continue;
            }

            float test_T = T * (1 - alpha);
            if (test_T < 0.0001f) {
                done = true;
                continue;
            }

            int global_id = collected_id[j];
            // debug image
            for (int ch = 0; ch < CHANNELS; ch++) {
                C[ch] += 0.5f * alpha * T;
            }
            float w = alpha * T;

            atomicAdd(&(out_feat_var[global_id * (F + 1) + 0]), w);
            for (int ch = 0; ch < F; ch++) {
                float feat_sample = feat_img[ch * H * W + pix_id];
                float feat_mean = out_feat[global_id * F + ch];
                atomicAdd(&(out_feat_var[global_id * (F + 1) + ch + 1]), w * (feat_sample - feat_mean) * (feat_sample - feat_mean));
            }
            T = test_T;
        }
    }

    // All threads that treat valid pixel write out their final
    // rendering data to the frame and auxiliary buffers.
    if (inside) {
        for (int ch = 0; ch < CHANNELS; ch++) {
            out_color[ch * H * W + pix_id] = C[ch];
        }
    }
}


template<int R>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
    featMarkDebugCUDA(
        const uint2* __restrict__ ranges,
        const uint32_t* __restrict__ point_list,
        int F,
        int W, int H,
        const float2* __restrict__ points_xy_image,
        const float4* __restrict__ conic_opacity,
        const float* __restrict__ feat,
        float* __restrict__ out_feat_img) {
    // Identify current tile and associated min/max pixel range.
    auto block = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
    uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
    uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
    uint32_t pix_id = W * pix.y + pix.x;
    float2 pixf = {(float)pix.x, (float)pix.y};

    // Check if this thread is associated with a valid pixel or outside.
    bool inside = pix.x < W && pix.y < H;
    // Done threads can help with fetching, but don't rasterize
    bool done = !inside;

    // Load start/end range of IDs to process in bit sorted list.
    uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;

    // Allocate storage for batches of collectively fetched data.
    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float4 collected_conic_opacity[BLOCK_SIZE];

    // Initialize helper variables
    float T = 1.0f;
    uint32_t contributor = 0;

    // Iterate over batches until all done or range is complete
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
        // End if entire block votes that it is done rasterizing
        int num_done = __syncthreads_count(done);
        if (num_done == BLOCK_SIZE) { break; }

        // Collectively fetch per-Gaussian data from global to shared
        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y) {
            int coll_id = point_list[range.x + progress];
            collected_id[block.thread_rank()] = coll_id;
            collected_xy[block.thread_rank()] = points_xy_image[coll_id];
            collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
        }
        block.sync();

        // Iterate over current batch
        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
            // Keep track of current position in range
            contributor++;

            // Resample using conic matrix (cf. "Surface
            // Splatting" by Zwicker et al., 2001)
            float2 xy = collected_xy[j];
            float2 d = {xy.x - pixf.x, xy.y - pixf.y};
            float4 con_o = collected_conic_opacity[j];
            float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
            if (power > 0.0f) {
                continue;
            }

            // Eq. (2) from 3D Gaussian splatting paper.
            // Obtain alpha by multiplying with Gaussian opacity
            // and its exponential falloff from mean.
            // Avoid numerical instabilities (see paper appendix).
            float alpha = min(0.99f, con_o.w * exp(power));
            if (alpha < 1.0f / 255.0f) {
                continue;
            }

            float test_T = T * (1 - alpha);
            if (test_T < 0.0001f) {
                done = true;
                continue;
            }

            int global_id = collected_id[j];
            float w = alpha * T;
            for (int ch = 0; ch < F; ch++) {
                out_feat_img[ch * H * W + pix_id] += w * feat[global_id * F + ch];
            }
            T = test_T;
        }
    }

    if (!inside) {
        for (int ch = 0; ch < F; ch++) {
            out_feat_img[ch * H * W + pix_id] = 0.0f;
        }
    }
}

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
	bool prefiltered) {

	FORWARD::gs_project_CUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
		P,
		means3D,
		scales,
		rotations,
		opacities,
		viewmatrix,
		projmatrix,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		conic_opacity,
		grid,
		tiles_touched);
}


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
    float* out_color) {
    featMarkCUDA<NUM_CHANNELS><<<grid, block>>>(
        ranges,
        point_list,
        F,
        W, H,
        means2D,
        conic_opacity,
        feat_img,
        out_feat,
        out_color);
}
 
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
    float* out_color) {
    featMarkVARCUDA<NUM_CHANNELS><<<grid, block>>>(
        ranges,
        point_list,
        F,
        W, H,
        means2D,
        conic_opacity,
        feat_img,
        out_feat,
        out_feat_var,
        out_color);
}
 

void feat_mark_debug_impl(
    const dim3 grid, dim3 block,
    const uint2* ranges,
    const uint32_t* point_list,
    int F,
    int W, int H,
    const float2* means2D,
    const float4* conic_opacity,
    const float* feat,
    float* out_feat_img) {
    featMarkDebugCUDA<0><<<grid, block>>>(
        ranges,
        point_list,
        F,
        W, H,
        means2D,
        conic_opacity,
        feat,
        out_feat_img);
}
 
}// namespace sail