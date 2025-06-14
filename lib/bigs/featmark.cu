/**
 * @file feat_mark.cu
 * @brief The Feature Marker Implementation
 * @author sailing-innocent
 * @date 2024-12-08
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cooperative_groups.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
 
#include "featmark.h"
#include "gs_state.h"
#include "featmark_impl.h"
#include "auxiliary.cuh"

namespace sail {

namespace cg = cooperative_groups;

// Helper function to find the next-highest bit of the MSB on the CPU.
inline uint32_t getHigherMsb(uint32_t n) {
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1) {
		step /= 2;
		if (n >> msb) {
			msb += step;
		} else {
			msb -= step;
		}
	}
	if (n >> msb) {
		msb++;
	}
	return msb;
}


// Generates one key/value pair for all Gaussian / tile overlaps.
// Run once per Gaussian (1:N mapping).
template<int R>
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid) {

	auto idx = cg::this_grid().thread_rank();
	if (idx >= P) { return; }

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0) {
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth.
		for (int y = rect_min.y; y < rect_max.y; y++) {
			for (int x = rect_min.x; x < rect_max.x; x++) {
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in
// the full sorted list. If yes, write start/end of this tile.
// Run once per instanced (duplicated) Gaussian ID.
template<int R>
__global__ void identifyTileRanges(int L, const uint64_t* point_list_keys, uint2* ranges) {
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L) { return; }
	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0) {
		ranges[currtile].x = 0;
	} else {
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile) {
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1) {
		ranges[currtile].y = L;
	}
}

// feat mark with debug info
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
    bool debug) {

    const float focal_y = height / (2.0f * tan_fovy);
    const float focal_x = width / (2.0f * tan_fovx);
    size_t chunk_size = required<GeometryState>(P);
    char* chunkptr = geometryBuffer(chunk_size);
    GeometryState geomState = GeometryState::fromChunk(chunkptr, P);
    dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    auto tsz = tile_grid.x * tile_grid.y;
    size_t tchunk_size = required<TileState>(tsz);
    char* tchunkptr = tileBuffer(tchunk_size);
    TileState tileState = TileState::fromChunk(tchunkptr, tsz);

    CHECK_CUDA(gs_project(
                P,
                means3D,
                (glm::vec3*)scales,
                (glm::vec4*)rotations,
                opacities,
                viewmatrix, projmatrix,
                width, height,
                focal_x, focal_y,
                tan_fovx, tan_fovy,
                radii,
                geomState.means2D,
                geomState.depths,
                geomState.cov3D,
                geomState.conic_opacity,
                tile_grid,
                geomState.tiles_touched,
                false),
            debug)
    // Compute prefix sum over full list of touched tile counts by Gaussians
    // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

    // Retrieve total number of Gaussian instances to launch and resize aux buffers
    int num_rendered;
    CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

    size_t binning_chunk_size = required<BinningState>(num_rendered);
    char* binning_chunkptr = binningBuffer(binning_chunk_size);
    BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

    // For each instance to be rendered, produce adequate [ tile | depth ] key
    // and corresponding dublicated Gaussian indices to be sorted
    duplicateWithKeys<0><<<(P + 255) / 256, 256>>>(
        P,
        geomState.means2D,
        geomState.depths,
        geomState.point_offsets,
        binningState.point_list_keys_unsorted,
        binningState.point_list_unsorted,
        radii,
        tile_grid)
        CHECK_CUDA(, debug)

    int bit = getHigherMsb(tile_grid.x * tile_grid.y);

    // Sort complete list of (duplicated) Gaussian indices by keys
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
                    binningState.list_sorting_space,
                    binningState.sorting_size,
                    binningState.point_list_keys_unsorted, binningState.point_list_keys,
                    binningState.point_list_unsorted, binningState.point_list,
                    num_rendered, 0, 32 + bit),
                debug);

    CHECK_CUDA(cudaMemset(tileState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

    // Identify start and end of per-tile workloads in sorted list
    // Identify start and end of per-tile workloads in sorted list
    if (num_rendered > 0) {
        identifyTileRanges<0><<<(num_rendered + 255) / 256, 256>>>(
            num_rendered,
            binningState.point_list_keys,
            tileState.ranges);
    }
    CHECK_CUDA(, debug);
    CHECK_CUDA(feat_mark_debug_impl(
                tile_grid, block,
                tileState.ranges,
                binningState.point_list,
                F,
                width,
                height,
                geomState.means2D,
                geomState.conic_opacity,
                feat,
                out_feat_img),
            debug)
    return num_rendered;
}

int gs_feat_mark(
    std::function<char*(size_t)> geometryBuffer,
    std::function<char*(size_t)> binningBuffer,
    std::function<char*(size_t)> tileBuffer,
    const int P, int F,
    const int width, int height,
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
    bool debug) {

    const float focal_y = height / (2.0f * tan_fovy);
    const float focal_x = width / (2.0f * tan_fovx);
    size_t chunk_size = required<GeometryState>(P);
    char* chunkptr = geometryBuffer(chunk_size);
    GeometryState geomState = GeometryState::fromChunk(chunkptr, P);
    dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    auto tsz = tile_grid.x * tile_grid.y;
    size_t tchunk_size = required<TileState>(tsz);
    char* tchunkptr = tileBuffer(tchunk_size);
    TileState tileState = TileState::fromChunk(tchunkptr, tsz);

    CHECK_CUDA(
        gs_project(
            P,
            means3D,
            (glm::vec3*)scales,
            (glm::vec4*)rotations,
            opacities,
            viewmatrix, projmatrix,
            width, height,
            focal_x, focal_y,
            tan_fovx, tan_fovy,
            radii,
            geomState.means2D,
            geomState.depths,
            geomState.cov3D,
            geomState.conic_opacity,
            tile_grid,
            geomState.tiles_touched,
            false),
        debug)

    // Compute prefix sum over full list of touched tile counts by Gaussians
    // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

    // Retrieve total number of Gaussian instances to launch and resize aux buffers
    int num_rendered;
    CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

    size_t binning_chunk_size = required<BinningState>(num_rendered);
    char* binning_chunkptr = binningBuffer(binning_chunk_size);
    BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

    // For each instance to be rendered, produce adequate [ tile | depth ] key
    // and corresponding dublicated Gaussian indices to be sorted
    duplicateWithKeys<0><<<(P + 255) / 256, 256>>>(
        P,
        geomState.means2D,
        geomState.depths,
        geomState.point_offsets,
        binningState.point_list_keys_unsorted,
        binningState.point_list_unsorted,
        radii,
        tile_grid)
        CHECK_CUDA(, debug);
    int bit = getHigherMsb(tile_grid.x * tile_grid.y);

    // Sort complete list of (duplicated) Gaussian indices by keys
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
                binningState.list_sorting_space,
                binningState.sorting_size,
                binningState.point_list_keys_unsorted, binningState.point_list_keys,
                binningState.point_list_unsorted, binningState.point_list,
                num_rendered, 0, 32 + bit),
            debug);

    CHECK_CUDA(cudaMemset(tileState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

    // Identify start and end of per-tile workloads in sorted list
    if (num_rendered > 0) {
        identifyTileRanges<0><<<(num_rendered + 255) / 256, 256>>>(
            num_rendered,
            binningState.point_list_keys,
            tileState.ranges);
    }
    CHECK_CUDA(, debug);
    CHECK_CUDA(feat_mark_impl(
                tile_grid, block,
                tileState.ranges,
                binningState.point_list,
                F,
                width,
                height,
                geomState.means2D,
                geomState.conic_opacity,
                feat_img,
                out_feat,
                out_color),
            debug);
    return num_rendered;
}

int gs_feat_mark_var(
    std::function<char*(size_t)> geometryBuffer,
    std::function<char*(size_t)> binningBuffer,
    std::function<char*(size_t)> tileBuffer,
    const int P, int F,
    const int width, int height,
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
    bool debug) {

    const float focal_y = height / (2.0f * tan_fovy);
    const float focal_x = width / (2.0f * tan_fovx);
    size_t chunk_size = required<GeometryState>(P);
    char* chunkptr = geometryBuffer(chunk_size);
    GeometryState geomState = GeometryState::fromChunk(chunkptr, P);
    dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    auto tsz = tile_grid.x * tile_grid.y;
    size_t tchunk_size = required<TileState>(tsz);
    char* tchunkptr = tileBuffer(tchunk_size);
    TileState tileState = TileState::fromChunk(tchunkptr, tsz);

    CHECK_CUDA(
        gs_project(
            P,
            means3D,
            (glm::vec3*)scales,
            (glm::vec4*)rotations,
            opacities,
            viewmatrix, projmatrix,
            width, height,
            focal_x, focal_y,
            tan_fovx, tan_fovy,
            radii,
            geomState.means2D,
            geomState.depths,
            geomState.cov3D,
            geomState.conic_opacity,
            tile_grid,
            geomState.tiles_touched,
            false),
        debug)

    // Compute prefix sum over full list of touched tile counts by Gaussians
    // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

    // Retrieve total number of Gaussian instances to launch and resize aux buffers
    int num_rendered;
    CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

    size_t binning_chunk_size = required<BinningState>(num_rendered);
    char* binning_chunkptr = binningBuffer(binning_chunk_size);
    BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

    // For each instance to be rendered, produce adequate [ tile | depth ] key
    // and corresponding dublicated Gaussian indices to be sorted
    duplicateWithKeys<0><<<(P + 255) / 256, 256>>>(
        P,
        geomState.means2D,
        geomState.depths,
        geomState.point_offsets,
        binningState.point_list_keys_unsorted,
        binningState.point_list_unsorted,
        radii,
        tile_grid)
        CHECK_CUDA(, debug);
    int bit = getHigherMsb(tile_grid.x * tile_grid.y);

    // Sort complete list of (duplicated) Gaussian indices by keys
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
                binningState.list_sorting_space,
                binningState.sorting_size,
                binningState.point_list_keys_unsorted, binningState.point_list_keys,
                binningState.point_list_unsorted, binningState.point_list,
                num_rendered, 0, 32 + bit),
            debug);

    CHECK_CUDA(cudaMemset(tileState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

    // Identify start and end of per-tile workloads in sorted list
    if (num_rendered > 0) {
        identifyTileRanges<0><<<(num_rendered + 255) / 256, 256>>>(
            num_rendered,
            binningState.point_list_keys,
            tileState.ranges);
    }
    CHECK_CUDA(, debug);
    CHECK_CUDA(feat_mark_var_impl(
                tile_grid, block,
                tileState.ranges,
                binningState.point_list,
                F,
                width,
                height,
                geomState.means2D,
                geomState.conic_opacity,
                feat_img,
                out_feat,
                out_feat_var,
                out_color),
            debug);
    return num_rendered;
}

} // namespace sail

