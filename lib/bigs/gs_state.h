/**
 * @file state.h
 * @brief General Gaussian State
 * @author sailing-innocent
 * @date 2024-12-08
 */

#ifndef FEATMARK_GS_STATE_H
#define FEATMARK_GS_STATE_H

#include <cuda_runtime_api.h>

namespace sail {

template<typename T>
static void obtain(char*& chunk, T*& ptr, size_t count, size_t alignment) {
    std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
    ptr = reinterpret_cast<T*>(offset);
    chunk = reinterpret_cast<char*>(ptr + count);
}

template<typename T>
size_t required(size_t P) {
    char* size = nullptr;
    T::fromChunk(size, P);
    return ((size_t)size) + 128;
}

struct GeometryState {
	size_t scan_size;
	float* depths;
	char* scanning_space;
	bool* clamped;
	int* internal_radii;
	float2* means2D;
	float* cov3D;
	float4* conic_opacity;
	float* rgb;
	uint32_t* point_offsets;
	uint32_t* tiles_touched;
	static GeometryState fromChunk(char*& chunk, size_t P);
};

// tile.x * tile.y
struct TileState {
	uint2* ranges;
	static TileState fromChunk(char*& chunk, size_t N);
};

struct BinningState {
	size_t sorting_size;
	uint64_t* point_list_keys_unsorted;
	uint64_t* point_list_keys;
	uint32_t* point_list_unsorted;
	uint32_t* point_list;
	char* list_sorting_space;

	static BinningState fromChunk(char*& chunk, size_t P);
};


}// namespace sail

#endif //FEATMARK_GS_STATE_H