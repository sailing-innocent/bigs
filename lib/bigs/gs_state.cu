/**
 * @file state.cu
 * @brief GS State impl
 * @author sailing-innocent
 * @date 2024-12-08
 */


#include "gs_state.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
 
namespace sail {
 
GeometryState GeometryState::fromChunk(char*& chunk, size_t P) {
    GeometryState geom;
    obtain(chunk, geom.depths, P, 128);
    obtain(chunk, geom.clamped, P * 3, 128);
    obtain(chunk, geom.internal_radii, P, 128);
    obtain(chunk, geom.means2D, P, 128);
    obtain(chunk, geom.cov3D, P * 6, 128);
    obtain(chunk, geom.conic_opacity, P, 128);
    obtain(chunk, geom.rgb, P * 3, 128);
    obtain(chunk, geom.tiles_touched, P, 128);
    cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
    obtain(chunk, geom.scanning_space, geom.scan_size, 128);
    obtain(chunk, geom.point_offsets, P, 128);
    return geom;
}

TileState TileState::fromChunk(char*& chunk, size_t N) {
	TileState tile;
	obtain(chunk, tile.ranges, N, 128);
	return tile;
}

BinningState BinningState::fromChunk(char*& chunk, size_t P) {
    BinningState binning;
    obtain(chunk, binning.point_list, P, 128);
    obtain(chunk, binning.point_list_unsorted, P, 128);
    obtain(chunk, binning.point_list_keys, P, 128);
    obtain(chunk, binning.point_list_keys_unsorted, P, 128);
    cub::DeviceRadixSort::SortPairs(
        nullptr, binning.sorting_size,
        binning.point_list_keys_unsorted, binning.point_list_keys,
        binning.point_list_unsorted, binning.point_list, P);
    obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
    return binning;
}
 }// namespace sail
 