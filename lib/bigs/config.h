/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 16


#define SAIL_HOST_DEVICE __host__ __device__

#define CHECK_CUDA(A, debug)                                                                                           \
	A;                                                                                                                 \
	if (debug) {                                                                                                       \
		auto ret = cudaDeviceSynchronize();                                                                            \
		if (ret != cudaSuccess) {                                                                                      \
			std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
			throw std::runtime_error(cudaGetErrorString(ret));                                                         \
		}                                                                                                              \
	}


#endif