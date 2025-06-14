/**
 * @file featmark_point.cu
 * @brief The implementation of the feature marking algorithm.
 * @author sailing-innocent
 * @date 2025-02-18
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "config.h"

#include <fstream>
#include <string>
#include <functional>

// Core Implementation
// ----------------------
#include "featmark.h"
// ----------------------

namespace sailtorch {
std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
        return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

 
int gs_feat_mark_debug(
	const torch::Tensor& means3D,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const torch::Tensor& feat,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	torch::Tensor& out_feat_img,
	torch::Tensor& radii,
	const bool debug) {

	if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}
	const int P = means3D.size(0);
	if (feat.ndimension() != 2) {
		AT_ERROR("out_feat must have dimensions (P, F)");
	}
	if (feat.size(0) != P) {
		AT_ERROR("feat must have dimensions (P, F)");
	}
	const int F = feat.size(1);

	if (out_feat_img.ndimension() != 3) {
		AT_ERROR("feat_img must have dimensions (F, H, W)");
	}
	if (out_feat_img.size(0) != F) {
		AT_ERROR("feat_img must have dimensions (F, H, W)");
	}
	const int W = out_feat_img.size(2);
	const int H = out_feat_img.size(1);

	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto float_opts = means3D.options().dtype(torch::kFloat32);

	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor tileBuffer = torch::empty({0}, options.device(device));

	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> tileFunc = resizeFunctional(tileBuffer);

	int rendered = 0;
	if (P != 0) {
		rendered = sail::gs_feat_mark_debug(
			geomFunc,
			binningFunc,
			tileFunc,
			P, F,
			W, H,
			means3D.contiguous().data_ptr<float>(),
			opacity.contiguous().data_ptr<float>(),
			scales.contiguous().data_ptr<float>(),
			rotations.contiguous().data_ptr<float>(),
			feat.contiguous().data_ptr<float>(),
			viewmatrix.contiguous().data_ptr<float>(),
			projmatrix.contiguous().data_ptr<float>(),
			tan_fovx,
			tan_fovy,
			out_feat_img.contiguous().data_ptr<float>(),
			radii.contiguous().data_ptr<int>(),
			debug);
	}

	return rendered;
}

int gs_feat_mark(
	const torch::Tensor& means3D,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	torch::Tensor& feat_img,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	torch::Tensor& out_feat,
	torch::Tensor& out_color,
	torch::Tensor& radii,
	const bool debug) {
	if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}
	const int P = means3D.size(0);
	if (feat_img.ndimension() != 3) {
		AT_ERROR("feat_img must have dimensions (F, H, W)");
	}
	const int F = feat_img.size(0);
	const int H = feat_img.size(1);
	const int W = feat_img.size(2);

	if (out_color.size(0) == 0) {
		// not use out color
	}

	if (out_feat.ndimension() != 2) {
		AT_ERROR("out_feat must have dimensions (P, F+1)");
	}
	if (out_feat.size(0) != P || out_feat.size(1) != F + 1) {
		AT_ERROR("out_feat must have dimensions (P, F+1)");
	}

	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto float_opts = means3D.options().dtype(torch::kFloat32);

	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor tileBuffer = torch::empty({0}, options.device(device));
	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> tileFunc = resizeFunctional(tileBuffer);

	int rendered = 0;
	if (P != 0) {
		rendered = sail::gs_feat_mark(
			geomFunc,
			binningFunc,
			tileFunc,
			P, F,
			W, H,
			means3D.contiguous().data_ptr<float>(),
			opacity.contiguous().data_ptr<float>(),
			scales.contiguous().data_ptr<float>(),
			rotations.contiguous().data_ptr<float>(),
			feat_img.contiguous().data_ptr<float>(),
			viewmatrix.contiguous().data_ptr<float>(),
			projmatrix.contiguous().data_ptr<float>(),
			tan_fovx,
			tan_fovy,
			out_color.contiguous().data_ptr<float>(),
			out_feat.contiguous().data_ptr<float>(),
			radii.contiguous().data_ptr<int>(),
			debug);
	}
	return rendered;
}


int gs_feat_mark_var(
	const torch::Tensor& means3D,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	torch::Tensor& feat_img,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	torch::Tensor& out_feat,
	torch::Tensor& out_feat_var,
	torch::Tensor& out_color,
	torch::Tensor& radii,
	const bool debug) {
	if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}
	const int P = means3D.size(0);
	if (feat_img.ndimension() != 3) {
		AT_ERROR("feat_img must have dimensions (F, H, W)");
	}
	const int F = feat_img.size(0);
	const int H = feat_img.size(1);
	const int W = feat_img.size(2);

	if (out_color.size(0) == 0) {
		// not use out color
	}

	if (out_feat.ndimension() != 2) {
		AT_ERROR("out_feat must have dimensions (P, F+1)");
	}

	if (out_feat.size(0) != P || out_feat.size(1) != F) {
		AT_ERROR("out_feat must have dimensions (P, F)");
	}

	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto float_opts = means3D.options().dtype(torch::kFloat32);

	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor tileBuffer = torch::empty({0}, options.device(device));
	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> tileFunc = resizeFunctional(tileBuffer);

	int rendered = 0;
	if (P != 0) {
		rendered = sail::gs_feat_mark_var(
			geomFunc,
			binningFunc,
			tileFunc,
			P, F,
			W, H,
			means3D.contiguous().data_ptr<float>(),
			opacity.contiguous().data_ptr<float>(),
			scales.contiguous().data_ptr<float>(),
			rotations.contiguous().data_ptr<float>(),
			feat_img.contiguous().data_ptr<float>(),
			viewmatrix.contiguous().data_ptr<float>(),
			projmatrix.contiguous().data_ptr<float>(),
			tan_fovx,
			tan_fovy,
			out_color.contiguous().data_ptr<float>(),
			out_feat.contiguous().data_ptr<float>(),
			out_feat_var.contiguous().data_ptr<float>(),
			radii.contiguous().data_ptr<int>(),
			debug);
	}
	return rendered;
}

} // namespace sailtorch