/**
 * @file featmark_point.h
 * @brief The FeatMark Point 
 * @author sailing-innocent
 * @date 2025-02-18
 */


#ifndef FEATMARK_POINT_H
#define FEATMARK_POINT_H
#include <torch/extension.h>

namespace sailtorch {

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
	const bool debug);

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
	const bool debug);

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
	const bool debug);

} // namespace sailtorch

#endif //FEATMARK_POINT_H