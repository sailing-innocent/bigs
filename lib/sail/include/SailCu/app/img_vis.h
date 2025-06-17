#pragma once
/**
  * @file img_vis.h
  * @brief Show Image Defined in CUDA
  * @author sailing-innocent
  * @date 2025-01-13
  */
#include <SailCu/app/point_vis.h>

namespace sail::cu {

class SAIL_CU_API ImgVisApp : public PointVisApp {
public:
	ImgVisApp(std::string _title, unsigned int _resw, unsigned int _resh) : PointVisApp(_title, _resw, _resh) {};
	void bind_img(const float* d_img, const int H, const int W, const int C);
	void update() override;

protected:
	int W_cached = 0;
	int H_cached = 0;
	float* d_pos = nullptr;
};

// show image <H, W, C> defined in CUDA
void SAIL_CU_API img_vis(
	const float* d_img,
	// image size
	const int H,
	const int W,
	const int C,
	// display size
	const unsigned int width = 800u,
	const unsigned int height = 600u);

}// namespace sail::cu