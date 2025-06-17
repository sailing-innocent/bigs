#pragma once
/**
  * @file heightmap_vis.h
  * @brief Heightmap Visualization
  * @author sailing-innocent
  * @date 2025-01-14
  */
#include <SailCu/app/point_vis.h>

namespace sail::cu {

class SAIL_CU_API HeightmapVisApp : public PointVisApp {
public:
	HeightmapVisApp(std::string _title, unsigned int _resw, unsigned int _resh) : PointVisApp(_title, _resw, _resh) {};
	void bind_heightmap(const float* d_heightmap, const int H, const int W, const float* d_color = nullptr);
	// void update() override;
protected:
	int W_cached = 0;
	int H_cached = 0;
	float* _d_pos = nullptr;
	float* _d_color = nullptr;
};

// show image <H, W, C> defined in CUDA
void SAIL_CU_API heightmap_vis(
	const float* d_heightmap,// H, W
	// image size
	const int H,
	const int W,
	// display size
	const float* d_color = nullptr,// H, W, 3
	const unsigned int width = 800u,
	const unsigned int height = 600u);

}// namespace sail::cu