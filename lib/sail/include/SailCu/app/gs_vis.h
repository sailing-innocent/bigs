#pragma once
/**
 * @file gs_vis.h
 * @brief Gaussian Visualizer
 * @author sailing-innocent
 * @date 2025-01-01
 */
#include "GLFW/glfw3.h"
#include "SailCu/config.h"
#include <span>
#include <SailCu/app/point_vis.h>

namespace sail::cu {

class SAIL_CU_API GSVisApp : public PointVisApp {
public:
	GSVisApp(std::string _title, unsigned int _resw, unsigned int _resh) : PointVisApp(_title, _resw, _resh) {};
	void gen_data(const int num_points) override;
	void bind_data(
		const float* d_pos,
		const float* d_color,
		const int num_points) override;

	// apply transforms for cube vertices according to scale and rotation quaternion
	void apply_transform(
		const float* d_pos,
		const float* d_scale,
		const float* d_rotq,
		const int num_points);

	void update() override;
	void terminate() override;

protected:
	void init_shaders() override;
	GLuint cube_ebo;
	struct cudaGraphicsResource* d_cube_ebo_resource;
};

void SAIL_CU_API gs_vis(
	const float* d_pos,
	const float* d_color,
	const float* d_scale,
	const float* d_rotq,
	const int num_points,
	std::span<float> debug_lines,
	const unsigned int width = 800u,
	const unsigned int height = 800u);

}// namespace sail::cu