/**
 * @file sailc.cpp
 * @brief The SailC module for point visualization.
 * @author sailing-innocent
 * @date 2025-06-17
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <iostream>
#include <vector>
#include "SailCu/app/point_vis.h"
#include "SailCu/app/gs_vis.h"

namespace py = pybind11;

void point_vis(
	const int64_t d_pos,
	const int64_t d_color,
	const int num_points,
	std::vector<float> debug_lines,
	float point_size = 10.0,
	const unsigned int width = 800u,
	const unsigned int height = 600u,
	const int d_pos_stride = 3,
	const int d_color_stride = 3) {

	float* d_pos_f = reinterpret_cast<float*>(d_pos);
	float* d_color_f = reinterpret_cast<float*>(d_color);
	std::cout << "point_vis called with " << num_points << " points." << std::endl;
	// std::cout << *(d_pos_f + d_pos_stride * 1 + 1) << std::endl;
	// process lines
	sail::PointVisApp app{
		"Point Visualization",
		width,
		height,
		true};

	app.init();
	app.before_run();
	if (point_size > 0) {
		app.set_point_size(point_size);
	}
	app.debug_lines(debug_lines);
	app.gen_data(num_points);
	app.bind_data(d_pos_f, d_color_f, num_points);
	while (!app.should_close()) {
		app.update();
	}
	app.terminate();
}

void gs_vis(
	const float* d_pos,
	const float* d_color,
	const float* d_scale,
	const float* d_rotq,
	const int num_points,
	std::span<float> debug_lines,
	const unsigned int width = 800u,
	const unsigned int height = 600u) {

	// process lines
	sail::GSVisApp app{
		"GS Visualization",
		width,
		height,
	};

	app.init();

	app.before_run();
	app.gen_data(num_points);
	app.debug_lines(debug_lines);

	app.bind_data(d_pos, d_color, num_points);
	app.apply_transform(d_pos, d_scale, d_rotq, num_points);
	while (!app.should_close()) {
		app.update();
	}
	app.terminate();
}

int add(int i, int j) {
	return i + j;
}

PYBIND11_MODULE(sailc, m) {
	m.doc() = "pybind11 example plugin";// optional module docstring
	m.def("add", &add, "A function that adds two numbers");
	m.def("point_vis", &point_vis, "Point Visualization");
	m.def("gs_vis", &gs_vis, "Gaussian Visualization");
}