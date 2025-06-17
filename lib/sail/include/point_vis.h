#pragma once
/**
 * @file point_vis.h
 * @brief Point Visualization App
 * @author sailing-innocent
 * @date 2024-12-09
 */

#include "GLFW/glfw3.h"
#include "SailCu/config.h"
#include "SailCu/util/shader/program.h"
#include <memory>
// INTEROP
#include <cuda_gl_interop.h>
#include <span>
#include <array>

namespace sail {

class SAIL_API PointVisApp {
public:
	PointVisApp(std::string _title, unsigned int _resw, unsigned int _resh, bool cam_flip_y = true);
	void init();
	void before_run();
	virtual void update();
	void update_gui();
	bool should_close();
	virtual void terminate();
	void set_point_size(float point_size) { m_point_size = point_size; }
	void debug_lines(std::span<float> debug_lines);

	virtual void gen_data(const int num_points);
	virtual void bind_data(
		const float* d_pos,
		const float* d_color,
		const int num_points);

protected:
	void init_window(unsigned int resw, unsigned int resh);
	virtual void init_shaders();
	void create_vbo(GLuint* vbo, struct cudaGraphicsResource** vbo_res, unsigned int vbo_res_flags);
	void delete_vbo(GLuint* vbo, struct cudaGraphicsResource* vbo_res);

	std::unique_ptr<sail::cu::ShaderProgram> mp_shader_program;
	GLuint VAO, pos_vbo, color_vbo;
	GLFWwindow* m_window;
	std::string m_title;
	unsigned int m_resw, m_resh;
	struct cudaGraphicsResource* d_pos_vbo_resource;
	struct cudaGraphicsResource* d_color_vbo_resource;
	int m_num_points = 0;
	unsigned int m_lines_vao;
	int m_num_lines = 0;
	float m_point_size = 10.0f;
	float m_line_width = 2.0f;
	std::array<float, 3> m_bg_color = {.1f, .2f, .3f};
	bool m_cam_flip_y = false;
};

SAIL_CU_API void point_vis(
	const float* d_pos,
	const float* d_color,
	const int num_points,
	std::span<float> debug_lines,
	float point_size = 10.0,
	const unsigned int width = 800u,
	const unsigned int height = 600u);

}// namespace sail::cu
