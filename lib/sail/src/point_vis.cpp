#pragma once
/**
 * @file point_vis.cu
 * @brief Point List Visualization Impl
 * @author sailing-innocent
 * @date 2024-12-09
 */

#include <glad/glad.h>
#include <glfw/glfw3.h>
#include "SailCu/app/point_vis.h"
#include "SailCu/util/gui.h"
#include <glm/glm.hpp>
#include <iostream>

#include <span>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

namespace sail {

PointVisApp ::PointVisApp(std::string _title, unsigned int _resw, unsigned int _resh, bool cam_flip_y) : m_title(_title), m_resw(_resw), m_resh(_resh), m_cam_flip_y(cam_flip_y) {};

void PointVisApp::init() {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	init_window(m_resw, m_resh);
	// bind gui
	auto& gui = sail::GUI::instance(m_resw, m_resh, m_cam_flip_y);
	gui.bind_window(m_window);

	// initialize imgui
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	(void)io;
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(m_window, true);
	ImGui_ImplOpenGL3_Init("#version 450");
}

void PointVisApp::before_run() {
	init_shaders();
}

bool PointVisApp::should_close() {
	return glfwWindowShouldClose(m_window);
}

void PointVisApp::terminate() {
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	delete_vbo(&pos_vbo, d_pos_vbo_resource);
	delete_vbo(&color_vbo, d_color_vbo_resource);
	glfwTerminate();
}

void PointVisApp::init_window(unsigned int resw, unsigned int resh) {
	m_window = glfwCreateWindow(resw, resh, m_title.c_str(), NULL, NULL);
	if (m_window == nullptr) {
		glfwTerminate();
	}
	glfwMakeContextCurrent(m_window);
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cout << "Failed to initialize GLAD" << std::endl;
	}
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
}

void PointVisApp::init_shaders() {
	std::string name = "point_cuda";
	std::string vert_source = "#version 450 core\nlayout(location = 0) in vec3 aPos;\nlayout(location = 1) in vec3 aColor;\nuniform mat4 view;\nuniform mat4 projection;\nout vec4 pColor;\nvoid main() {\ngl_Position = projection * view * vec4(aPos, 1.0);\npColor = vec4(aColor,1.0);\n}";
	std::string frag_source = "#version 450 core\nin vec4 pColor;\nout vec4 FragColor;\nvoid main() {\nFragColor = pColor;\n}";

	mp_shader_program = std::make_unique<ShaderProgram>(name);
	mp_shader_program->attach_shader(ShaderBase::from_source(vert_source.c_str(), ShaderType(GL_VERTEX_SHADER, "VERTEX"), "point_vert"));
	mp_shader_program->attach_shader(ShaderBase::from_source(frag_source.c_str(), ShaderType(GL_FRAGMENT_SHADER, "FRAGMENT"), "point_frag"));
	mp_shader_program->link_program();
}

void PointVisApp::create_vbo(GLuint* vbo, struct cudaGraphicsResource** vbo_res, unsigned int vbo_res_flags) {
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	glBufferData(GL_ARRAY_BUFFER, 3 * 3 * sizeof(float), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags);
}

void PointVisApp::delete_vbo(GLuint* vbo, struct cudaGraphicsResource* vbo_res) {
	cudaGraphicsUnregisterResource(vbo_res);
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);
	*vbo = 0;
}

void PointVisApp::debug_lines(std::span<float> debug_lines) {
	int N = debug_lines.size();
	m_num_lines = N / 6;
	std::vector<float> color;
	for (int i = 0; i < m_num_lines; i++) {
		color.push_back(1.0f);
		color.push_back(0.0f);
		color.push_back(0.0f);
		color.push_back(1.0f);
		color.push_back(0.0f);
		color.push_back(0.0f);
	}
	// Create and bind the VAO
	glGenVertexArrays(1, &m_lines_vao);
	glBindVertexArray(m_lines_vao);
	// Create and bind the position VBO
	GLuint lines_vbo;
	glGenBuffers(1, &lines_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, lines_vbo);
	glBufferData(GL_ARRAY_BUFFER, N * sizeof(float), debug_lines.data(), GL_STATIC_DRAW);
	// Set up vertex attributes
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	GLuint line_color_vbo;
	glGenBuffers(1, &line_color_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, line_color_vbo);
	glBufferData(GL_ARRAY_BUFFER, N * sizeof(float), color.data(), GL_STATIC_DRAW);

	// Set up vertex attributes
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);
	// Unbind the VAO
	glBindVertexArray(0);
}

void PointVisApp::gen_data(const int num_points) {
	m_num_points = num_points;
	size_t sz = num_points * 3 * sizeof(float);
	// Create and bind the VAO
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
	// Create and bind the position VBO
	glGenBuffers(1, &pos_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);
	glBufferData(GL_ARRAY_BUFFER, sz, nullptr, GL_DYNAMIC_DRAW);
	cudaGraphicsGLRegisterBuffer(&d_pos_vbo_resource, pos_vbo, cudaGraphicsRegisterFlagsWriteDiscard);
	// Create and bind the color VBO
	glGenBuffers(1, &color_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, color_vbo);
	glBufferData(GL_ARRAY_BUFFER, sz, nullptr, GL_DYNAMIC_DRAW);
	cudaGraphicsGLRegisterBuffer(&d_color_vbo_resource, color_vbo, cudaGraphicsRegisterFlagsWriteDiscard);
}

void PointVisApp::bind_data(
	const float* d_pos,
	const float* d_color,
	const int num_points) {
	// bind VAO
	glBindVertexArray(VAO);
	// bind cudaGraphicsResource
	size_t sz = num_points * 3 * sizeof(float);
	// Map the position VBO for CUDA access
	cudaGraphicsMapResources(1, &d_pos_vbo_resource, 0);
	float* d_pos_vbo_ptr;
	cudaGraphicsResourceGetMappedPointer((void**)&d_pos_vbo_ptr, &sz, d_pos_vbo_resource);
	cudaMemcpy(d_pos_vbo_ptr, d_pos, sz, cudaMemcpyDeviceToDevice);
	cudaGraphicsUnmapResources(1, &d_pos_vbo_resource, 0);

	// Map the color VBO for CUDA access
	cudaGraphicsMapResources(1, &d_color_vbo_resource, 0);
	float* d_color_vbo_ptr;
	cudaGraphicsResourceGetMappedPointer((void**)&d_color_vbo_ptr, &sz, d_color_vbo_resource);
	cudaMemcpy(d_color_vbo_ptr, d_color, sz, cudaMemcpyDeviceToDevice);
	cudaGraphicsUnmapResources(1, &d_color_vbo_resource, 0);

	// Set up vertex attributes
	glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, color_vbo);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);
	// Unbind the VAO
	glBindVertexArray(0);
}

void PointVisApp::update_gui() {
	auto& gui = GUI::instance();
	// Start the ImGui frame
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	// Create a window to show camera position and direction
	ImGui::Begin("Camera Info");
	auto cam_pos = gui.camera->get_position();
	auto cam_dir = gui.camera->get_direction();
	ImGui::Text("Camera Position: (%.2f, %.2f, %.2f)", cam_pos.x, cam_pos.y, cam_pos.z);
	ImGui::Text("Camera Direction: (%.2f, %.2f, %.2f)", cam_dir.x, cam_dir.y, cam_dir.z);

	static float camera_position[3] = {0.0f, 0.0f, 0.0f};
	static float camera_direction[3] = {0.0f, 0.0f, 0.0f};

	ImGui::InputFloat3("Position", camera_position);
	ImGui::InputFloat3("Direction", camera_direction);
	ImGui::SliderFloat("Point Size", &m_point_size, 1.0f, 20.0f);
	ImGui::SliderFloat("Line Width", &m_line_width, 1.0f, 20.0f);
	ImGui::ColorEdit3("Background Color", m_bg_color.data());

	if (ImGui::Button("Update Camera")) {
		gui.camera->set_position(glm::vec3(camera_position[0], camera_position[1], camera_position[2]));
		gui.camera->set_direction(glm::vec3(camera_direction[0], camera_direction[1], camera_direction[2]));
	}
	ImGui::End();
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void PointVisApp::update() {
	auto& gui = sail::GUI::instance();
	gui.process_input_callback(m_window);
	// glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClearColor(m_bg_color[0], m_bg_color[1], m_bg_color[2], 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// setting point size
	glPointSize(m_point_size);
	glLineWidth(m_line_width);
	mp_shader_program->use();
	mp_shader_program->set_mat4("view", gui.camera->get_view_matrix());
	mp_shader_program->set_mat4("projection", gui.camera->get_projection_matrix());
	glBindVertexArray(VAO);
	glDrawArrays(GL_POINTS, 0, m_num_points);
	glBindVertexArray(m_lines_vao);
	glDrawArrays(GL_LINES, 0, m_num_lines * 2);
	glBindVertexArray(0);

	update_gui();

	glfwSwapBuffers(m_window);
	glfwPollEvents();
}

void point_vis(
	const float* d_pos,
	const float* d_color,
	const int num_points,
	std::span<float> debug_lines,
	float point_size,
	const unsigned int width,
	const unsigned int height) {

	// process lines
	PointVisApp app{
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
	app.bind_data(d_pos, d_color, num_points);
	while (!app.should_close()) {
		app.update();
	}
	app.terminate();
}

}// namespace sail