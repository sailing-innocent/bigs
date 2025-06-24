/**
 * @file gs_vis.cpp
 * @brief Gaussian Visualizer Impl
 * @author sailing-innocent
 * @date 2025-01-01
 */

#include "SailCu/app/gs_vis.h"
#include "SailCu/app/point_vis.h"
#include "SailCu/util/gui.h"
#include "SailCu/gs/vis.h"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

namespace sail {

void GSVisApp::init_shaders() {
	std::string name = "point_cuda";
	std::string vert_source = "#version 450 core\nlayout(location = 0) in vec3 aPos;\nlayout(location = 1) in vec3 aColor;\nuniform mat4 view;\nuniform mat4 projection;\nout vec4 pColor;\nvoid main() {\ngl_Position = projection * view * vec4(aPos, 1.0);\npColor = vec4(aColor,1.0);\n}";
	std::string frag_source = "#version 450 core\nin vec4 pColor;\nout vec4 FragColor;\nvoid main() {\nFragColor = pColor;\n}";

	mp_shader_program = std::make_unique<ShaderProgram>(name);
	mp_shader_program->attach_shader(ShaderBase::from_source(vert_source.c_str(), sail::ShaderType(GL_VERTEX_SHADER, "VERTEX"), "point_vert"));
	mp_shader_program->attach_shader(ShaderBase::from_source(frag_source.c_str(), ShaderType(GL_FRAGMENT_SHADER, "FRAGMENT"), "point_frag"));
	mp_shader_program->link_program();
}

void GSVisApp::apply_transform(
	const float* d_pos,
	const float* d_scale,
	const float* d_rotq,
	const int num_points) {

	size_t sz = num_points * 3 * sizeof(float) * 8;
	cudaGraphicsMapResources(1, &d_pos_vbo_resource, 0);
	float* d_pos_vbo_ptr;
	cudaGraphicsResourceGetMappedPointer((void**)&d_pos_vbo_ptr, &sz, d_pos_vbo_resource);

	point_bb_verts_transform(d_scale, d_rotq, d_pos, d_pos_vbo_ptr, num_points);
}
void GSVisApp::gen_data(const int num_points) {
	m_num_points = num_points;
	size_t sz = num_points * 3 * sizeof(float) * 8;
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

	// CREATE and bind the cube_ebo
	glGenBuffers(1, &cube_ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cube_ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 36 * num_points * sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW);
	cudaGraphicsGLRegisterBuffer(&d_cube_ebo_resource, cube_ebo, cudaGraphicsRegisterFlagsWriteDiscard);
}

void GSVisApp::bind_data(
	const float* d_pos,
	const float* d_color,
	const int num_points) {
	glBindVertexArray(VAO);
	size_t sz = num_points * 3 * sizeof(float) * 8;
	// Map the position VBO and cube face EBO for CUDA access
	cudaGraphicsMapResources(1, &d_pos_vbo_resource, 0);
	float* d_pos_vbo_ptr;
	cudaGraphicsResourceGetMappedPointer((void**)&d_pos_vbo_ptr, &sz, d_pos_vbo_resource);
	cudaGraphicsMapResources(1, &d_cube_ebo_resource, 0);
	int* d_cube_ebo_ptr;
	cudaGraphicsResourceGetMappedPointer((void**)&d_cube_ebo_ptr, &sz, d_cube_ebo_resource);

	point_to_bb_verts(d_pos, d_pos_vbo_ptr, d_cube_ebo_ptr, num_points);

	cudaGraphicsUnmapResources(1, &d_pos_vbo_resource, 0);
	cudaGraphicsUnmapResources(1, &d_cube_ebo_resource, 0);

	// Map the color VBO for CUDA access
	cudaGraphicsMapResources(1, &d_color_vbo_resource, 0);
	float* d_color_vbo_ptr;
	cudaGraphicsResourceGetMappedPointer((void**)&d_color_vbo_ptr, &sz, d_color_vbo_resource);

	// cudaMemcpy(d_color_vbo_ptr, d_color, sz, cudaMemcpyDeviceToDevice);
	expand_color(d_color, d_color_vbo_ptr, num_points);

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

void GSVisApp::update() {
	auto& gui = sail::GUI::instance();
	gui.process_input_callback(m_window);
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// setting point size
	glPointSize(10.0f);
	mp_shader_program->use();
	mp_shader_program->set_mat4("view", gui.camera->get_view_matrix());
	mp_shader_program->set_mat4("projection", gui.camera->get_projection_matrix());

	// point
	glBindVertexArray(VAO);
	glDrawElements(GL_TRIANGLES, m_num_points * 36, GL_UNSIGNED_INT, 0);
	// glDrawArrays(GL_POINTS, 0, m_num_points * 8);

	// camera debug lines
	glBindVertexArray(m_lines_vao);
	glDrawArrays(GL_LINES, 0, m_num_lines * 2);
	glBindVertexArray(0);

	update_gui();

	glfwSwapBuffers(m_window);
	glfwPollEvents();
}

void GSVisApp::terminate() {
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	cudaGraphicsUnregisterResource(d_pos_vbo_resource);
	cudaGraphicsUnregisterResource(d_color_vbo_resource);
	cudaGraphicsUnregisterResource(d_cube_ebo_resource);
	glDeleteBuffers(1, &pos_vbo);
	glDeleteBuffers(1, &color_vbo);
	glDeleteBuffers(1, &cube_ebo);
	glDeleteVertexArrays(1, &VAO);
	glfwTerminate();
}

}// namespace sail
