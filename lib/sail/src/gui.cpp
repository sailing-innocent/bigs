/**
 * @file gui.cpp
 * @brief The OpenGL GUI Singleton Implementation
 * @author sailing-innocent
 * @date 2024-11-29
 */

#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "SailCu/util/gui.h"
// #include <stb_image_write.h>
#include <iostream>
#include <memory>

namespace sail {

ScreenState::ScreenState(int _w, int _h) {
	update(_w, _h);
}

void ScreenState::update(int _w, int _h) {
	width = _w;
	height = _h;
	aspect_ratio = static_cast<float>(width) / static_cast<float>(height);
}

}// namespace sail

namespace sail {

GUI::GUI(int width, int height, bool flipy) {
	control_state = std::make_unique<ControlState>();
	screen_state = std::make_unique<ScreenState>(width, height);
	camera = std::make_unique<GLMCamera>();
	if (flipy) {
		camera->set_coord_type(CameraCoordType::kFlipY);
	}
	camera->set_aspect(screen_state->aspect_ratio);
	camera->set_direction(sail::radians(control_state->pitch), sail::radians(control_state->yaw));
};

GUI::~GUI() {
};

void GUI::bind_window(GLFWwindow* window) {
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetKeyCallback(window, key_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetCursorPosCallback(window, cursor_pos_callback);
	glfwSetScrollCallback(window, scroll_callback);
}

void GUI::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	// if 0 pressed, save the current frame
	if (key == GLFW_KEY_F12 && action == GLFW_PRESS) {
		auto& gui = GUI::instance();
		unsigned char* data = new unsigned char[gui.screen_state->width * gui.screen_state->height * 3];
		// bind default framebuffer
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glReadPixels(0, 0, gui.screen_state->width, gui.screen_state->height, GL_RGB, GL_UNSIGNED_BYTE, data);
		// stbi_flip_vertically_on_write(true);
		// stbi_write_png("screenshot.png", gui.screen_state->width, gui.screen_state->height, 3, data, 0);
		// std::cout << "Screenshot saved!" << std::endl;
		delete[] data;
	}
}

void GUI::process_input_callback(GLFWwindow* window) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, true);
	}

	float current_frame = glfwGetTime();
	control_state->delta_time = current_frame - control_state->last_frame;
	control_state->last_frame = current_frame;

	float camera_speed = 2.5f * control_state->delta_time;
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		camera->move(MovementType::kMOVE_FORWARD, camera_speed);
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		camera->move(MovementType::kMOVE_BACKWARD, camera_speed);
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		camera->move(MovementType::kMOVE_LEFT, camera_speed);
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		camera->move(MovementType::kMOVE_RIGHT, camera_speed);
	}
}

void GUI::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
	auto& gui = GUI::instance();
	if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
		gui.control_state->is_right_mouse_pressed = true;
	}
	if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE) {
		gui.control_state->is_right_mouse_pressed = false;
		gui.control_state->is_first_mouse = true;
	}
}

void GUI::cursor_pos_callback(GLFWwindow* window, double _xpos, double _ypos) {
	// if right mouse is not pressed, nothing to do
	auto& gui = GUI::instance();
	if (!gui.control_state->is_right_mouse_pressed) {
		return;
	}

	float xpos = static_cast<float>(_xpos);
	float ypos = static_cast<float>(_ypos);
	if (gui.control_state->is_first_mouse) {
		gui.control_state->last_x = xpos;
		gui.control_state->last_y = ypos;
		gui.control_state->is_first_mouse = false;
	}
	float xoffset = xpos - gui.control_state->last_x;
	float yoffset = gui.control_state->last_y - ypos;
	gui.control_state->last_x = xpos;
	gui.control_state->last_y = ypos;

	float sensitivity = 0.1f;
	xoffset *= sensitivity;
	yoffset *= sensitivity;

	gui.control_state->yaw += xoffset;
	gui.control_state->pitch += yoffset;

	if (gui.control_state->pitch > 89.0f) {
		gui.control_state->pitch = 89.0f;
	}
	if (gui.control_state->pitch < -89.0f) {
		gui.control_state->pitch = -89.0f;
	}

	gui.camera->set_direction(sail::radians(gui.control_state->pitch), sail::radians(gui.control_state->yaw));
}

void GUI::scroll_callback(GLFWwindow* window, double _xoffset, double _yoffset) {
	auto& gui = GUI::instance();
	if (gui.control_state->fov >= 1.0f && gui.control_state->fov <= 45.0f) {
		gui.control_state->fov -= static_cast<float>(_yoffset);
	}
	if (gui.control_state->fov <= 1.0f) {
		gui.control_state->fov = 1.0f;
	}
	if (gui.control_state->fov >= 45.0f) {
		gui.control_state->fov = 45.0f;
	}

	gui.camera->set_fov(gui.control_state->fov);
}

void GUI::framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
	auto& gui = GUI::instance();
	gui.screen_state->update(width, height);
	gui.camera->set_aspect(gui.screen_state->aspect_ratio);
}

}// namespace sail
