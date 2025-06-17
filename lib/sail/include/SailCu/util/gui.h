#pragma once
/**
 * @file gui.h
 * @brief GUI impl in SailCu.dll
 * @author sailing-innocent
 * @date 2024-12-10
 */

#include "GLFW/glfw3.h"
#include "SailCu/util/camera.h"
#include <memory>

namespace sail {

struct ControlState {
	volatile bool is_first_mouse = true;
	volatile float last_x = 0.0f;
	volatile float last_y = 0.0f;
	volatile float yaw = 90.0f;
	volatile float pitch = 0.0f;
	volatile float fov = 45.0f;
	volatile float delta_time = 0.0f;
	volatile float last_frame = 0.0f;
	volatile bool is_right_mouse_pressed = false;
};

struct ScreenState {
	int width;
	int height;
	float aspect_ratio;
	ScreenState(int _w, int _h);
	void update(int _w, int _h);
};

class GUI {
private:
	GUI(int width, int height, bool flipy = false);
	~GUI();

public:
	GUI(const GUI&) = delete;
	GUI& operator=(const GUI&) = delete;
	GUI(GUI&&) = delete;
	GUI& operator=(GUI&&) = delete;

	static GUI& instance(int width = 800, int height = 600, bool flipy = false) {
		static GUI instance(width, height, flipy);
		return instance;
	}
	void process_input_callback(GLFWwindow* window);
	void bind_window(GLFWwindow* window);
	static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
	static void cursor_pos_callback(GLFWwindow* window, double _xpos, double _ypos);
	static void scroll_callback(GLFWwindow* window, double _xoffset, double _yoffset);
	static void framebuffer_size_callback(GLFWwindow* window, int width, int height);

	std::unique_ptr<ControlState> control_state = nullptr;
	std::unique_ptr<ScreenState> screen_state = nullptr;
	std::unique_ptr<GLMCamera> camera = nullptr;
};

}// namespace sail