#pragma once
/**
  * @file camera.h
  * @brief The GLM Camera Class
  * @author sailing-innocent
  * @date 2024-07-29
  */

#include "SailCu/util/camera_base.hpp"
#include <glm/glm.hpp>

namespace sail {

class GLMCamera : public Camera<glm::mat4, glm::vec3, float> {
public:
	GLMCamera(
		glm::vec3 position = {0.0f, -2.0f, 1.0f},
		glm::vec3 target = {0.0f, 0.0f, 0.0f},
		glm::vec3 up = {0.0f, 0.0f, 1.0f},
		float fov = sail::radians(45.0f),
		float aspect = 1.0f,
		float near = 0.1f,
		float far = 100.0f) : Camera{position, target, up, fov, aspect, near, far} {}
	~GLMCamera() = default;
	glm::mat4 get_view_matrix();
	glm::mat4 get_projection_matrix();
	glm::mat4 get_view_projection_matrix();
	glm::mat4 get_inverse_view_matrix();
	glm::mat4 get_inverse_projection_matrix();
};

}// namespace sail
