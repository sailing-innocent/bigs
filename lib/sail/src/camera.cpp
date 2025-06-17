/**
  * @file glm_camera.cpp
  * @brief The GLM Camera Class Implementation
  * @author sailing-innocent
  * @date 2024-07-29
  */

#include "SailCu/util/camera.h"
#include <glm/gtc/matrix_transform.hpp>

namespace sail {

glm::mat4 GLMCamera::get_view_matrix() {
	if (m_is_view_dirty) {
		m_is_view_dirty = false;
		glm::vec3 position = get_position();
		glm::vec3 direction = get_direction();
		glm::vec3 up = get_up();
		m_view_matrix = glm::lookAt(position, position + direction, up);
	}
	return m_view_matrix;
}

glm::mat4 GLMCamera::get_projection_matrix() {
	if (m_is_projection_dirty) {
		m_is_projection_dirty = false;
		m_projection_matrix = glm::perspective(data.fov, data.aspect, data.near, data.far);
	}
	return m_projection_matrix;
}

glm::mat4 GLMCamera::get_view_projection_matrix() {
	return get_projection_matrix() * get_view_matrix();
}

glm::mat4 GLMCamera::get_inverse_view_matrix() {
	return glm::inverse(get_view_matrix());
}

glm::mat4 GLMCamera::get_inverse_projection_matrix() {
	return glm::inverse(get_projection_matrix());
}

}// namespace sail