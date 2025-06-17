#pragma once
/** 
 * @file opengl/shader/shader_program.h
 * @author sailing-innocent
 * @date 2023-12-15
 * @brief the shader program wrapper
 */
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "base.h"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>

namespace sail {

using std::string_view;

class ShaderProgram {
public:
	ShaderProgram(std::string name);
	virtual ~ShaderProgram();
	ShaderProgram* attach_shader(ShaderBase s);
	void link_program();

	void use() const noexcept;
	// setter
	void set_mat4(const string_view name, const glm::mat4& mat) const noexcept;
	void set_bool(const string_view name, bool value) const noexcept;
	void set_int(const string_view name, int value) const noexcept;
	void set_float(const string_view name, float value) const noexcept;
	void set_float4(const string_view name, float v0, float v1, float v2, float v3) const noexcept;
	void set_vec3(const string_view name, const glm::vec3& vec) const noexcept;
	void set_vec4(const string_view name, const glm::vec4& vec) const noexcept;

protected:
	unsigned int m_id;
	bool is_linked, is_compute;
	std::vector<unsigned int> m_shaders;
	std::string m_name;
};

}// namespace sail::cu