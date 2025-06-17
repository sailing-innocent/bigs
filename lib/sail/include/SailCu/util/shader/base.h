#pragma once
/** 
 * @file shader/shader_base.h
 * @author sailing-innocent
 * @date 2023-12-15
 * @brief the terrain base shader
 */

#include <string>

namespace sail {

struct ShaderType {
	ShaderType() : m_type(-1) {}
	ShaderType(unsigned int type, std::string name) : m_type(type), m_name(name) {}
	unsigned int m_type;
	std::string m_name;
};// struct ShaderType

bool check_compile_errors(unsigned int shader, std::string type_name, std::string shader_name);
std::string get_shader_name(const char* shader_path);
ShaderType get_shader_type(const char* shader_path);

class ShaderBase {
public:
	// ShaderBase(const char* shader_path);
	static ShaderBase from_source(const char* shader_source, ShaderType type, std::string name);
	static ShaderBase from_file(const char* shader_path);

	virtual ~ShaderBase();
	std::string get_name() const { return get_shader_name(m_shader_path.c_str()); }
	ShaderType get_type() const { return m_shader_type; }
	unsigned int get_shad() const { return m_shad; }

private:
	std::string m_shader_path;
	ShaderType m_shader_type;
	unsigned int m_shad;
};

}// namespace sail