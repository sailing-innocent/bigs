#include "SailCu/util/misc.h"
#include <fstream>
#include <iostream>

namespace sail {

void read_file(std::string_view fname, std::vector<char>& buffer) {
	using std::ios;
	using std::string;

	std::ifstream file(fname.data(), ios::ate | ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("failed to open file: " + string(fname));
	}

	size_t fileSize = (size_t)file.tellg();

	buffer.resize(fileSize);
	file.seekg(0);
	file.read(buffer.data(), fileSize);
	file.close();
}

std::string read_file(std::string_view fname) {
	using std::ios;
	using std::string;
	std::ifstream file(fname.data(), ios::ate | ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("failed to open file: " + string(fname));
	}
	size_t fileSize = (size_t)file.tellg();
	std::string buffer(fileSize, '\0');
	file.seekg(0);
	file.read(buffer.data(), fileSize);
	file.close();
	return buffer;
}

}// namespace sail