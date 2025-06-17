#include <vector>
#include <string>
#include <string_view>

namespace sail {

void read_file(std::string_view fname, std::vector<char>& buffer);
std::string read_file(std::string_view fname);

}// namespace sail
