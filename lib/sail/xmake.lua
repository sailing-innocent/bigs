set_xmakever("2.9.8")
add_rules("mode.release", "mode.debug", "mode.releasedbg")
set_languages("c++20")

add_requires("pybind11")
add_requires("glfw")

if is_mode("debug") then
    set_targetdir("bin/debug")
    set_runtimes("MDd")
elseif is_mode("releasedbg") then
    set_targetdir("bin/releasedbg")
    set_runtimes("MD")
else
    set_targetdir("bin/release")
    set_runtimes("MD")
end

target("sailc")
    add_rules("python.library")
    set_basename("sailc")
    add_packages("pybind11")
    add_packages("glfw")
    add_includedirs("include", {public = true})
    set_kind("shared")
    set_extension(".pyd")
    add_files("src/**.cpp", "src/**.cu")
target_end()