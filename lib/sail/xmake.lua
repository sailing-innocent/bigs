set_xmakever("2.9.8")
add_rules("mode.release", "mode.debug", "mode.releasedbg")
set_languages("c++20")
add_requires("pybind11")
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
    add_packages("pybind11")
    set_kind("shared")
    set_extension(".pyd")
    add_files("src/**.cpp")
target_end()