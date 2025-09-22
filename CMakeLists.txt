cmake_minimum_required(VERSION 3.16)
project(untitled1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_TOOLCHAIN_FILE "C:/Users/jakem/vcpkg/scripts/buildsystems/vcpkg.cmake")

find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(ftxui REQUIRED)

# Add GLAD source file directly
add_executable(untitled1 main.cpp glad.c)

target_include_directories(untitled1 PRIVATE include)

target_link_libraries(untitled1
        glfw
        glm::glm
        ftxui::screen
        ftxui::dom
        ftxui::component
        opengl32
)