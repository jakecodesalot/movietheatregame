cmake_minimum_required(VERSION 3.16)
project(GraphicsEngine)

set(CMAKE_CXX_STANDARD 17)

# vcpkg integration
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

add_executable(GraphicsEngine main.cpp)

target_link_libraries(GraphicsEngine 
    PRIVATE 
    glfw 
    glad::glad 
    glm::glm
)

# OpenGL
if(WIN32)
    target_link_libraries(GraphicsEngine PRIVATE opengl32)
elseif(APPLE)
    find_package(OpenGL REQUIRED)
    target_link_libraries(GraphicsEngine PRIVATE ${OPENGL_LIBRARIES})
else()
    target_link_libraries(GraphicsEngine PRIVATE GL dl)
endif()
