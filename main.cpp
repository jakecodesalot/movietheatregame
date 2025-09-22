#define _USE_MATH_DEFINES
#include <cmath>

// windows h is mean
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#include <conio.h>
#else
#include <termios.h>
#include <unistd.h>
#include <sys/ioctl.h>
#endif

// open gl shebang
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// ftxui shabang
#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>

//fwd
class Camera;
class Scene;
class GameObject;
class Component;
class FTXUIRenderer;

// this my name space
using namespace ftxui;

//mathing it
namespace Math {
    using Vec2 = glm::vec2;
    using Vec3 = glm::vec3;
    using Vec4 = glm::vec4;
    using Mat4 = glm::mat4;
    using Quat = glm::quat;

    inline Mat4 identity() {
        return glm::mat4(1.0f);
    }

    inline Mat4 translate(const Mat4& m, const Vec3& v) {
        return glm::translate(m, v);
    }

    inline Mat4 rotate(const Mat4& m, float angle, const Vec3& axis) {
        return glm::rotate(m, angle, axis);
    }

    inline Mat4 scale(const Mat4& m, const Vec3& v) {
        return glm::scale(m, v);
    }

    inline Mat4 perspective(float fov, float aspect, float nearVal, float farVal) {
        return glm::perspective(fov, aspect, nearVal, farVal);
    }

    inline Mat4 lookAt(const Vec3& eye, const Vec3& center, const Vec3& up) {
        return glm::lookAt(eye, center, up);
    }
}

// colored pallettte
struct ColorPalette {
    std::vector<ftxui::Color> colors;

    ColorPalette() {
        // azul placeholder
        colors = {
            ftxui::Color::RGB(0, 0, 20),      // brooding blue
            ftxui::Color::RGB(0, 0, 40),      // kind blue
            ftxui::Color::RGB(0, 20, 60),     // mild timid blue
            ftxui::Color::RGB(0, 40, 80),     // jake blue
            ftxui::Color::RGB(20, 60, 100),   // light blue
            ftxui::Color::RGB(40, 80, 120),   // lighter blue
            ftxui::Color::RGB(60, 100, 140),  // lighterest blue
            ftxui::Color::RGB(80, 120, 160),  // very light blue
            ftxui::Color::RGB(100, 140, 180), // pale evil blue
            ftxui::Color::RGB(120, 160, 200), // very evil blue
            ftxui::Color::RGB(140, 180, 220), // almost white blue
            ftxui::Color::RGB(200, 220, 255)  // gus blue (cuz its white!!)
        };
    }

    ftxui::Color getColor(float intensity) const {
        if (intensity <= 0.0f) return colors[0];
        if (intensity >= 1.0f) return colors.back();

        float scaledIntensity = intensity * (colors.size() - 1);
        int index = static_cast<int>(scaledIntensity);

        if (index >= static_cast<int>(colors.size()) - 1) {
            return colors.back();
        }

        return colors[index];
    }
};

//timing it
class Time {
public:
    static float deltaTime;
    static float totalTime;
    static void update() {
        float currentTime = static_cast<float>(glfwGetTime());
        deltaTime = currentTime - totalTime;
        totalTime = currentTime;
    }
};
float Time::deltaTime = 0.0f;
float Time::totalTime = 0.0f;

//input handling
class Input {
private:
    static GLFWwindow* window;
    static std::unordered_map<int, bool> keys;
    static std::unordered_map<int, bool> keysPressed;
    static Math::Vec2 mousePos;
    static Math::Vec2 mouseDelta;
    static bool firstMouse;

public:
    static void initialize(GLFWwindow* win) {
        window = win;
        glfwSetKeyCallback(window, keyCallback);
        glfwSetCursorPosCallback(window, mouseCallback);
    }

    static bool getKey(int key) { return keys[key]; }
    static bool getKeyDown(int key) { return keysPressed[key]; }
    static Math::Vec2 getMouseDelta() { return mouseDelta; }

private:
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        if (action == GLFW_PRESS) {
            keys[key] = true;
            keysPressed[key] = true;
        }
        else if (action == GLFW_RELEASE) {
            keys[key] = false;
        }
    }

    static void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
        if (firstMouse) {
            mousePos = Math::Vec2(static_cast<float>(xpos), static_cast<float>(ypos));
            firstMouse = false;
        }

        mouseDelta = Math::Vec2(static_cast<float>(xpos) - mousePos.x, mousePos.y - static_cast<float>(ypos));
        mousePos = Math::Vec2(static_cast<float>(xpos), static_cast<float>(ypos));
    }

public:
    static void update() {
        for (auto& pair : keysPressed) {
            pair.second = false;
        }
        mouseDelta = Math::Vec2(0.0f);
    }
};

GLFWwindow* Input::window = nullptr;
std::unordered_map<int, bool> Input::keys;
std::unordered_map<int, bool> Input::keysPressed;
Math::Vec2 Input::mousePos = Math::Vec2(0.0f);
Math::Vec2 Input::mouseDelta = Math::Vec2(0.0f);
bool Input::firstMouse = true;

//shader handling
class Shader {
private:
    unsigned int ID;
    mutable std::unordered_map<std::string, int> uniformLocationCache;

public:
    Shader(const std::string& vertexSource, const std::string& fragmentSource) {
        ID = createProgram(vertexSource, fragmentSource);
    }

    ~Shader() { glDeleteProgram(ID); }

    void use() const { glUseProgram(ID); }

    void setMat4(const std::string& name, const Math::Mat4& mat) const {
        glUniformMatrix4fv(getUniformLocation(name), 1, GL_FALSE, glm::value_ptr(mat));
    }

    void setVec3(const std::string& name, const Math::Vec3& vec) const {
        glUniform3fv(getUniformLocation(name), 1, glm::value_ptr(vec));
    }

    void setFloat(const std::string& name, float value) const {
        glUniform1f(getUniformLocation(name), value);
    }

    void setInt(const std::string& name, int value) const {
        glUniform1i(getUniformLocation(name), value);
    }

private:
    int getUniformLocation(const std::string& name) const {
        if (uniformLocationCache.find(name) != uniformLocationCache.end()) {
            return uniformLocationCache[name];
        }

        int location = glGetUniformLocation(ID, name.c_str());
        uniformLocationCache[name] = location;
        return location;
    }

    unsigned int createProgram(const std::string& vertexSource, const std::string& fragmentSource) {
        unsigned int vertex = compileShader(GL_VERTEX_SHADER, vertexSource);
        unsigned int fragment = compileShader(GL_FRAGMENT_SHADER, fragmentSource);

        unsigned int program = glCreateProgram();
        glAttachShader(program, vertex);
        glAttachShader(program, fragment);
        glLinkProgram(program);

        int success;
        char infoLog[512];
        glGetProgramiv(program, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(program, 512, NULL, infoLog);
            std::cerr << "Shader linking failed: " << infoLog << std::endl;
        }

        glDeleteShader(vertex);
        glDeleteShader(fragment);
        return program;
    }

    unsigned int compileShader(GLenum type, const std::string& source) {
        unsigned int shader = glCreateShader(type);
        const char* src = source.c_str();
        glShaderSource(shader, 1, &src, nullptr);
        glCompileShader(shader);

        int success;
        char infoLog[512];
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 512, NULL, infoLog);
            std::cerr << "Shader compilation failed womp womp: " << infoLog << std::endl;
        }

        return shader;
    }
};

//im mesh handling
struct Vertex {
    Math::Vec3 Position;
    Math::Vec3 Normal;
    Math::Vec2 TexCoords;
};

class Mesh {
private:
    unsigned int VAO, VBO, EBO;

public:
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    Mesh(const std::vector<Vertex>& vertices, const std::vector<unsigned int>& indices)
        : vertices(vertices), indices(indices) {
        setupMesh();
    }

    ~Mesh() {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &EBO);
    }

    void render() const {
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }

private:
    void setupMesh() {
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(0);

        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, Normal)));
        glEnableVertexAttribArray(1);

        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, TexCoords)));
        glEnableVertexAttribArray(2);

        glBindVertexArray(0);
    }
};

// it says what it is read class jake
class FTXUIRenderer {
private:
    unsigned int framebuffer = 0, colorTexture = 0, depthBuffer = 0;
    int renderWidth = 1920, renderHeight = 1080;
    int displayWidth = 160, displayHeight = 45;
    std::vector<std::vector<ftxui::Color>> colorBuffer;
    std::vector<std::string> asciiBuffer;
    std::atomic<bool> shouldStop{false};
    std::thread renderThread;
    std::mutex bufferMutex;
    ColorPalette palette;

    ScreenInteractive screen = ScreenInteractive::Fullscreen();
    const std::string asciiChars = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";

public:
    FTXUIRenderer() {
        updateDisplaySize(displayWidth, displayHeight);
        std::cout << "FTXUI ASCII Renderer initialized" << std::endl;
    }

    void initializeOpenGL() {
        setupFramebuffer();
        std::cout << "OpenGL initialized" << std::endl;
    }

    ~FTXUIRenderer() {
        stop();
        if (framebuffer != 0) {
            glDeleteFramebuffers(1, &framebuffer);
        }
        if (colorTexture != 0) {
            glDeleteTextures(1, &colorTexture);
        }
        if (depthBuffer != 0) {
            glDeleteRenderbuffers(1, &depthBuffer);
        }
    }

    void start() {
        shouldStop = false;
        renderThread = std::thread(&FTXUIRenderer::runUI, this);
    }

    void stop() {
        shouldStop = true;
        if (renderThread.joinable()) {
            screen.ExitLoopClosure()();
            renderThread.join();
        }
    }

    void updateDisplaySize(int width, int height) {
        std::lock_guard<std::mutex> lock(bufferMutex);
        displayWidth = std::clamp(width, 40, 200);
        displayHeight = std::clamp(height, 20, 80);
        asciiBuffer.assign(displayHeight, std::string(displayWidth, ' '));
        colorBuffer.assign(displayHeight, std::vector<ftxui::Color>(displayWidth, ftxui::Color::Black));
    }

    void beginRender() {
        if (framebuffer != 0) {
            glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
            glViewport(0, 0, renderWidth, renderHeight);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        }
    }

    void endRender() {
        if (framebuffer == 0) return;

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        std::vector<float> pixels(renderWidth * renderHeight * 3);
        glBindTexture(GL_TEXTURE_2D, colorTexture);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, pixels.data());

        std::lock_guard<std::mutex> lock(bufferMutex);

        float scaleX = static_cast<float>(renderWidth) / displayWidth;
        float scaleY = static_cast<float>(renderHeight) / displayHeight;

        for (int y = 0; y < displayHeight; y++) {
            for (int x = 0; x < displayWidth; x++) {
                float totalR = 0, totalG = 0, totalB = 0;
                int samples = 0;

                int startX = static_cast<int>(x * scaleX);
                int endX = static_cast<int>((x + 1) * scaleX);
                int startY = static_cast<int>(y * scaleY);
                int endY = static_cast<int>((y + 1) * scaleY);

                for (int py = startY; py < endY && py < renderHeight; py++) {
                    for (int px = startX; px < endX && px < renderWidth; px++) {
                        int pixelIndex = (py * renderWidth + px) * 3;
                        totalR += pixels[pixelIndex];
                        totalG += pixels[pixelIndex + 1];
                        totalB += pixels[pixelIndex + 2];
                        samples++;
                    }
                }

                if (samples > 0) {
                    totalR /= samples;
                    totalG /= samples;
                    totalB /= samples;
                }

                float luminance = 0.299f * totalR + 0.587f * totalG + 0.114f * totalB;
                int charIndex = static_cast<int>(luminance * (asciiChars.length() - 1));
                charIndex = std::clamp(charIndex, 0, static_cast<int>(asciiChars.length() - 1));

                ftxui::Color pixelColor = palette.getColor(luminance);

                int asciiY = displayHeight - 1 - y;
                if (asciiY >= 0 && asciiY < displayHeight) {
                    asciiBuffer[asciiY][x] = asciiChars[charIndex];
                    colorBuffer[asciiY][x] = pixelColor;
                }
            }
        }
    }

    Math::Vec2 getAspectRatio() const {
        return Math::Vec2(static_cast<float>(renderWidth), static_cast<float>(renderHeight));
    }

private:
    void setupFramebuffer() {
        if (framebuffer != 0) {
            glDeleteFramebuffers(1, &framebuffer);
            glDeleteTextures(1, &colorTexture);
            glDeleteRenderbuffers(1, &depthBuffer);
        }

        glGenFramebuffers(1, &framebuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

        glGenTextures(1, &colorTexture);
        glBindTexture(GL_TEXTURE_2D, colorTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, renderWidth, renderHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture, 0);

        glGenRenderbuffers(1, &depthBuffer);
        glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, renderWidth, renderHeight);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, depthBuffer);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            std::cerr << "Framebuffer not complete!" << std::endl;
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void runUI() {
        auto component = Renderer([&] {
            std::lock_guard<std::mutex> lock(bufferMutex);

            std::vector<Element> lines;
            for (int i = 0; i < displayHeight; i++) {
                if (i < static_cast<int>(asciiBuffer.size())) {
                    std::vector<Element> chars;
                    for (int j = 0; j < displayWidth && j < static_cast<int>(asciiBuffer[i].length()); j++) {
                        std::string charStr(1, asciiBuffer[i][j]);
                        chars.push_back(text(charStr) | color(colorBuffer[i][j]));
                    }
                    lines.push_back(hbox(chars));
                }
            }

            return vbox({
                text("3D Renderer [1920x1080 -> " +
                     std::to_string(displayWidth) + "x" + std::to_string(displayHeight) + "]") | bold | center,
                separator(),
                vbox(lines),
                separator(),
                hbox({
                    text("WASD+Mouse: Move | Space/Shift: Up/Down") | dim,
                    text(" | ") | dim,
                    text("T: Toggle OpenGL | R: Reset Camera | ESC: Exit") | dim
                }) | center
            }) | border;
        });

        std::thread refresh_thread([&] {
            while (!shouldStop) {
                std::this_thread::sleep_for(std::chrono::milliseconds(33));
                if (!shouldStop) {
                    screen.PostEvent(Event::Custom);
                }
            }
        });

        screen.Loop(component);
        shouldStop = true;

        if (refresh_thread.joinable()) {
            refresh_thread.join();
        }
    }
};

class Transform {
public:
    Math::Vec3 position{0.0f};
    Math::Vec3 rotation{0.0f};
    Math::Vec3 scale{1.0f};

    Math::Mat4 getMatrix() const {
        Math::Mat4 t = Math::translate(Math::identity(), position);
        Math::Mat4 r = Math::rotate(Math::identity(), rotation.x, Math::Vec3(1, 0, 0));
        r = Math::rotate(r, rotation.y, Math::Vec3(0, 1, 0));
        r = Math::rotate(r, rotation.z, Math::Vec3(0, 0, 1));
        Math::Mat4 s = Math::scale(Math::identity(), scale);
        return t * r * s;
    }

    void translate(const Math::Vec3& delta) { position += delta; }
    void rotate(const Math::Vec3& delta) { rotation += delta; }
};

class Camera {
public:
    Math::Vec3 position{0.0f, 0.0f, 8.0f};
    Math::Vec3 front{0.0f, 0.0f, -1.0f};
    Math::Vec3 up{0.0f, 1.0f, 0.0f};
    Math::Vec3 right{1.0f, 0.0f, 0.0f};

    float yaw{-90.0f};
    float pitch{0.0f};
    float fov{45.0f};
    float nearPlane{0.1f};
    float farPlane{100.0f};

    void processInput(float deltaTime) {
        float speed = 8.0f * deltaTime;

        if (Input::getKey(GLFW_KEY_W))
            position += speed * front;
        if (Input::getKey(GLFW_KEY_S))
            position -= speed * front;
        if (Input::getKey(GLFW_KEY_A))
            position -= speed * right;
        if (Input::getKey(GLFW_KEY_D))
            position += speed * right;
        if (Input::getKey(GLFW_KEY_SPACE))
            position += speed * up;
        if (Input::getKey(GLFW_KEY_LEFT_SHIFT))
            position -= speed * up;
    }

    void processMouseInput(Math::Vec2 mouseDelta) {
        float sensitivity = 0.15f;
        yaw += mouseDelta.x * sensitivity;
        pitch += mouseDelta.y * sensitivity;

        if (pitch > 89.0f) pitch = 89.0f;
        if (pitch < -89.0f) pitch = -89.0f;

        updateVectors();
    }

    void reset() {
        position = Math::Vec3(0.0f, 0.0f, 8.0f);
        yaw = -90.0f;
        pitch = 0.0f;
        updateVectors();
    }

    Math::Mat4 getViewMatrix() const {
        return Math::lookAt(position, position + front, up);
    }

    Math::Mat4 getProjectionMatrix(float aspectRatio) const {
        return Math::perspective(glm::radians(fov), aspectRatio, nearPlane, farPlane);
    }

private:
    void updateVectors() {
        Math::Vec3 newFront;
        newFront.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        newFront.y = sin(glm::radians(pitch));
        newFront.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        front = glm::normalize(newFront);
        right = glm::normalize(glm::cross(front, Math::Vec3(0.0f, 1.0f, 0.0f)));
        up = glm::normalize(glm::cross(right, front));
    }
};

class GameObject {
public:
    Transform transform;
    std::unique_ptr<Mesh> mesh;
    bool active = true;

    GameObject() = default;

    void setMesh(std::unique_ptr<Mesh> newMesh) {
        mesh = std::move(newMesh);
    }

    virtual void update(float deltaTime) {}

    void render(const Shader& shader) {
        if (mesh && active) {
            shader.setMat4("model", transform.getMatrix());
            mesh->render();
        }
    }
};

class SpinningObject : public GameObject {
public:
    Math::Vec3 spinSpeed{0.0f, 1.0f, 0.5f};

    void update(float deltaTime) override {
        transform.rotate(spinSpeed * deltaTime);
    }
};

class Scene {
private:
    std::vector<std::unique_ptr<GameObject>> gameObjects;

public:
    void addGameObject(std::unique_ptr<GameObject> obj) {
        gameObjects.push_back(std::move(obj));
    }

    void update(float deltaTime) {
        for (auto& obj : gameObjects) {
            if (obj && obj->active) {
                obj->update(deltaTime);
            }
        }
    }

    void render(const Shader& shader) {
        for (auto& obj : gameObjects) {
            if (obj) {
                obj->render(shader);
            }
        }
    }

    size_t getObjectCount() const { return gameObjects.size(); }
};

class GraphicsEngine {
private:
    GLFWwindow* window;
    std::unique_ptr<Shader> defaultShader;
    std::unique_ptr<FTXUIRenderer> ftxuiRenderer;
    Camera camera;
    Scene scene;
    bool showOpenGLWindow = false;

    static const int OPENGL_WIDTH = 1920;
    static const int OPENGL_HEIGHT = 1080;

    const std::string vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aNormal;
        layout (location = 2) in vec2 aTexCoords;

        out vec3 FragPos;
        out vec3 Normal;
        out vec2 TexCoords;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        void main() {
            FragPos = vec3(model * vec4(aPos, 1.0));
            Normal = mat3(transpose(inverse(model))) * aNormal;
            TexCoords = aTexCoords;

            gl_Position = projection * view * vec4(FragPos, 1.0);
        }
    )";

    const std::string fragmentShaderSource = R"(
        #version 330 core
        out vec4 FragColor;

        in vec3 FragPos;
        in vec3 Normal;
        in vec2 TexCoords;

        uniform vec3 lightPos;
        uniform vec3 viewPos;
        uniform vec3 lightColor;
        uniform vec3 objectColor;

        void main() {
            float ambientStrength = 0.15;
            vec3 ambient = ambientStrength * lightColor;

            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;

            float specularStrength = 0.6;
            vec3 viewDir = normalize(viewPos - FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 64);
            vec3 specular = specularStrength * spec * lightColor;

            vec3 result = (ambient + diffuse + specular) * objectColor;
            FragColor = vec4(result, 1.0);
        }
    )";

public:
    bool initialize() {
        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW" << std::endl;
            return false;
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_FOCUSED, GLFW_TRUE);

        window = glfwCreateWindow(OPENGL_WIDTH, OPENGL_HEIGHT, "3D Renderer", nullptr, nullptr);
        if (!window) {
            std::cerr << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return false;
        }

        glfwMakeContextCurrent(window);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        glfwSwapInterval(0);

        if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
            std::cerr << "Failed to initialize GLAD" << std::endl;
            return false;
        }

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glEnable(GL_MULTISAMPLE);

        Input::initialize(window);
        defaultShader = std::make_unique<Shader>(vertexShaderSource, fragmentShaderSource);
        ftxuiRenderer = std::make_unique<FTXUIRenderer>();

        ftxuiRenderer->initializeOpenGL();

        std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
        std::cout << "\nControls:" << std::endl;
        std::cout << "  WASD + Mouse: Move camera" << std::endl;
        std::cout << "  Space/Shift: Move up/down" << std::endl;
        std::cout << "  T: Toggle OpenGL window" << std::endl;
        std::cout << "  R: Reset camera" << std::endl;
        std::cout << "  ESC: Exit" << std::endl;

        ftxuiRenderer->start();
        return true;
    }

    void createAdvancedTestScene() {
        auto cube = std::make_unique<SpinningObject>();
        cube->transform.scale = Math::Vec3(2.0f);
        cube->spinSpeed = Math::Vec3(0.3f, 0.8f, 0.4f);
        cube->setMesh(createCubeMesh());
        scene.addGameObject(std::move(cube));

        auto pyramid = std::make_unique<SpinningObject>();
        pyramid->transform.position = Math::Vec3(5.0f, 0.0f, 0.0f);
        pyramid->transform.scale = Math::Vec3(1.5f);
        pyramid->spinSpeed = Math::Vec3(-0.2f, 1.2f, -0.3f);
        pyramid->setMesh(createPyramidMesh());
        scene.addGameObject(std::move(pyramid));

        auto complexObj = std::make_unique<SpinningObject>();
        complexObj->transform.position = Math::Vec3(-4.0f, 2.0f, -2.0f);
        complexObj->transform.scale = Math::Vec3(1.2f);
        complexObj->spinSpeed = Math::Vec3(0.5f, -0.4f, 0.7f);
        complexObj->setMesh(createComplexMesh());
        scene.addGameObject(std::move(complexObj));

        auto floor = std::make_unique<GameObject>();
        floor->transform.position = Math::Vec3(0.0f, -3.0f, 0.0f);
        floor->transform.scale = Math::Vec3(10.0f, 0.1f, 10.0f);
        floor->setMesh(createCubeMesh());
        scene.addGameObject(std::move(floor));

        std::cout << "test scene is great" << std::endl;
    }
// my cube.. my shame
    std::unique_ptr<Mesh> createCubeMesh() {
        std::vector<Vertex> vertices = {
            {{-1.0f, -1.0f,  1.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
            {{ 1.0f, -1.0f,  1.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
            {{ 1.0f,  1.0f,  1.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
            {{-1.0f,  1.0f,  1.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
            {{-1.0f, -1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
            {{-1.0f,  1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
            {{ 1.0f,  1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},
            {{ 1.0f, -1.0f, -1.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
            {{-1.0f, -1.0f, -1.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
            {{-1.0f, -1.0f,  1.0f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
            {{-1.0f,  1.0f,  1.0f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
            {{-1.0f,  1.0f, -1.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
            {{ 1.0f, -1.0f, -1.0f}, { 1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
            {{ 1.0f,  1.0f, -1.0f}, { 1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
            {{ 1.0f,  1.0f,  1.0f}, { 1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
            {{ 1.0f, -1.0f,  1.0f}, { 1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
            {{-1.0f,  1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
            {{-1.0f,  1.0f,  1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
            {{ 1.0f,  1.0f,  1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
            {{ 1.0f,  1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
            {{-1.0f, -1.0f, -1.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
            {{ 1.0f, -1.0f, -1.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
            {{ 1.0f, -1.0f,  1.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
            {{-1.0f, -1.0f,  1.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}}
        };

        std::vector<unsigned int> indices = {
            0,  1,  2,   2,  3,  0,   4,  5,  6,   6,  7,  4,
            8,  9,  10,  10, 11, 8,   12, 13, 14,  14, 15, 12,
            16, 17, 18,  18, 19, 16,  20, 21, 22,  22, 23, 20
        };

        return std::make_unique<Mesh>(vertices, indices);
    }

    std::unique_ptr<Mesh> createPyramidMesh() {
        std::vector<Vertex> vertices = {
            {{-1.0f, -1.0f, -1.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
            {{ 1.0f, -1.0f, -1.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
            {{ 1.0f, -1.0f,  1.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
            {{-1.0f, -1.0f,  1.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
            {{ 0.0f,  1.5f,  0.0f}, {0.0f, 1.0f, 0.0f}, {0.5f, 0.5f}}
        };

        std::vector<unsigned int> indices = {
            0, 1, 2, 2, 3, 0,  0, 4, 1,  1, 4, 2,  2, 4, 3,  3, 4, 0
        };

        return std::make_unique<Mesh>(vertices, indices);
    }

    std::unique_ptr<Mesh> createComplexMesh() {
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;

        const float radius = 1.0f;
        const int segments = 8;

        vertices.push_back({{0.0f, 1.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.5f, 1.0f}});

        for (int i = 0; i < segments; ++i) {
            float angle = 2.0f * static_cast<float>(M_PI) * i / segments;
            float x = radius * cos(angle);
            float z = radius * sin(angle);
            Math::Vec3 normal = glm::normalize(Math::Vec3(x, 0.0f, z));
            vertices.push_back({{x, 0.0f, z}, normal, {static_cast<float>(i) / segments, 0.5f}});
        }

        vertices.push_back({{0.0f, -1.5f, 0.0f}, {0.0f, -1.0f, 0.0f}, {0.5f, 0.0f}});

        for (int i = 0; i < segments; ++i) {
            indices.push_back(0);
            indices.push_back(1 + i);
            indices.push_back(1 + (i + 1) % segments);
        }

        for (int i = 0; i < segments; ++i) {
            indices.push_back(segments + 1);
            indices.push_back(1 + (i + 1) % segments);
            indices.push_back(1 + i);
        }

        return std::make_unique<Mesh>(vertices, indices);
    }

    void run() {
        std::cout << "ASCII rendering..." << std::endl;

        const double targetFrameTime = 1.0 / 30.0;
        double lastFrameTime = glfwGetTime();

        while (!glfwWindowShouldClose(window)) {
            double currentTime = glfwGetTime();
            double deltaTime = currentTime - lastFrameTime;

            if (deltaTime >= targetFrameTime) {
                Time::update();
                glfwPollEvents();

                if (Input::getKey(GLFW_KEY_ESCAPE)) {
                    glfwSetWindowShouldClose(window, true);
                    break;
                }

                if (Input::getKeyDown(GLFW_KEY_T)) {
                    showOpenGLWindow = !showOpenGLWindow;
                    if (showOpenGLWindow) {
                        glfwShowWindow(window);
                    } else {
                        glfwHideWindow(window);
                    }
                }

                if (Input::getKeyDown(GLFW_KEY_R)) {
                    camera.reset();
                }

                camera.processInput(Time::deltaTime);
                camera.processMouseInput(Input::getMouseDelta());
                scene.update(Time::deltaTime);

                int termWidth = 160, termHeight = 45;

                #ifdef _WIN32
                CONSOLE_SCREEN_BUFFER_INFO csbi;
                HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
                if (GetConsoleScreenBufferInfo(hConsole, &csbi)) {
                    termWidth = std::max(80, csbi.srWindow.Right - csbi.srWindow.Left + 1);
                    termHeight = std::max(24, csbi.srWindow.Bottom - csbi.srWindow.Top + 1 - 4);
                }
                #endif

                ftxuiRenderer->updateDisplaySize(termWidth, termHeight);

                Math::Vec2 renderSize = ftxuiRenderer->getAspectRatio();
                float aspectRatio = renderSize.x / renderSize.y;

                ftxuiRenderer->beginRender();
                renderScene(aspectRatio);
                ftxuiRenderer->endRender();

                if (showOpenGLWindow) {
                    glBindFramebuffer(GL_FRAMEBUFFER, 0);
                    glViewport(0, 0, OPENGL_WIDTH, OPENGL_HEIGHT);
                    glClearColor(0.02f, 0.02f, 0.08f, 1.0f);
                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                    renderScene(static_cast<float>(OPENGL_WIDTH) / OPENGL_HEIGHT);
                    glfwSwapBuffers(window);
                }

                Input::update();
                lastFrameTime = currentTime;
            } else {
                glfwPollEvents();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        ftxuiRenderer->stop();
    }

    void renderScene(float aspectRatio) {
        defaultShader->use();
        defaultShader->setMat4("view", camera.getViewMatrix());
        defaultShader->setMat4("projection", camera.getProjectionMatrix(aspectRatio));
        defaultShader->setVec3("viewPos", camera.position);
        defaultShader->setVec3("lightPos", Math::Vec3(6.0f, 10.0f, 8.0f));
        defaultShader->setVec3("lightColor", Math::Vec3(1.2f, 1.1f, 1.0f));
        defaultShader->setVec3("objectColor", Math::Vec3(0.9f, 0.85f, 0.7f));

        scene.render(*defaultShader);
    }

    void shutdown() {
        #ifdef _WIN32
        system("cls");
        #else
        system("clear");
        #endif

        std::cout << " complete" << std::endl;
        glfwTerminate();
    }
};

int main() {
    GraphicsEngine engine;

    if (!engine.initialize()) {
        return -1;
    }

    engine.createAdvancedTestScene();
    engine.run();
    engine.shutdown();

    return 0;
}