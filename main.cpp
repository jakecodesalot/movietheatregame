#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <algorithm>

//fwd
class Renderer;
class Camera;
class Scene;
class GameObject;
class Component;
class ASCIIFilter;

//mathing it
namespace Math {
    using Vec2 = glm::vec2;
    using Vec3 = glm::vec3;
    using Vec4 = glm::vec4;
    using Mat4 = glm::mat4;
    using Quat = glm::quat;

    inline Mat4 identity() { return glm::mat4(1.0f); }
    inline Mat4 translate(const Mat4& m, const Vec3& v) { return glm::translate(m, v); }
    inline Mat4 rotate(const Mat4& m, float angle, const Vec3& axis) { return glm::rotate(m, angle, axis); }
    inline Mat4 scale(const Mat4& m, const Vec3& v) { return glm::scale(m, v); }
    inline Mat4 perspective(float fov, float aspect, float near, float far) {
        return glm::perspective(fov, aspect, near, far);
    }
    inline Mat4 lookAt(const Vec3& eye, const Vec3& center, const Vec3& up) {
        return glm::lookAt(eye, center, up);
    }
}

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

//in my put till she input
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

//shading it
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

    void setVec2(const std::string& name, const Math::Vec2& vec) const {
        glUniform2fv(getUniformLocation(name), 1, glm::value_ptr(vec));
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

        //checking it till i linking it
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

        //checking it till im compiling it
        int success;
        char infoLog[512];
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 512, NULL, infoLog);
            std::cerr << "Shader compilation failed: " << infoLog << std::endl;
        }

        return shader;
    }
};

//meshing it
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

        // Position
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(0);

        // Normal
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, Normal)));
        glEnableVertexAttribArray(1);

        // Texture coordinates
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, TexCoords)));
        glEnableVertexAttribArray(2);

        glBindVertexArray(0);
    }
};

// loading mah resources
class ResourceLoader {
public:
    static std::unique_ptr<Mesh> loadOBJ(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Failed to open OBJ file: " << filepath << std::endl;
            return nullptr;
        }

        std::vector<Math::Vec3> temp_vertices;
        std::vector<Math::Vec3> temp_normals;
        std::vector<Math::Vec2> temp_texCoords;
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string prefix;
            iss >> prefix;

            if (prefix == "v") {
                Math::Vec3 vertex;
                iss >> vertex.x >> vertex.y >> vertex.z;
                temp_vertices.push_back(vertex);
            }
            else if (prefix == "vn") {
                Math::Vec3 normal;
                iss >> normal.x >> normal.y >> normal.z;
                temp_normals.push_back(normal);
            }
            else if (prefix == "vt") {
                Math::Vec2 texCoord;
                iss >> texCoord.x >> texCoord.y;
                temp_texCoords.push_back(texCoord);
            }
            else if (prefix == "f") {
                std::string vertex1, vertex2, vertex3;
                iss >> vertex1 >> vertex2 >> vertex3;

                auto parseVertex = [&](const std::string& vertexStr) -> Vertex {
                    std::istringstream vss(vertexStr);
                    std::string item;
                    std::vector<std::string> items;
                    while (std::getline(vss, item, '/')) {
                        items.push_back(item);
                    }

                    Vertex vertex;

                    //pos
                    int v_idx = std::stoi(items[0]) - 1;
                    vertex.Position = temp_vertices[v_idx];

                    //texture coords
                    if (items.size() > 1 && !items[1].empty()) {
                        int vt_idx = std::stoi(items[1]) - 1;
                        vertex.TexCoords = temp_texCoords[vt_idx];
                    } else {
                        vertex.TexCoords = Math::Vec2(0.0f);
                    }

                    //new norm
                    if (items.size() > 2 && !items[2].empty()) {
                        int vn_idx = std::stoi(items[2]) - 1;
                        vertex.Normal = temp_normals[vn_idx];
                    } else {
                        vertex.Normal = Math::Vec3(0.0f, 1.0f, 0.0f);
                    }

                    return vertex;
                };

                vertices.push_back(parseVertex(vertex1));
                vertices.push_back(parseVertex(vertex2));
                vertices.push_back(parseVertex(vertex3));

                indices.push_back(static_cast<unsigned int>(vertices.size() - 3));
                indices.push_back(static_cast<unsigned int>(vertices.size() - 2));
                indices.push_back(static_cast<unsigned int>(vertices.size() - 1));
            }
        }

        std::cout << "Loaded OBJ: " << vertices.size() << " vertices, " << indices.size() << " indices" << std::endl;
        return std::make_unique<Mesh>(vertices, indices);
    }
};

//asciing it
class ASCIIFilter {
private:
    unsigned int framebuffer, colorTexture, depthBuffer;
    unsigned int quadVAO, quadVBO;
    std::unique_ptr<Shader> asciiShader;
    int width, height;
    float charSize;

    const std::string asciiVertexShader = R"(
        #version 330 core
        layout (location = 0) in vec2 aPos;
        layout (location = 1) in vec2 aTexCoords;

        out vec2 TexCoords;

        void main() {
            TexCoords = aTexCoords;
            gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
        }
    )";

    const std::string asciiFragmentShader = R"(
        #version 330 core
        out vec4 FragColor;

        in vec2 TexCoords;

        uniform sampler2D screenTexture;
        uniform vec2 resolution;
        uniform float charSize;

        // Simple ASCII character patterns using mathematical functions
        float getCharPattern(int charIndex, vec2 localPos) {
            vec2 pos = localPos;
            float result = 0.0;

            if (charIndex == 0) return 0.0; // space
            if (charIndex == 1) { // .
                float dist = distance(pos, vec2(0.5));
                return 1.0 - smoothstep(0.1, 0.2, dist);
            }
            if (charIndex == 2) { // :
                float d1 = distance(pos, vec2(0.5, 0.3));
                float d2 = distance(pos, vec2(0.5, 0.7));
                return (1.0 - smoothstep(0.05, 0.15, d1)) + (1.0 - smoothstep(0.05, 0.15, d2));
            }
            if (charIndex == 3) { // -
                return (abs(pos.y - 0.5) < 0.1) ? 1.0 : 0.0;
            }
            if (charIndex == 4) { // =
                float line1 = (abs(pos.y - 0.4) < 0.05) ? 1.0 : 0.0;
                float line2 = (abs(pos.y - 0.6) < 0.05) ? 1.0 : 0.0;
                return line1 + line2;
            }
            if (charIndex == 5) { // +
                float h = (abs(pos.y - 0.5) < 0.08) ? 1.0 : 0.0;
                float v = (abs(pos.x - 0.5) < 0.08) ? 1.0 : 0.0;
                return max(h, v);
            }
            if (charIndex == 6) { // *
                float d1 = abs(pos.x - pos.y) < 0.08 ? 1.0 : 0.0;
                float d2 = abs(pos.x + pos.y - 1.0) < 0.08 ? 1.0 : 0.0;
                float h = (abs(pos.y - 0.5) < 0.05) ? 1.0 : 0.0;
                float v = (abs(pos.x - 0.5) < 0.05) ? 1.0 : 0.0;
                return max(max(d1, d2), max(h, v));
            }
            if (charIndex == 7) { // #
                float h1 = (abs(pos.y - 0.35) < 0.05) ? 1.0 : 0.0;
                float h2 = (abs(pos.y - 0.65) < 0.05) ? 1.0 : 0.0;
                float v1 = (abs(pos.x - 0.35) < 0.05) ? 1.0 : 0.0;
                float v2 = (abs(pos.x - 0.65) < 0.05) ? 1.0 : 0.0;
                return max(max(h1, h2), max(v1, v2));
            }
            if (charIndex == 8) { // %
                float c1 = distance(pos, vec2(0.25, 0.75)) < 0.15 ? 1.0 : 0.0;
                float c2 = distance(pos, vec2(0.75, 0.25)) < 0.15 ? 1.0 : 0.0;
                float line = abs(pos.x + pos.y - 1.0) < 0.05 ? 1.0 : 0.0;
                return max(max(c1, c2), line);
            }
            // Default for index 9 (@) - solid block
            return 1.0;
        }

        void main() {
            // Calculate ASCII grid position
            vec2 gridSize = resolution / charSize;
            vec2 gridPos = floor(TexCoords * gridSize);
            vec2 cellPos = fract(TexCoords * gridSize);

            // Sample color at cell center
            vec2 samplePos = (gridPos + 0.5) / gridSize;
            vec3 color = texture(screenTexture, samplePos).rgb;

            // Calculate luminance
            float luminance = dot(color, vec3(0.299, 0.587, 0.114));

            // Map to ASCII character
            int charIndex = int(luminance * 9.0);
            charIndex = clamp(charIndex, 0, 9);

            // Get pattern value
            float pattern = getCharPattern(charIndex, cellPos);

            // Terminal colors
            vec3 terminalGreen = vec3(0.0, 1.0, 0.3);
            vec3 bgColor = vec3(0.0, 0.05, 0.0);

            // Final color
            vec3 finalColor = mix(bgColor, terminalGreen * luminance, pattern);
            FragColor = vec4(finalColor, 1.0);
        }
    )";

public:
    ASCIIFilter(int w, int h) : width(w), height(h), charSize(8.0f) {
        setupFramebuffer();
        setupQuad();
        asciiShader = std::make_unique<Shader>(asciiVertexShader, asciiFragmentShader);

        std::cout << "ASCII Filter initialized: " << width/charSize << "x" << height/charSize << " characters" << std::endl;
    }

    ~ASCIIFilter() {
        glDeleteFramebuffers(1, &framebuffer);
        glDeleteTextures(1, &colorTexture);
        glDeleteRenderbuffers(1, &depthBuffer);
        glDeleteVertexArrays(1, &quadVAO);
        glDeleteBuffers(1, &quadVBO);
    }

    void beginRender() {
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    void endRender() {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);

        glDisable(GL_DEPTH_TEST);

        asciiShader->use();
        asciiShader->setVec2("resolution", Math::Vec2(width, height));
        asciiShader->setFloat("charSize", charSize);
        asciiShader->setInt("screenTexture", 0);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, colorTexture);

        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);

        glEnable(GL_DEPTH_TEST);
    }

    void setCharacterSize(float size) {
        charSize = std::max(4.0f, std::min(32.0f, size));
    }

private:
    void setupFramebuffer() {
        glGenFramebuffers(1, &framebuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

        //coloring it till i texture
        glGenTextures(1, &colorTexture);
        glBindTexture(GL_TEXTURE_2D, colorTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture, 0);

        //depthing it till i buffer
        glGenRenderbuffers(1, &depthBuffer);
        glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, depthBuffer);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            std::cerr << "Framebuffer not complete!" << std::endl;
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void setupQuad() {
        const float quadVertices[] = {
            // positions   // tex coords
            -1.0f,  1.0f,  0.0f, 1.0f,
            -1.0f, -1.0f,  0.0f, 0.0f,
             1.0f, -1.0f,  1.0f, 0.0f,

            -1.0f,  1.0f,  0.0f, 1.0f,
             1.0f, -1.0f,  1.0f, 0.0f,
             1.0f,  1.0f,  1.0f, 1.0f
        };

        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
    }
};

//transforming it till i component
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

//cameraing it till i component
class Camera {
public:
    Math::Vec3 position{0.0f, 0.0f, 3.0f};
    Math::Vec3 front{0.0f, 0.0f, -1.0f};
    Math::Vec3 up{0.0f, 1.0f, 0.0f};
    Math::Vec3 right{1.0f, 0.0f, 0.0f};

    float yaw{-90.0f};
    float pitch{0.0f};
    float fov{45.0f};
    float nearPlane{0.1f};
    float farPlane{100.0f};

    void processInput(float deltaTime) {
        float speed = 5.0f * deltaTime;

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
        float sensitivity = 0.1f;
        yaw += mouseDelta.x * sensitivity;
        pitch += mouseDelta.y * sensitivity;

        if (pitch > 89.0f) pitch = 89.0f;
        if (pitch < -89.0f) pitch = -89.0f;

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

//gaming it till i object
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

//spinning mah object for testing
class SpinningObject : public GameObject {
public:
    Math::Vec3 spinSpeed{0.0f, 1.0f, 0.5f};

    void update(float deltaTime) override {
        transform.rotate(spinSpeed * deltaTime);
    }
};

//scening my manage
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

//grafix
class GraphicsEngine {
private:
    GLFWwindow* window;
    std::unique_ptr<Shader> defaultShader;
    std::unique_ptr<ASCIIFilter> asciiFilter;
    Camera camera;
    Scene scene;
    bool asciiMode = true;

    static const int WIDTH = 1200;
    static const int HEIGHT = 800;

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
            // Ambient
            float ambientStrength = 0.15;
            vec3 ambient = ambientStrength * lightColor;

            // Diffuse
            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;

            // Specular
            float specularStrength = 0.8;
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
        // initialize GLFW
        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW" << std::endl;
            return false;
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_SAMPLES, 4);

        window = glfwCreateWindow(WIDTH, HEIGHT, "ASCII 3D Graphics Engine", nullptr, nullptr);
        if (!window) {
            std::cerr << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return false;
        }

        glfwMakeContextCurrent(window);
        glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        glfwSwapInterval(1);

        // initialize OpenGL
        if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
            std::cerr << "Failed to initialize GLAD" << std::endl;
            return false;
        }

        // OpenGL settings
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_MULTISAMPLE);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);

        // initialize systems
        Input::initialize(window);
        defaultShader = std::make_unique<Shader>(vertexShaderSource, fragmentShaderSource);
        asciiFilter = std::make_unique<ASCIIFilter>(WIDTH, HEIGHT);

        std::cout << "ASCII Graphics Engine initialized successfully!" << std::endl;
        std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
        std::cout << "\nControls:" << std::endl;
        std::cout << "  WASD + Mouse: Move camera" << std::endl;
        std::cout << "  Space/Shift: Move up/down" << std::endl;
        std::cout << "  T: Toggle ASCII mode" << std::endl;
        std::cout << "  +/-: Adjust ASCII character size" << std::endl;
        std::cout << "  ESC: Exit" << std::endl;

        return true;
    }

    void loadTestScene() {
        // loading mah obj
        auto mesh = ResourceLoader::loadOBJ("model.obj");
        if (mesh) {
            auto spinningObj = std::make_unique<SpinningObject>();
            spinningObj->setMesh(std::move(mesh));
            scene.addGameObject(std::move(spinningObj));
            std::cout << "Test scene loaded with spinning object" << std::endl;
        } else {
            std::cout << "Could not load model.obj, creating default scene" << std::endl;
            createDefaultCube();
        }

        // there was a second obj mr president
        createSecondObject();
    }

    void createDefaultCube() {
        std::vector<Vertex> vertices = {
            // Front face
            {{-0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
            {{ 0.5f, -0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
            {{ 0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
            {{-0.5f,  0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
            // Back face
            {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
            {{-0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
            {{ 0.5f,  0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},
            {{ 0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}}
        };

        std::vector<unsigned int> indices = {
            0, 1, 2, 2, 3, 0,  // Front
            4, 5, 6, 6, 7, 4,  // Back
            4, 0, 3, 3, 5, 4,  // Left
            1, 7, 6, 6, 2, 1,  // Right
            3, 2, 6, 6, 5, 3,  // Top
            4, 7, 1, 1, 0, 4   // Bottom
        };

        auto cube = std::make_unique<SpinningObject>();
        cube->setMesh(std::make_unique<Mesh>(vertices, indices));
        scene.addGameObject(std::move(cube));
        std::cout << "Created default cube" << std::endl;
    }

    void createSecondObject() {
        // Create pyramid
        std::vector<Vertex> vertices = {
            {{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
            {{ 0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
            {{ 0.5f, -0.5f,  0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
            {{-0.5f, -0.5f,  0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
            {{ 0.0f,  0.5f,  0.0f}, {0.0f, 1.0f, 0.0f}, {0.5f, 0.5f}}
        };

        std::vector<unsigned int> indices = {
            0, 1, 2, 2, 3, 0,  // Base
            0, 4, 1, 1, 4, 2,  // Sides
            2, 4, 3, 3, 4, 0
        };

        auto pyramid = std::make_unique<SpinningObject>();
        pyramid->spinSpeed = Math::Vec3(0.5f, -0.8f, 0.3f);
        pyramid->transform.position = Math::Vec3(2.5f, 0.0f, 0.0f);
        pyramid->setMesh(std::make_unique<Mesh>(vertices, indices));
        scene.addGameObject(std::move(pyramid));
        std::cout << "Created pyramid object" << std::endl;
    }

    void run() {
        std::cout << "Starting main loop with ASCII filter..." << std::endl;

        while (!glfwWindowShouldClose(window)) {
            Time::update();

            // Input processing
            if (Input::getKey(GLFW_KEY_ESCAPE)) {
                glfwSetWindowShouldClose(window, true);
            }

            if (Input::getKeyDown(GLFW_KEY_T)) {
                asciiMode = !asciiMode;
                std::cout << "ASCII mode: " << (asciiMode ? "ON" : "OFF") << std::endl;
            }

            // Character size adjustment
            static float charSize = 8.0f;
            if (Input::getKey(GLFW_KEY_EQUAL) || Input::getKey(GLFW_KEY_KP_ADD)) {
                charSize = std::max(4.0f, charSize - 0.2f);
                asciiFilter->setCharacterSize(charSize);
            }
            if (Input::getKey(GLFW_KEY_MINUS) || Input::getKey(GLFW_KEY_KP_SUBTRACT)) {
                charSize = std::min(32.0f, charSize + 0.2f);
                asciiFilter->setCharacterSize(charSize);
            }

            camera.processInput(Time::deltaTime);
            camera.processMouseInput(Input::getMouseDelta());

            scene.update(Time::deltaTime);

            // Rendering
            if (asciiMode) {
                asciiFilter->beginRender();
                renderScene();
                asciiFilter->endRender();
            } else {
                glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                renderScene();
            }

            glfwSwapBuffers(window);
            glfwPollEvents();
            Input::update();
        }
    }

    void renderScene() {
        defaultShader->use();
        defaultShader->setMat4("view", camera.getViewMatrix());
        defaultShader->setMat4("projection", camera.getProjectionMatrix(static_cast<float>(WIDTH) / HEIGHT));
        defaultShader->setVec3("viewPos", camera.position);
        defaultShader->setVec3("lightPos", Math::Vec3(2.0f, 4.0f, 2.0f));
        defaultShader->setVec3("lightColor", Math::Vec3(1.0f, 1.0f, 1.0f));
        defaultShader->setVec3("objectColor", Math::Vec3(0.8f, 0.6f, 0.4f));

        scene.render(*defaultShader);
    }

    void shutdown() {
        glfwTerminate();
        std::cout << "ASCII Graphics Engine shut down" << std::endl;
    }

private:
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
        glViewport(0, 0, width, height);
    }
};

// main
int main() {
    GraphicsEngine engine;

    if (!engine.initialize()) {
        return -1;
    }

    engine.loadTestScene();
    engine.run();
    engine.shutdown();

    return 0;
}