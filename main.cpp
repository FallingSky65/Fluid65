// references: https://matthias-research.github.io/pages/publications/sca03.pdf

#include <climits>
#include <cmath>
#include <opencv2/videoio.hpp>
#include <random>
#include <raylib-cpp.hpp>
#include <raylib.h>
#include <raymath.h>
#include <vector>
#include <opencv2/opencv.hpp>

const int screenWidth = 1080;
const int screenHeight = 1080;

const int numParticles = 1000;

const float sphereSize = 40.0;

const float sampleRadius = 12.0f;

const float restDensity = 0.0001f;

const float gasConstant = 100.0f;

const float viscosity = 0.01f;

const float surfaceTension = 50.0f;

struct Particle {
    Vector3 position;
    Vector3 velocity;
    Vector3 acceleration;
    float mass;

    float density;
    float pressure;
    Vector3 colorGradient;
};

Particle particles[numParticles];

cv::Mat textureToMat(Texture2D texture) {
    Image image = LoadImageFromTexture(texture);
    Color* pixels = LoadImageColors(image);

    cv::Mat mat(image.height, image.width, CV_8UC4);
    
    for (int y = 0; y < image.height; y++) {
        for (int x = 0; x < image.width; x++) {
            Color pixel = pixels[y * image.width + x];
            mat.at<cv::Vec4b>(y, x) = cv::Vec4b(pixel.b, pixel.g, pixel.r, pixel.a);
        }
    }

    UnloadImageColors(pixels);
    UnloadImage(image);

    // Convert BGRA to BGR (optional, if you don't need the alpha channel)
    cv::Mat mat_bgr;
    cv::cvtColor(mat, mat_bgr, cv::COLOR_BGRA2BGR);

    return mat_bgr;
}

float W_poly6(Vector3 r, float h) {
    float magnitude = Vector3Length(r);
    if (0 <= magnitude && magnitude <= h) {
        return (315.f/(64.f*PI*powf(h, 9.f)))*powf(h*h - magnitude*magnitude, 3.f);
    } else {
        return 0.f;
    }
}

float W_poly6_Gradient(Vector3 r, float h) {
    float magnitude = Vector3Length(r);
    if (0 <= magnitude && magnitude <= h) {
        return (315.f/(64.f*PI*powf(h, 9.f)))*(-2.f*magnitude)*3.f*powf(h*h - magnitude*magnitude, 2.f);
    } else {
        return 0.f;
    }
}

float W_poly6_Laplacian(Vector3 r, float h) {
    float magnitude = Vector3Length(r);
    if (0 <= magnitude && magnitude <= h) {
        return (315.f/(64.f*PI*powf(h, 9.f)))*6.f*(h*h - magnitude*magnitude)*(5*magnitude*magnitude - h*h);
    } else {
        return 0.f;
    }
}


float W_viscosity_Laplacian(Vector3 r, float h) {
    float magnitude = Vector3Length(r);
    if (0 <= magnitude && magnitude <= h) {
        return (45.f/(PI*powf(h, 6.f)))*(h - magnitude);
    } else {
        return 0.f;
    }
}

float W_spiky(Vector3 r, float h) {
    float magnitude = Vector3Length(r);
    if (0 <= magnitude && magnitude <= h) {
        return (15.f/(PI*powf(h, 6.f)))*powf(h - magnitude, 3.f);
    } else {
        return 0.f;
    }
}

float W_spiky_Gradient(Vector3 r, float h) {
    float magnitude = Vector3Length(r);
    if (0 <= magnitude && magnitude <= h) {
        return (15.f/(PI*powf(h, 6.f)))*(-1.f)*3.f*powf(h - magnitude, 2.f);
    } else {
        return 0.f;
    }
}

float sampleDensity(Particle particle) {
    float density = 0.0f;
    for (int i = 0; i < numParticles; i++) {
        density += particles[i].mass * W_poly6(Vector3Subtract(particle.position, particles[i].position), sampleRadius);
    }
    return density;
}

float samplePressure(Particle particle) {
    float pressure = gasConstant*(particle.density - restDensity);
    return pressure;
}

float sampleColor(Particle particle) {
    float color = 0.0f;
    for (int i = 0; i < numParticles; i++) {
        color += particles[i].mass * (1.0f/particles[i].density) * W_poly6(Vector3Subtract(particle.position, particles[i].position), sampleRadius);
    }
    return color;
}

Vector3 sampleColorGradient(Particle particle) {
    Vector3 colorGradient = Vector3Zero();
    for (int i = 0; i < numParticles; i++) {
        Vector3 r = Vector3Subtract(particles[i].position, particle.position);
        colorGradient = Vector3Add(colorGradient, Vector3Scale(Vector3Normalize(r), particles[i].mass * (1.0f/particles[i].density) * W_poly6_Gradient(r, sampleRadius)));
    }
    return colorGradient;
}

Vector3 sampleColorDivergence(Particle particle) {
    Vector3 colorDivergence = Vector3Zero();
    for (int i = 0; i < numParticles; i++) {
        colorDivergence = Vector3Add(colorDivergence, Vector3Scale(particles[i].colorGradient, particles[i].mass * (1.0f/particles[i].density) * W_poly6_Laplacian(Vector3Subtract(particle.position, particles[i].position), sampleRadius)));
    }
    return colorDivergence;
}

Vector3 samplePressureForce(Particle particle) {
    Vector3 pressureForce = Vector3Zero();
    for (int i = 0; i < numParticles; i++) {
        Vector3 r = Vector3Subtract(particle.position, particles[i].position);
        pressureForce = Vector3Subtract(pressureForce, Vector3Scale(Vector3Normalize(r), particles[i].mass * (particle.pressure + particles[i].pressure)/(2.0f * particles[i].density) * W_spiky_Gradient(r, sampleRadius)));
    }
    return pressureForce;
}

Vector3 sampleViscosityForce(Particle particle) {
    Vector3 viscosityForce = Vector3Zero();
    for (int i = 0; i < numParticles; i++) {
        viscosityForce = Vector3Add(viscosityForce, Vector3Scale(Vector3Subtract(particles[i].velocity, particle.velocity), viscosity * particles[i].mass * (1.f/particles[i].density) * W_viscosity_Laplacian(Vector3Subtract(particle.position, particles[i].position), sampleRadius)));
    }
    return viscosityForce;
}

Vector3 sampleSurfaceTractionForce(Particle particle) {
    Vector3 surfaceTractionForce = Vector3Scale(Vector3Normalize(particle.colorGradient), -surfaceTension*Vector3Length(sampleColorDivergence(particle)));
    return surfaceTractionForce;
}

void updateParticles(float deltaTime) {
    for (int i = 0; i < numParticles; i++) particles[i].density = sampleDensity(particles[i]);
    for (int i = 0; i < numParticles; i++) particles[i].pressure = samplePressure(particles[i]);
    for (int i = 0; i < numParticles; i++) particles[i].colorGradient = sampleColorGradient(particles[i]);

    for (int i = 0; i < numParticles; i++) {
        Vector3 netForce = Vector3Add(samplePressureForce(particles[i]), Vector3Scale({0.0f, 1.0f, 0.0f}, -0.1f*particles[i].mass));
        netForce = Vector3Add(netForce, sampleViscosityForce(particles[i]));
        netForce = Vector3Add(netForce, sampleSurfaceTractionForce(particles[i]));
        particles[i].acceleration = Vector3Scale(netForce, 1.0f/particles[i].density);
    }
    for (int i = 0; i < numParticles; i++) {
        particles[i].velocity = Vector3Add(particles[i].velocity, Vector3Scale(particles[i].acceleration, deltaTime));
        if (Vector3Length(particles[i].position) >= sphereSize - 1.0f) {
            //particles[i].position = Vector3Scale(Vector3Normalize(particles[i].position), sphereSize-1.0f);
            //particles[i].velocity = Vector3Scale(Vector3Normalize(particles[i].position), sphereSize*-0.01f);
            if (Vector3DotProduct(particles[i].position, particles[i].velocity) > 0.0f) particles[i].velocity = Vector3Scale(Vector3Reflect(particles[i].velocity, Vector3Negate(Vector3Normalize(particles[i].position))), 0.8f);
        }
        particles[i].position = Vector3Add(particles[i].position, Vector3Scale(particles[i].velocity, deltaTime));
    }
}

int main() {

    int codec = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
    cv::VideoWriter videoWriter;

    SetTraceLogLevel(LOG_WARNING);
    raylib::Window window(screenWidth, screenHeight, "Fluid65");

    raylib::Camera3D camera({0.0f, 0.0f, sphereSize*3}, {0.0f, 0.0f, 0.0f}, {0.0f, -1.0f, 0.0f}, 45.0f, CAMERA_PERSPECTIVE);

    raylib::Mesh sphere = GenMeshSphere(1.0f, 6, 12);    
    
    std::default_random_engine generator(GetRandomValue(0, INT_MAX));
    //std::uniform_real_distribution<float> distribution(-50.0, 50.0);
    std::normal_distribution<float> distribution(0.0, 5.0);

    for (int i = 0; i < numParticles; i++) {
        particles[i].position = {distribution(generator), distribution(generator), distribution(generator)};
        particles[i].velocity = Vector3Zero();
        particles[i].mass = 1.0f;
    }

    Shader shader = LoadShader("shaders/vert.glsl", "shaders/frag.glsl");
    shader.locs[SHADER_LOC_VECTOR_VIEW] = GetShaderLocation(shader, "viewPos");
    
    // Ambient light level (some basic lighting)
    int ambientLoc = GetShaderLocation(shader, "ambient");
    SetShaderValue(shader, ambientLoc, (float[4]){ 0.1f, 0.1f, 0.1f, 1.0f }, SHADER_UNIFORM_VEC4);

    Material material = LoadMaterialDefault();
    material.shader = shader;
    material.maps[MATERIAL_MAP_DIFFUSE].color = BLUE;

    raylib::RenderTexture2D canvas(screenWidth, screenHeight);

    SetTargetFPS(30);

    while (!window.ShouldClose())
    {
        UpdateCamera(&camera, CAMERA_THIRD_PERSON);
        SetMousePosition(0, 0);

        SetShaderValue(shader, shader.locs[SHADER_LOC_VECTOR_VIEW], &camera.position.x, SHADER_UNIFORM_VEC3);

        updateParticles(0.04f);

        canvas.BeginMode();
        {
            ClearBackground(BLACK);
            camera.BeginMode();
            {
                for (int i = 0; i < numParticles; i++) sphere.Draw(material, MatrixTranslate(particles[i].position.x, particles[i].position.y, particles[i].position.z));
                //for (int i = 0; i < numParticles; i++) DrawCylinderEx(particles[i].position, Vector3Add(particles[i].position, Vector3Scale(particles[i].acceleration, 0.03f)), 0.05f, 0.05f, 4, RED);
                //DrawMeshInstanced(sphere, material, transforms.data(), numParticles);
                //DrawCubeWires(Vector3Zero(), 100.0f, 100.0f, 100.0f, RED);
                DrawSphereWires(Vector3Zero(), sphereSize, 24, 48, GRAY);
            }
            camera.EndMode();
            //raylib::DrawText(TextFormat("density = %.5f", particles[0].density), 10, 40, 20, WHITE);
        }
        canvas.EndMode();

        window.BeginDrawing();
        {
            canvas.GetTexture().Draw();
            window.DrawFPS();
        }
        window.EndDrawing();

        cv::Mat frame = textureToMat(canvas.texture);
        if (!videoWriter.isOpened()) {
            cv::Size frameSize = frame.size();
            videoWriter.open("renders/render.mp4", codec, 30, frameSize, true);
            if (!videoWriter.isOpened()) TraceLog(LOG_WARNING, "Failed to open videoWriter");
        }

        videoWriter.write(frame);
    }

    videoWriter.release();
    UnloadShader(shader);
    //UnloadMaterial(material);
}
