#include "particle.hpp"
#include "./include/raylib.h"
#include "math.hpp"

class MyCamera
{
private:
    MyCamera() = default;
public:

    MyCamera(sycl::vec<float,3> eye, sycl::vec<float,3> lookat, sycl::vec<float,3> upVector)
        : m_eye(std::move(eye))
        , m_lookAt(std::move(lookat))
        , m_upVector(std::move(upVector))
    {
        UpdateViewMatrix();
    }

    Mat4x4 GetViewMatrix() const { return m_viewMatrix; }
    sycl::vec<float,3> GetEye() const { return m_eye; }
    sycl::vec<float,3> GetUpVector() const { return m_upVector; }
    sycl::vec<float,3> GetLookAt() const { return m_lookAt; }

    // MyCamera forward is -z
    sycl::vec<float,3> GetViewDir() const { 
        sycl::vec<float, 4> res = -(m_viewMatrix.transpose()).m[2]; 
        return sycl::vec<float, 3>{res.x(), res.y(), res.z()}; 
    }
    sycl::vec<float,3> GetRightVector() const { 
        sycl::vec<float, 4> res = (m_viewMatrix.transpose()).m[0]; 
        return sycl::vec<float, 3>{res.x(), res.y(), res.z()}; 
    }

    void SetCameraView(sycl::vec<float,3> eye, sycl::vec<float,3> lookat, sycl::vec<float,3> up)
    {
        m_eye = std::move(eye);
        m_lookAt = std::move(lookat);
        m_upVector = std::move(up);
        UpdateViewMatrix();
    }

    void UpdateViewMatrix()
    {
        // Generate view matrix using the eye, lookAt and up vector
        m_viewMatrix.setViewMatrix(m_eye, m_lookAt, m_upVector);
    }
    Mat4x4 m_viewMatrix;
    sycl::vec<float,3> m_eye; // MyCamera position in 3D
    sycl::vec<float,3> m_lookAt; // Point that the camera is looking at
    sycl::vec<float,3> m_upVector; // Orientation of the camera
};

class Renderer
{
private:
    Mat4x4 proj;
    // Mat4x4 view;
    size_t width;
    size_t hieght;
    MyCamera camera;
    float radius = 100.0f;

public:
    Renderer(size_t w, size_t h): proj(), width(w), hieght(h), camera(sycl::vec<float, 3>{-100.0f, -100.0f, 100.0f}, sycl::vec<float, 3>{0.0f, 0.0f, 0.0f}, sycl::vec<float, 3>{0.0f, 1.0f, 0.0f}) {
        float fov_rad = 90.0f * (M_PI / 180.0f);
        proj.setProjectionMatrix(fov_rad, static_cast<float>(width) / hieght, 0.01f, 10000.0f);
        

    }
    ~Renderer() = default;



void draw(float dt, Image &im, Texture2D &tex, sycl::vec<unsigned char, 4> *color, size_t width, size_t hieght, Particle_system &p, sycl::queue &q)
{ 
    (void)p;
    q.submit([&](sycl::handler &h){
        auto acc = color;
        h.parallel_for(sycl::range<1>(width*hieght), [=](sycl::id<1> idx_d){
            size_t idx = idx_d.get(0);
            acc[idx] = sycl::vec<unsigned char, 4>{0, 0, 0, 255};
        });
    }).wait();
    q.wait();

    // Mat4x4 proj;
    // Mat4x4 view;
    // Mat4x4 model;
    double time = GetTime();
    // model.setTranslation(sycl::vec<float, 3>{0.0f, 0.0f, 0.0f});
    // model.setScale(sycl::vec<float, 3>{1.0f, 1.0f, 1.0f});
    // model.setRotation(sycl::vec<float, 3>{0.0f, 0.0f, 0.0f});
    radius += GetMouseWheelMove();
    if(radius < 1.0f) radius = 1.0f;
    float camX   = sin(time/10.0f) * radius;
    float camZ   = cos(time/10.0f) * radius;
    // float camY   = 
    // view.setViewMatrix(sycl::vec<float, 3>{camX, 0.0f, camZ}, sycl::vec<float, 3>{0.0f, 0.0f, 0.0f}, sycl::vec<float, 3>{0.0f, 1.0f, 0.0f});
    // Mat4x4 view = this->view;
    // Mat4x4 view = camera.GetViewMatrix();

    camera.SetCameraView(sycl::vec<float, 3>{camX, 0.0f, camZ}, sycl::vec<float, 3>{0.0f, 0.0f, 0.0f}, sycl::vec<float, 3>{0.0f, 1.0f, 0.0f});
    Mat4x4 view = camera.GetViewMatrix();
    Mat4x4 proj = this->proj;
    q.submit([&](sycl::handler &h){
        auto acc = p.m_particle;
        auto acc_col = color;
        h.parallel_for(sycl::range<1>(p.size), [=](sycl::id<1> idx_d){
            size_t idx = idx_d.get(0);
            if(acc[idx].alive == false)
            {
                return;
            }
            sycl::vec<float, 4> pos_world = acc[idx].pos;
            sycl::vec<float, 4> pos_clip = proj * view * pos_world;

            // Perspective divide (already done in matrix multiplication if w != 1)
            // If w is 0, the point is at infinity, skip it
            if (pos_clip.w() == 0.0f) return;
            // pos_clip.x() *= width / 5;
            // pos_clip.y() *= hieght / 5; // Corrected to use pos_clip.w() for height
            
            // Assuming perspective divide happened in operator*:
            sycl::vec<float, 3> pos_ndc = {pos_clip.x(), pos_clip.y(), pos_clip.z()};

            // Check if the point is within the clip volume (NDC range)
            if (pos_ndc.x() >= -1.0f && pos_ndc.x() <= 1.0f &&
                pos_ndc.y() >= -1.0f && pos_ndc.y() <= 1.0f &&
                pos_ndc.z() >= -1.0f && pos_ndc.z() <= 1.0f) // Check Z as well
            {
                // Map NDC to screen coordinates
                float screenX = (pos_ndc.x() + 1.0f) * 0.5f * width;
                float screenY = (1.0f - pos_ndc.y()) * 0.5f * hieght; // Y is often inverted
                // float screenX = ((pos_world.x() + width) / 2.0f); // X is often inverted
                // float screenY = ((pos_world.y() + hieght) / 2.0f); // Y is often inverted
                // Check if screen coordinates are within image bounds
                if (screenX >= 0 && screenX < width && screenY >= 0 && screenY < hieght)
                {
                    long pixelIndex = static_cast<long>(screenY) * width + static_cast<long>(screenX);
                    // if(acc_col[pixelIndex].x() == 0 && acc_col[pixelIndex].y() == 0 && acc_col[pixelIndex].z() == 0)
                    // {
                    //     acc_col[pixelIndex] = acc[idx].col.convert<char>(); // Draw particle color
                    // }
                    // else
                    // {
                    //     acc_col[pixelIndex] |=  acc[idx].col.convert<char>(); // Blend with existing color
                    // }
                    // sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
                    acc_col[pixelIndex] = sycl::mix(acc_col[pixelIndex].convert<float>(), acc[idx].col, sycl::float4(0.5f)).convert<unsigned char>(); // Draw particle color
                    // sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
                }
            }
        });
    }).wait();
    q.wait();
    
    q.copy<sycl::vec<unsigned char, 4>>(color, (sycl::vec<unsigned char, 4>*)im.data, width*hieght);
    q.wait();
    UpdateTexture(tex, im.data);
    DrawTexture(tex, 0, 0, WHITE);
}

};