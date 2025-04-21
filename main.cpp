#include "updater.hpp"
#include "generator.hpp"
#include "particle.hpp"
#include <sycl/sycl.hpp>
#include <iostream>
#include "include/raylib.h"
#include "vector_cpu.hpp"
#include "parallel_sycl_sorting.hpp"
#include "math.hpp"
#include <cmath> // For M_PI if available, otherwise define PI
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void simpl_sort(size_t *data, size_t size)
{
    size_t thread_count = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(thread_count);
    size_t index = 0;
    std::mutex mtx;
    for(size_t i = 0; i < thread_count; i++)
    {
        threads[i] = std::thread([=, &data, &index, &mtx]() {
            for(size_t j = i; j < size; j+= thread_count)
            {
                if(data[j] != SIZE_MAX)
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    if(data[index] != SIZE_MAX)
                    {
                        index++;
                    }
                    std::swap(data[index], data[j]);
                    index++;
                }

            }
        });
    }
    for(size_t i = 0; i < thread_count; i++)
    {
        threads[i].join();
    }
}

void emit(double dt, Particle_system &p, size_t m_emitRate)
{
    if (p.m_countAlive >= p.size) return; // No more particles to emit
    if (m_emitRate <= 0) return;              // No emission rate

    const size_t maxNewParticles = static_cast<size_t>(dt*m_emitRate);
    const size_t count_start = p.m_countAlive;
    const size_t count_end = std::min(count_start + maxNewParticles, p.size -1);
    if((count_end - count_start) <= 0) return; 

    // p.q.submit([&](sycl::handler &h){
    //     auto w_acc = w;
    //     h.parallel_for(sycl::range<1>(count_end - count_start), [=](sycl::id<1> idx_d){
    //         size_t idx = idx_d.get(0);
    //         w_acc[idx] = count_start + idx;
    //     });
    // }).wait();
    // p.q.wait();
    // std::cout << "here -1\n";
    // p.wake(count_end - count_start);
    Gen gen;
    gen.generate(p, count_end - count_start);
    // std::cout << "here -2\n";
}

void draw(Image &im, Texture2D &tex, sycl::vec<char, 4> *color, size_t width, size_t hieght, Particle_system &p, sycl::queue &q)
{ 
    (void)p;
    q.submit([&](sycl::handler &h){
        auto acc = color;
        h.parallel_for(sycl::range<1>(width*hieght), [=](sycl::id<1> idx_d){
            size_t idx = idx_d.get(0);
            acc[idx] = sycl::vec<char, 4>{0, 0, 0, 255};
        });
    }).wait();
    q.wait();

    Mat4x4 proj;
    Mat4x4 view;
    // Mat4x4 model;

    // model.setTranslation(sycl::vec<float, 3>{0.0f, 0.0f, 0.0f});
    // model.setScale(sycl::vec<float, 3>{1.0f, 1.0f, 1.0f});
    // model.setRotation(sycl::vec<float, 3>{0.0f, 0.0f, 0.0f});

    view.setViewMatrix(sycl::vec<float, 3>{-100.0f, -100.0f, 100.0f}, sycl::vec<float, 3>{0.0f, 0.0f, 0.0f}, sycl::vec<float, 3>{0.0f, 1.0f, 0.0f});
    
    float fov_rad = 90.0f * (M_PI / 180.0f);
    proj.setProjectionMatrix(fov_rad, static_cast<float>(width) / hieght, 0.01f, 10000.0f);

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
                    acc_col[pixelIndex] |= acc[idx].col.convert<char>(); // Draw particle color
                }
            }
        });
    }).wait();
    q.wait();
    
    q.copy<sycl::vec<char, 4>>(color, (sycl::vec<char, 4>*)im.data, width*hieght);
    q.wait();
    UpdateTexture(tex, im.data);
    DrawTexture(tex, 0, 0, WHITE);
}

int main()
{

    Particle_system system(1000000);
    
    EulerUpdater eu;

    InitWindow(0, 0, "Getting Pissed On Simulator");
    int screenWidth = GetMonitorWidth(0);
    int screenHeight = GetMonitorHeight(0);
    SetWindowSize(screenWidth, screenHeight);
    std::cout<< "screenWidth: " << screenWidth << ", screenHeight: " << screenHeight << "\n";
    SetWindowState(FLAG_FULLSCREEN_MODE);
    Camera2D cam;
    cam.target = (Vector2){ -screenWidth/2.0f, -screenHeight/2.0f};
    cam.offset = (Vector2){ screenWidth/2.0f, screenHeight/2.0f };
    cam.rotation = 0.0f;
    cam.zoom = 1.0f;
    Image canvas = GenImageColor(screenWidth, screenHeight, {255, 255, 255, 255});
    sycl::vec<char, 4> *color = sycl::malloc_device<sycl::vec<char, 4>>(screenWidth*screenHeight, system.q);
    ImageFormat(&canvas, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
    Texture2D tex = LoadTextureFromImage(canvas);
    // size_t *to_kill = sycl::malloc_device<size_t>(system.size, system.q);
    // size_t *to_kill_host = sycl::malloc_host<size_t>(system.size, system.q);
    // const size_t thread_count = system.q.get_device().get_info<sycl::info::device::max_compute_units>() * 128;
    // size_t *kill_count_local = sycl::malloc_shared<size_t>(thread_count, system.q);
    // size_t *w = sycl::malloc_device<size_t>(system.size, system.q);

    while (!WindowShouldClose())
    {
        double dt = GetFrameTime();
        ClearBackground(RAYWHITE);
        draw(canvas, tex, color, screenWidth, screenHeight, system, system.q);
        eu.update(dt, system);
        emit(dt, system, 30000);
        BeginDrawing();
        DrawText("Particle System", 10, 10, 20, DARKGRAY);
        DrawText("Press ESC to exit", 10, 30, 20, DARKGRAY);
        DrawText(TextFormat("Alive particles : %d", system.m_countAlive), 10, 50, 20, DARKGRAY);
        DrawText(TextFormat("Total particles: %d", system.size), 10, 70, 20, DARKGRAY);
        DrawText(TextFormat("FPS: %d", GetFPS()), 10, 90, 20, DARKGRAY);
        EndDrawing();
    }
    // sycl::free(to_kill, system.q);
    // sycl::free(kill_count_local, system.q);
    // // sycl::free(w, system.q);
    // sycl::free(to_kill_host, system.q);
    sycl::free(color, system.q);
    UnloadImage(canvas);
    UnloadTexture(tex);

    CloseWindow();
}