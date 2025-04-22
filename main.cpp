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
#include "renderer.hpp"
#include "input.hpp"
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

void emit(double dt, Particle_system &p, Gen gen, size_t m_emitRate)
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
    gen.generate(p, count_end - count_start);
    // std::cout << "here -2\n";
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
    Renderer renderer(screenWidth, screenHeight);
    Gen gen;
    MyInput input;
    size_t emmit_count = 30000;
    // size_t *to_kill = sycl::malloc_device<size_t>(system.size, system.q);
    // size_t *to_kill_host = sycl::malloc_host<size_t>(system.size, system.q);
    // const size_t thread_count = system.q.get_device().get_info<sycl::info::device::max_compute_units>() * 128;
    // size_t *kill_count_local = sycl::malloc_shared<size_t>(thread_count, system.q);
    // size_t *w = sycl::malloc_device<size_t>(system.size, system.q);

    while (!WindowShouldClose())
    {
        BeginDrawing();
        double dt = GetFrameTime();
        ClearBackground(RAYWHITE);
        renderer.draw(dt, canvas, tex, color, screenWidth, screenHeight, system, system.q);
        eu.update(dt, system);
        emit(dt, system, gen, emmit_count);
        DrawText("Particle System", 10, 10, 20, DARKGRAY);
        DrawText("Press ESC to exit", 10, 30, 20, DARKGRAY);
        DrawText(TextFormat("Alive particles : %d", system.m_countAlive), 10, 50, 20, DARKGRAY);
        DrawText(TextFormat("Total particles: %d", system.size), 10, 70, 20, DARKGRAY);
        DrawText(TextFormat("FPS: %d", GetFPS()), 10, 90, 20, DARKGRAY);
        input.processInput(gen, eu, emmit_count);
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