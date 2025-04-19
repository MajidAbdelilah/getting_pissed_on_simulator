#include "updater.hpp"
#include "generator.hpp"
#include "particle.hpp"
#include <sycl/sycl.hpp>
#include <iostream>
#include "include/raylib.h"
#include "vector_cpu.hpp"
#include "parallel_sycl_sorting.hpp"


void simpl_sort(size_t *data, size_t size)
{
    size_t thread_count = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(thread_count);
    size_t index = 0;
    std::mutex mtx;
    for(size_t i = 0; i < thread_count; i++)
    {
        // index[i] = i;
        threads[i] = std::thread([=, &data, &index, &mtx]() {
            // size_t start = i * (size / thread_count);
            // size_t end = (i + 1) * (size / thread_count);
            for(size_t j = i; j < size; j+= thread_count)
            {
                if(data[j] < SIZE_MAX)
                {
                    std::lock_guard<std::mutex> lock(mtx);
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
    // size_t index_2 = 0;
    // size_t thread_counter = 0;
    // for(size_t i = 0; i < size; i++)
    // {

    //     if(data[i] < SIZE_MAX && i != index_2) // Check if data[i] is less than SIZE_MAX
    //     {
    //         std::swap(data[index_2], data[i]);
    //         index_2++;
    //     }else if(i == index_2)
    //     {
    //         index_2++;
    //     }
    //     else if(data[i] == SIZE_MAX)
    //     {
    //         i += thread_count - thread_counter - 1;
    //         thread_counter = 0;
    //         continue;
    //     }
    //     thread_counter++;
    //     if(thread_counter >= thread_count)
    //     {
    //         thread_counter = 0;
    //     }
    // }
    //         }
    //     });
    // }
    // for(size_t i = 0; i < size; i++)
    // {
    //     if(data[i] < SIZE_MAX)
    //     {
    //         std::swap(data[index], data[i]);
    //         index++;
    //     }
    // }
}

void emit(double dt, Particle_system &p, size_t m_emitRate, size_t *w)
{
    if (p.m_countAlive >= p.size) return; // No more particles to emit
    if (m_emitRate <= 0) return;              // No emission rate

    const size_t maxNewParticles = static_cast<size_t>(dt*m_emitRate);
    const size_t startId = p.m_countAlive;
    const size_t endId = std::min(startId + maxNewParticles, p.size -1);
    if((endId - startId) <= 0) return; 
    // std::cout << "startId: " << startId << ", endId: " << endId << "\n";
    // std::vector<size_t> w(endId - startId);
    // (fill_with_index(startId), endId - startId);
    // sycl::buffer<size_t, 1> w_buf(w);
    p.q.submit([&](sycl::handler &h){
        auto w_acc = w;
        h.parallel_for(sycl::range<1>(endId - startId), [=](sycl::id<1> idx_d){
            size_t idx = idx_d.get(0);
            w_acc[idx] = startId + idx;
        });
    }).wait();
    p.q.wait();
    // w.resize(endId - startId);
    // // std::cout << "startId: " << startId << ", endId: " << endId << "\n";
    
    
    // for (size_t i = startId; i < endId; i++)
    // {
    //     w[i - startId] = i;
    // }
    p.wake(w, endId - startId);
    Gen gen;
    // BasicVelGen bvg;
    // BasicColorGen bc;
    // BasicTimeGen btg;
    gen.generate(p, startId, endId);
    // bvg.generate(p, startId, endId);
    // bc.generate(p, startId, endId);
    // btg.generate(p, startId, endId);

    

    // for(size_t i = startId; i < endId; i++)
    // {
    //     p.m_particle[i].alive = true;
    // }
    // p.m_countAlive += endId - startId;
    // w.clear();
}

void draw(Image &im, Texture2D &tex, sycl::vec<char, 4> *color, size_t width, size_t hieght, Particle_system &p, sycl::queue &q)
{ 
    (void)p;
    // sycl::buffer<Color, 1> buf(((Color*)im.data), sycl::range<1>(im.width*im.height));
    q.submit([&](sycl::handler &h){
        auto acc = color;
        h.parallel_for(sycl::range<1>(width*hieght), [=](sycl::id<1> idx_d){
            size_t idx = idx_d.get(0);
            acc[idx] = sycl::vec<char, 4>{0, 0, 0, 255};
        });
    }).wait();
    q.wait();

    // sycl::buffer<Particle, 1> buf_p(p.m_particle);
    
    q.submit([&](sycl::handler &h){
        auto acc = p.m_particle;
        auto acc_col = color;
        h.parallel_for(sycl::range<1>(p.m_countAlive), [=](sycl::id<1> idx_d){
            size_t idx = idx_d.get(0);
            if(acc[idx].alive)
            {
                long x = (long)(acc[idx].pos.x());
                long y = (long)(acc[idx].pos.y());
                x+= width;
                y+= hieght;
                if((x < (long)width && x >= 0) 
                && (y < (long)hieght && y >= 0))
                {
                    acc_col[y * im.width + x] = sycl::vec<char, 4>{255, 255, 255, 255};
                }
            }
        });
    }).wait();
    q.wait();
    
    q.copy<sycl::vec<char, 4>>(color, (sycl::vec<char, 4>*)im.data, width*hieght);
    q.wait();
    // for(size_t i = 0; i < p.m_countAlive; ++i)
    // {
    //     if(p.m_particle[i].alive)
    //     {
    //         size_t x = j % width;
    //         size_t y = i;
    //         ((Color*)im.data)[y*im.width + x] = (Color){ 0, 0, 0, 255 };
    //     }    
    // }
    
    

    // for(size_t i = 0; i < p.m_countAlive; ++i)
    // {
    //     if(p.m_particle[i].alive)
    //     {
    //         size_t x = static_cast<size_t>(p.m_particle[i].pos.x());
    //         size_t y = static_cast<size_t>(p.m_particle[i].pos.y());
    //         x+= width;
    //         y+= hieght;
    //         if((x < width && x >= 0) 
    //         && (y < hieght && y >= 0))
    //         {
    //             // std::cout << "x: " << x << ", y: " << y << "\n";
    //             // std::cout << "color: " << p.m_particle[i].col.x() << ", " << p.m_particle[i].col.y() << ", " << p.m_particle[i].col.z() << ", " << p.m_particle[i].col.w() << "\n";
    //             ImageDrawPixel(&im, x, y, WHITE);
    //         }
    //     }
    // }
    // UnloadImageColors(color)
    UpdateTexture(tex, im.data);
    // UnloadTexture(tex);
    // tex =  LoadTextureFromImage(im);
    DrawTexture(tex, -width, -hieght, {255, 255, 255, 255});
    
}

int main()
{

    Particle_system system(1000000);
    
    EulerUpdater eu;

    int screenWidth = 1900;
    int screenHeight = 1000;
    InitWindow(screenWidth, screenHeight, "Particle System");
    // SetTargetFPS(60);
    Camera2D cam;
    cam.target = (Vector2){ -screenWidth/2.0f, -screenHeight/2.0f};
    cam.offset = (Vector2){ screenWidth/2.0f, screenHeight/2.0f };
    cam.rotation = 0.0f;
    cam.zoom = 1.0f;
    Image canvas = GenImageColor(screenWidth, screenHeight, {255, 255, 255, 255});
    sycl::vec<char, 4> *color = sycl::malloc_device<sycl::vec<char, 4>>(screenWidth*screenHeight, system.q);
    ImageFormat(&canvas, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
    Texture2D tex = LoadTextureFromImage(canvas);
    // Color *color = LoadImageColors(canvas);
    size_t *to_kill = sycl::malloc_device<size_t>(system.size, system.q);
    size_t *to_kill_host = sycl::malloc_host<size_t>(system.size, system.q);
    const size_t thread_count = system.q.get_device().get_info<sycl::info::device::max_compute_units>() * 128;
    size_t *kill_count_local = sycl::malloc_shared<size_t>(thread_count, system.q);
    size_t *w = sycl::malloc_device<size_t>(system.size, system.q);

    while (!WindowShouldClose())
    // while(true)
    {
        double dt = GetFrameTime();
        // std::cout << "here 0\n";
        // double dt = 1.0f/30.0f;
        // Update
        // system.kill({ 0, 1, 2, 3, 4 });
        emit(dt, system, 70000, w);
        eu.update(dt, system);
        std::cout << "here 1\n";
        BeginDrawing();
        BeginMode2D(cam);
        ClearBackground(RAYWHITE);
        DrawCircle(200, 200, 100, BLACK);
        draw(canvas, tex, color, screenWidth, screenHeight, system, system.q);
        if (system.m_countAlive > 0)
        {
            // to_kill.resize(system.m_countAlive);
            size_t kill_count_total = 0;
            std::cout << "here 2\n";
            // std::vector<std::thread> threads(thread_count);
            // std::cout << "thread count: " << thread_count << "\n";
            // std::mutex mutex;
            
            system.q.submit([&](sycl::handler &h){
                auto acc = system.m_particle;
                auto count_alive = system.m_countAlive;
                h.parallel_for(sycl::range<1>(thread_count), [=](sycl::id<1> idx_d){
                    size_t idx = idx_d.get(0);
                    kill_count_local[idx] = 0;
                    for(size_t j = idx; j < count_alive; j+= thread_count)
                    {
                        to_kill[j] = SIZE_MAX;
                        if (acc[j].alive == false || acc[j].time.x() < 0.0f){
                            to_kill[j] = j;
                            kill_count_local[idx]++;
                        }
                    }
                    
                });
            }).wait();
            system.q.wait();
            
                // std::accumulate(kill_count_local, kill_count_local + thread_count, 0, [&](size_t a, size_t b) {
                //     return a + b;
                // });
            for(size_t i = 0; i < thread_count; ++i)
            {
                kill_count_total += kill_count_local[i];
            }
            std::cout << "here 3\n";
            system.q.copy<size_t>(to_kill, to_kill_host, system.m_countAlive);
            system.q.wait();
            simpl_sort(to_kill_host, system.m_countAlive);
            system.q.copy<size_t>(to_kill_host, to_kill, system.m_countAlive);
            system.q.wait();
            std::cout << "here 4\n";
            // to_kill.resize(kill_count_total);
            // for(size_t i = 0; i < thread_count; ++i)
            // {
            //     // threads[i] = std::thread([&system, &to_kill, i, thread_count, &kill_count_total, &mutex](){
            //         size_t start = (system.m_countAlive / thread_count) * i;
            //         size_t end = (system.m_countAlive / thread_count) * (i + 1);
            //         if(i+1 == thread_count)
            //         {
            //             end = system.m_countAlive;
            //         }
            //         size_t kill_count_local = 1;
            //         for (size_t j = start; j < end; ++j)
            //         {
            //             to_kill[j] = SIZE_MAX;
            //             if (system.m_particle[j].alive == false || system.m_particle[j].time.x() < 0.0f){
            //                 to_kill[kill_count_local*i] = j;
            //                 kill_count_local++;
            //             }
            //         }
            //         std::cout << "thread: " << i << ", start: " << start << ", end: " << end << ", kill_count_local: " << kill_count_local  << " count alive: " << system.m_countAlive << "\n";

            //         // std::lock_guard<std::mutex> lock(mutex);
            //         kill_count_total += kill_count_local - 1;
            //     // });
            // }

            // std::cout << "here 3\n";
            // size_t kill_count[4] = {0};
            // for (size_t i = 0; i < ((system.m_countAlive/3) * 1); ++i)
            // {

            //     to_kill[i] = SIZE_MAX;
            //     if (system.m_particle[i].alive == false || system.m_particle[i].time.x() < 0.0f){
            //         to_kill[0 * ((system.m_countAlive/3)) + kill_count[0]] = i;
            //         kill_count[0]++;
            //     }
            // }for (size_t i = ((system.m_countAlive/3) * 1); i < ((system.m_countAlive/3) * 2); ++i)
            // {

            //     to_kill[i] = SIZE_MAX;
            //     if (system.m_particle[i].alive == false || system.m_particle[i].time.x() < 0.0f){
            //         to_kill[1 * ((system.m_countAlive/3)) + kill_count[1]] = i;
            //         kill_count[1]++;
            //     }
            // }for (size_t i = ((system.m_countAlive/3) * 2); i < ((system.m_countAlive/3) * 3); ++i)
            // {

            //     to_kill[i] = SIZE_MAX;
            //     if (system.m_particle[i].alive == false || system.m_particle[i].time.x() < 0.0f){
            //         to_kill[2 * ((system.m_countAlive/3)) + kill_count[2]] = i;
            //         kill_count[2]++;
            //     }
            // }
            // for(int i = 0; i < 3; ++i)
            // {
            //     kill_count_total += kill_count[i];
            // }
            // to_kill.resize(kill_count);
            // std::cout << "kill_count: " << kill_count_total << "\n";
            system.kill(to_kill, kill_count_total);
            std::cout << "here 5\n";
        }
        EndMode2D();
        DrawText("Particle System", 10, 10, 20, DARKGRAY);
        DrawText("Press ESC to exit", 10, 30, 20, DARKGRAY);
        DrawText(TextFormat("Alive particles : %d", system.m_countAlive), 10, 50, 20, DARKGRAY);
        DrawText(TextFormat("Total particles: %d", system.size), 10, 70, 20, DARKGRAY);
        DrawText(TextFormat("FPS: %d", GetFPS()), 10, 90, 20, DARKGRAY);
        EndDrawing();
    }
    sycl::free(to_kill, system.q);
    sycl::free(kill_count_local, system.q);
    sycl::free(w, system.q);
    sycl::free(to_kill_host, system.q);
    sycl::free(color, system.q);
    UnloadImage(canvas);
    // UnloadImageColors(color);
    UnloadTexture(tex);

    CloseWindow();
}