#include "updater.hpp"
#include "generator.hpp"
#include "particle.hpp"
#include <sycl/sycl.hpp>
#include <iostream>
#include "include/raylib.h"
#include "vector_cpu.hpp"

class fill_with_index
{
    public:
    size_t startId;
    fill_with_index(size_t start): startId(start){}
    size_t operator()(size_t &value, size_t idx)
    {
        (void)value;
        return startId + idx;
    }
};

void emit(double dt, Particle_system &p, size_t m_emitRate)
{
    if (p.m_countAlive >= p.size) return; // No more particles to emit
    if (m_emitRate <= 0) return;              // No emission rate

    const size_t maxNewParticles = static_cast<size_t>(dt*m_emitRate);
    const size_t startId = p.m_countAlive;
    const size_t endId = std::min(startId + maxNewParticles, p.size -1);
    if((endId - startId) <= 0) return; 
    // std::cout << "startId: " << startId << ", endId: " << endId << "\n";
    std::vector<size_t> w(endId - startId);
    // (fill_with_index(startId), endId - startId);
    sycl::buffer<size_t, 1> w_buf(w);
    p.q.submit([&](sycl::handler &h){
        auto w_acc = w_buf.get_access<sycl::access::mode::write>(h);
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
    p.wake(w);
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

void draw(Image &im, Texture2D &tex, size_t width, size_t hieght, Particle_system &p, sycl::queue &q)
{ 
    (void)p;
    sycl::buffer<Color, 1> buf(((Color*)im.data), sycl::range<1>(im.width*im.height));
    q.submit([&](sycl::handler &h){
        auto acc = buf.get_access<sycl::access::mode::write>(h);
        h.parallel_for(sycl::range<1>(width*hieght), [=](sycl::id<1> idx_d){
            size_t idx = idx_d.get(0);
            acc[idx] = BLACK;
        });
    }).wait();
    q.wait();

    // sycl::buffer<Particle, 1> buf_p(p.m_particle);
    
    q.submit([&](sycl::handler &h){
        auto acc = p.m_particle;
        auto acc_im = buf.get_access<sycl::access::mode::write>(h);
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
                    acc_im[y * im.width + x] = WHITE;
                }
            }
        });
    }).wait();
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
    // UnloadImageColors(color);
    UnloadTexture(tex);
    tex =  LoadTextureFromImage(im);
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
    ImageFormat(&canvas, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
    Texture2D tex = LoadTextureFromImage(canvas);
    Color *color = LoadImageColors(canvas);
    while (!WindowShouldClose())
    // while(true)
    {
        double dt = GetFrameTime();
        // std::cout << "here 0\n";
        // double dt = 1.0f/30.0f;
        // Update
        // system.kill({ 0, 1, 2, 3, 4 });
        emit(dt, system, 200000);
        eu.update(dt, system);
        // std::cout << "here 1\n";
        BeginDrawing();
        BeginMode2D(cam);
        ClearBackground(RAYWHITE);
        DrawCircle(200, 200, 100, BLACK);
        draw(canvas, tex, screenWidth, screenHeight, system, system.q);
        if (system.m_countAlive > 0)
        {
            std::vector<size_t> to_kill;
            to_kill.resize(system.m_countAlive);
            size_t kill_count_total = 0;
            const size_t thread_count = std::thread::hardware_concurrency();
            // std::cout << "here 2\n";
            std::vector<std::thread> threads(thread_count);
            // std::cout << "thread count: " << thread_count << "\n";
            // std::mutex mutex;
            
            std::vector<size_t> kill_count_local(thread_count);
            for(size_t i = 0; i < thread_count; ++i)
            {
                kill_count_local[i] = 0;
                // std::cout << "thread: " << i << "\n";
                threads[i] = std::thread([&system, &to_kill, i, thread_count, &kill_count_total, &kill_count_local](){
                    // std::cout << "here 2.1, thread: " << i << "\n";
                    // size_t start = (system.m_countAlive / thread_count) * i;
                    // size_t end = (system.m_countAlive / thread_count) * (i + 1);
                    // if(i+1 == thread_count)
                    // {
                    //     end = system.m_countAlive;
                    // }
                    // std::cout << "here 2.2, thread: " << i << ", start: " << start << ", end: " << end << "\n";
                    for (size_t j = i; j < system.m_countAlive; j+= thread_count)
                    {
                        to_kill[j] = SIZE_MAX;
                        if (system.m_particle[j].alive == false || system.m_particle[j].time.x() < 0.0f){
                            to_kill[j] = j;
                            kill_count_local[i]++;
                        }
                    }
                    // std::cout << "thread: " << i << ", start: " << start << ", end: " << end << ", kill_count_local: " << kill_count_local[i]  << " count alive: " << system.m_countAlive << "\n";

                    // std::lock_guard<std::mutex> lock(mutex);
                });
            }
            
            for(size_t i = 0; i < thread_count; ++i)
            {
                threads[i].join();
            }
            for(size_t i = 0; i < thread_count; ++i)
            {
                kill_count_total += kill_count_local[i];
            }
            std::sort(to_kill.begin(), to_kill.end());
            to_kill.resize(kill_count_total);
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

        }
        EndMode2D();
        DrawText("Particle System", 10, 10, 20, DARKGRAY);
        DrawText("Press ESC to exit", 10, 30, 20, DARKGRAY);
        DrawText(TextFormat("Alive particles: %d", system.m_countAlive), 10, 50, 20, DARKGRAY);
        DrawText(TextFormat("Total particles: %d", system.size), 10, 70, 20, DARKGRAY);
        DrawText(TextFormat("FPS: %d", GetFPS()), 10, 90, 20, DARKGRAY);
        EndDrawing();
    }


    CloseWindow();
}