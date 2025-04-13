#include "updater.hpp"
#include "generator.hpp"
#include "particle.hpp"
#include <sycl/sycl.hpp>
#include <iostream>
#include "include/raylib.h"

void emit(double dt, Particle_system &p, size_t m_emitRate)
{
    if (p.m_countAlive >= p.m_particle.size()) return; // No more particles to emit
    if (m_emitRate <= 0) return;              // No emission rate

    const size_t maxNewParticles = static_cast<size_t>(dt*m_emitRate);
    const size_t startId = p.m_countAlive;
    const size_t endId = std::min(startId + maxNewParticles, p.m_particle.size() -1);
    // std::cout << "startId: " << startId << ", endId: " << endId << "\n";
    std::vector<size_t> w;
    w.resize(endId - startId);
    // std::cout << "startId: " << startId << ", endId: " << endId << "\n";
    for (size_t i = startId; i < endId; ++i)  // << wake loop
    {
        w[i - startId] = i;
    }
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

int main()
{

    Particle_system system(400000);
    
    EulerUpdater eu;

    int screenWidth = 1900;
    int screenHeight = 1000;
    InitWindow(screenWidth, screenHeight, "Particle System");
    SetTargetFPS(60);
    Camera2D cam;
    cam.target = (Vector2){ -screenWidth/2.0f, -screenHeight/2.0f};
    cam.offset = (Vector2){ screenWidth/2.0f, screenHeight/2.0f };
    cam.rotation = 0.0f;
    cam.zoom = 1.0f;

    while (!WindowShouldClose())
    {
        double dt = GetFrameTime();

        // Update
        // system.kill({ 0, 1, 2, 3, 4 });
        eu.update(dt, system);
        emit(dt, system, 30000);
        BeginDrawing();
        BeginMode2D(cam);
        ClearBackground(RAYWHITE);
        // system.m_countAlive = system.m_particle.size();
        DrawCircle(200, 200, 100, BLACK);
        if (system.m_countAlive > 0)
        {
            std::vector<size_t> to_kill;
            for (size_t i = 0; i < system.m_countAlive; ++i)
            {
                if(system.m_particle[i].alive)
                {
                    DrawRectangle(system.m_particle[i].pos.x(), system.m_particle[i].pos.y(), 5, 5, Color{ (unsigned char)system.m_particle[i].col.x(), (unsigned char)system.m_particle[i].col.y(), (unsigned char)system.m_particle[i].col.z(), (unsigned char)system.m_particle[i].col.w() });
                }else{
                    to_kill.push_back(i);
                }
                // DrawText(TextFormat("Particle %d", i), system.m_particle[i].pos.x(), system.m_particle[i].pos.y(), 20, Color{ (unsigned char)system.m_particle[i].col.x(), (unsigned char)system.m_particle[i].col.y(), (unsigned char)system.m_particle[i].col.z(), (unsigned char)system.m_particle[i].col.w() });
                // DrawText(TextFormat("Particle Position: (%.2f, %.2f)", system.m_particle[i].pos.x(), system.m_particle[i].pos.y()), system.m_particle[i].pos.x(), system.m_particle[i].pos.y() + 100, 20, DARKGRAY);
                // DrawText(TextFormat("Particle Color: (%.2f, %.2f, %.2f, %.2f)", system.m_particle[i].col.x(), system.m_particle[i].col.y(), system.m_particle[i].col.z(), system.m_particle[i].col.w()), system.m_particle[i].pos.x(), system.m_particle[i].pos.y() + 40, 20, DARKGRAY);
                // std::cout << "particle pos: " << system.m_particle[10].pos.x() << ", " << system.m_particle[10].pos.y() << std::endl;
                // std::cout << "particle color: " << system.m_particle[i].startCol.x() << ", " << system.m_particle[i].startCol.y() << ", " << system.m_particle[i].startCol.z() << ", " << system.m_particle[i].startCol.w() << std::endl;
                // std::cout << 
            }
            system.kill(to_kill);
        }
        EndMode2D();
        DrawText("Particle System", 10, 10, 20, DARKGRAY);
        DrawText("Press ESC to exit", 10, 30, 20, DARKGRAY);
        DrawText(TextFormat("Alive particles: %d", system.m_countAlive), 10, 50, 20, DARKGRAY);
        DrawText(TextFormat("Total particles: %d", system.m_particle.size()), 10, 70, 20, DARKGRAY);
        DrawText(TextFormat("FPS: %d", GetFPS()), 10, 90, 20, DARKGRAY);
        EndDrawing();
    }


    CloseWindow();
}