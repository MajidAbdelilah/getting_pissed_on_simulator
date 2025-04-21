#pragma once
#include <sycl/sycl.hpp>
#include "particle.hpp"
// #include "random.hpp"
#include "my_random.hpp"

// using Random = effolkronium::random_static;




sycl::vec<float, 4> random_vec(sycl::vec<float, 4> min, sycl::vec<float, 4> max, unsigned int time)
{
    return random_rangef(min, max, time);
}


class Gen 
{
public:
    sycl::vec<float, 4> m_pos;
    sycl::vec<float, 4> m_maxStartPosOffset;
    sycl::vec<float, 4> m_minStartCol{ 0.0 };
    sycl::vec<float, 4> m_maxStartCol{ 0.0 };
    sycl::vec<float, 4> m_minEndCol{ 0.0 };
    sycl::vec<float, 4> m_maxEndCol{ 0.0 };
    sycl::vec<float, 4> m_minStartVel{ 0.0 };
    sycl::vec<float, 4> m_maxStartVel{ 0.0 };
    float m_minTime;
    float m_maxTime;
    sycl::queue q;
    size_t *rev_count_tmp;
public:
    Gen(): m_pos(0.0f), m_maxStartPosOffset(100.0), m_minStartCol(255.0f, 0, 0, 255.0f), m_maxStartCol(255, 150, 150, 255), m_minEndCol(0, 255.0f, 255.0f, 255.0f), m_maxEndCol(0, 255.0f, 255.0f, 255.0f), m_minStartVel(-50), m_maxStartVel(50), m_minTime(10.0f), m_maxTime(60.0f), q(sycl::gpu_selector_v) 
    { 
        rev_count_tmp = sycl::malloc_device<size_t>(1, q);
        q.memset(rev_count_tmp, 0, sizeof(size_t)).wait();
    }

    void generate(Particle_system &p, size_t rev_size)
    {
        sycl::vec<float, 4> posMin{ m_pos.x() - m_maxStartPosOffset.x(), m_pos.y() - m_maxStartPosOffset.y(), m_pos.z() - m_maxStartPosOffset.z(), 1.0 };
        sycl::vec<float, 4> posMax{ m_pos.x() + m_maxStartPosOffset.x(), m_pos.y() + m_maxStartPosOffset.y(), m_pos.z() + m_maxStartPosOffset.z(), 1.0 };
        
        // sycl::buffer<Particle, 1> buf(p.m_particle);
        
        // Determine number of threads to use
        // const unsigned int numThreads = std::thread::hardware_concurrency();
        
        // std::vector<std::thread> threads;
        // threads.reserve(numThreads);
        unsigned int current_time = time(0);

        q.memset(rev_count_tmp, 0, sizeof(size_t)).wait();

        // Lambda function for thread work with strided access pattern
        q.submit([&](sycl::handler &h){
            auto buf_acc = p.m_particle;
            auto m_minStartCol = this->m_minStartCol;
            auto m_maxStartCol = this->m_maxStartCol;
            auto m_minEndCol = this->m_minEndCol;
            auto m_maxEndCol = this->m_maxEndCol;
            auto m_minStartVel = this->m_minStartVel;
            auto m_maxStartVel = this->m_maxStartVel;
            auto m_minTime = this->m_minTime;
            auto m_maxTime = this->m_maxTime;
            size_t size_all = p.size;
            size_t *rev_count_tmp = this->rev_count_tmp;
            // sycl::buffer<size_t> maxBuf { &p.m_countAlive, 1 };
            // auto maxReduction = reduction(maxBuf, h, sycl::plus<>());
            // std::cout << "rev_size = " << rev_size << "\n";
            h.parallel_for(sycl::range<1>(size_all), [=](sycl::id<1> idx_d){
                size_t idx = idx_d.get(0);
                if (buf_acc[idx].alive == false && *rev_count_tmp < rev_size)
                {

                    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device> rev_ref(*rev_count_tmp);
                    size_t prev_val =  rev_ref.fetch_add(1);
                    if( prev_val < rev_size)
                    {
                        buf_acc[idx].alive = true;
                        buf_acc[idx].pos = random_vec(posMin, posMax, current_time + idx * 1000);
                        buf_acc[idx].startCol = random_vec(m_minStartCol, m_maxStartCol, current_time + idx * 1000);
                        buf_acc[idx].endCol = random_vec(m_minEndCol, m_maxEndCol, current_time + idx * 1000);
                        buf_acc[idx].vel = random_vec(m_minStartVel, m_maxStartVel, current_time + idx * 1000);
                        buf_acc[idx].time.x() = buf_acc[idx].time.y() = random_rangef(m_minTime, m_maxTime, current_time + idx * 1000);
                        buf_acc[idx].time.z() = (float)0.0;
                        buf_acc[idx].time.w() = (float)1.0 / buf_acc[idx].time.x();
                        
                        // rev_ref.fetch_sub(1);
                        // max += 1;
                        // return ;
                    }
                    // size_t i = idx;
                }
                
            });
        }).wait();

        // p.m_countAlive += rev_size;
        // auto threadWork = [&](size_t threadId) {
        //     // Start at threadId and increment by numThreads
        //     // std::cout << "thread id = " << threadId << "\n";
        //     for (size_t i = startId + threadId; i < endId; i += numThreads) {
            
        //     p.m_particle[i].pos = random_vec(posMin, posMax, current_time + i * 1000);
        //     p.m_particle[i].startCol = random_vec(m_minStartCol, m_maxStartCol, current_time + i * 1000);
        //     p.m_particle[i].endCol = random_vec(m_minEndCol, m_maxEndCol, current_time + i * 1000);
        //     p.m_particle[i].vel = random_vec(m_minStartVel, m_maxStartVel, current_time + i * 1000);
        //     p.m_particle[i].time.x() = p.m_particle[i].time.y() = random_rangef(m_minTime, m_maxTime, current_time + i * 1000);
        //     p.m_particle[i].time.z() = (float)0.0;
        //     p.m_particle[i].time.w() = (float)1.0 / p.m_particle[i].time.x();
        //     }
        // };

        
        

        // for (size_t i = startId; i < endId; ++i)
        // {
        //     p.m_particle[i].pos = random_vec(posMin, posMax);
        //     p.m_particle[i].startCol = random_vec(m_minStartCol, m_maxStartCol);
        //     p.m_particle[i].endCol = random_vec(m_minEndCol, m_maxEndCol);
        //     p.m_particle[i].vel = random_vec(m_minStartVel, m_maxStartVel);
        //     p.m_particle[i].time.x() = p.m_particle[i].time.y() = my_randf(m_minTime, m_maxTime);
        //     p.m_particle[i].time.z() = (float)0.0;
        //     p.m_particle[i].time.w() = (float)1.0 / p.m_particle[i].time.x();
        // }
    }
};

// class RoundPosGen 
// {
// public:
//     sycl::vec<float, 4> m_center{ 0.0 };
//     float m_radX{ 0.0 };
//     float m_radY{ 0.0 };
// public:
//     RoundPosGen() { }
//     RoundPosGen(const sycl::vec<float, 4> &center, double radX, double radY)
//         : m_center(center)
//         , m_radX((float)radX)
//         , m_radY((float)radY)
//     { }

//     void generate(Particle_system &p, size_t startId, size_t endId)
//     {
//         for (size_t i = startId; i < endId; ++i)
//         {
//             double ang = my_randf(0.0, M_PI*2.0);
//             p.m_particle[i].pos = m_center + sycl::vec<float, 4>(m_radX*sin(ang), m_radY*cos(ang), 0.0, 1.0);
//         }
//     }
// };

// class BasicColorGen 
// {
// public:
// public:
//     BasicColorGen() { }

//     void generate(Particle_system &p, size_t startId, size_t endId)
//     {
//         for (size_t i = startId; i < endId; ++i)
//         {
//         }
//     }};

// class BasicVelGen 
// {
// public:
// public:
//     BasicVelGen() { }

//     void generate(Particle_system &p, size_t startId, size_t endId)
//     {
//         for (size_t i = startId; i < endId; ++i)
//         {
//         }
//     }
// };

// class SphereVelGen 
// {
// public:
//     float m_minVel{ 0.0f };
//     float m_maxVel{ 0.0f };
// public:
//     SphereVelGen() { }

//     void generate(Particle_system &p, size_t startId, size_t endId)
//     {
//         float phi, theta, v, r;
//         for (size_t i = startId; i < endId; ++i)
//         {
//             phi = my_randf(-M_PI, M_PI);
//             theta = my_randf(-M_PI, M_PI);
//             v = my_randf(m_minVel, m_maxVel);

//             r = v*sinf(phi);
//             // p.m_particle[i].vel.z() = v*cosf(phi);
//             // p.m_particle[i].vel.x() = r*cosf(theta);
//             // p.m_particle[i].vel.y() = r*sinf(theta);
//             p.m_particle[i].vel = sycl::vec<float, 4>(r*cosf(theta), r*sinf(theta), v*cosf(phi), 1.0);
//         }
//     }
// };

// class BasicTimeGen 
// {
// public:
// public:
//     BasicTimeGen() { }

//     void generate(Particle_system &p, size_t startId, size_t endId)
//     {
//         for (size_t i = startId; i < endId; ++i)
//         {
           
//             // p.m_particle[i].time = sycl::vec<float, 4>(my_randf(m_minTime, m_maxTime), my_randf(m_minTime, m_maxTime), 0.0f, 1.0f / p.m_particle[i].time.x());
            
//         }
//     }
// };