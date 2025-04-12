#pragma once
#include <sycl/sycl.hpp>
#include "vector_gpu.hpp"

class Particle
{
public:
    Particle() = default;
    Particle(sycl::vec<float, 4> position, sycl::vec<float, 4> color,
             sycl::vec<float, 4> startColor, sycl::vec<float, 4> endColor,
             sycl::vec<float, 4> velocity, sycl::vec<float, 4> acceleration,
             sycl::vec<float, 4> timeToLive)
        : pos(position), col(color), startCol(startColor), endCol(endColor),
          vel(velocity), acc(acceleration), time(timeToLive), alive(true) {}
    Particle(Particle &&other) = default;
    Particle(const Particle &other)
    {
        pos = other.pos;
        col = other.col;
        startCol = other.startCol;
        endCol = other.endCol;
        vel = other.vel;
        acc = other.acc;
        time = other.time;
        alive = other.alive;
    }
    Particle &operator=(Particle &&other) = default;
    Particle &operator=(const Particle &other)
    {
        pos = other.pos;
        col = other.col;
        startCol = other.startCol;
        endCol = other.endCol;
        vel = other.vel;
        acc = other.acc;
        time = other.time;
        alive = other.alive;
        return *this;
    }
    ~Particle() = default;

    sycl::vec<float, 4> pos;
    sycl::vec<float, 4> col;
    sycl::vec<float, 4> startCol;
    sycl::vec<float, 4> endCol;
    sycl::vec<float, 4> vel;
    sycl::vec<float, 4> acc;
    sycl::vec<float, 4> time;
    bool  alive;
};


template<>
struct sycl::is_device_copyable<Particle> : std::true_type {};


class Particle_system
{
public:
    Particle_system(size_t p_count): m_countAlive(0), q(sycl::gpu_selector_v), buf(m_particle.data(), sycl::range<1>(p_count)){
        m_particle.resize(p_count);
        // buf = sycl::buffer<Particle, 1>(m_particle.data(), sycl::range<1>(p_count));
    }
    
    void kill(std::vector<size_t> k)
    {
        if(m_countAlive == 0) return;
        if(k.size() == 0) return;
        if(k.size() > m_countAlive) throw std::runtime_error("k size is greater than m_countAlive");
        // Create SYCL buffer from the vector data
        // sycl::buffer<Particle, 1> buf(m_particle.data(), sycl::range<1>(m_particle.size()));
        sycl::buffer<size_t, 1> k_buf(k.data(), sycl::range<1>(k.size()));
        size_t k_size = k.size();
        size_t count_alive = m_countAlive;
        
        q.submit([&](sycl::handler &h){
            auto buf_acc = buf.template get_access<sycl::access::mode::read_write>(h);
            auto k_acc = k_buf.template get_access<sycl::access::mode::read>(h);
            h.parallel_for(sycl::range<1>(k.size()), [=](size_t idx){
                size_t index = k_acc[idx];
                buf_acc[index].alive = false;
                Particle tmp = buf_acc[(count_alive - ((k_size - idx) - 1)) - 1];
                buf_acc[(count_alive - ((k_size - idx) - 1)) - 1] = buf_acc[index];
                buf_acc[index] = tmp;
            });
        });
        
        q.wait();

        m_countAlive -= k.size();
        
        k.clear();
    }

    void wake(std::vector<size_t> w)
    {
        if((m_countAlive + w.size()) > m_particle.size()) return;
        // sycl::buffer<Particle, 1> buf(m_particle.data(), sycl::range<1>(m_particle.size()));
        sycl::buffer<size_t, 1> w_buf(w.data(), sycl::range<1>(w.size()));
        size_t w_size = w.size();
        
        size_t erased = 0;
        for(size_t idx = 0; idx < w_size; ++idx)
        {
            if(w[idx] >= m_countAlive && w[idx] < (m_countAlive + w_size))
            {
                // Swap the elements
                Particle tmp = m_particle[m_countAlive];
                m_particle[m_countAlive] = m_particle[w[idx]];
                m_particle[w[idx]] = tmp;
                m_particle[m_countAlive].alive = true;
                m_countAlive++;
                size_t size_tmp = w[(w.size() - erased)- 1]; // swap
                w[(w.size() - erased)- 1] = w[idx];
                w[idx] = size_tmp;
                erased++;
                idx--;
            }
        }

        w.resize(w.size() - erased);
        if(w.size() == 0) return;

        w_size = w.size();
        size_t count_alive = m_countAlive;
        q.submit([&](sycl::handler &h){
            auto buf_acc = buf.template get_access<sycl::access::mode::read_write>(h);
            auto w_acc = w_buf.template get_access<sycl::access::mode::read>(h);
            h.parallel_for(sycl::range<1>(w.size()), [=](size_t idx){
                size_t index = w_acc[idx];
                buf_acc[index].alive = true;
                Particle tmp = buf_acc[count_alive + idx];
                buf_acc[count_alive + idx] = buf_acc[index];
                buf_acc[index] = tmp;
            });
        });

        q.wait();
        // Update the count of alive particles
        m_countAlive += w.size();

        w.clear();
    }

    Lp_parallel_vector_GPU<Particle> m_particle;
    sycl::queue q;
    sycl::buffer<Particle, 1> buf;
    size_t m_countAlive{ 0 };
};


// class generate_func
// {
// public:
//     Particle operator()(Particle &p, size_t idx)
//     {

//     }
// };

// class kill_func
// {
// public:
//     kill_func(size_t m_countAlive) : m_countAlive(m_countAlive) {}
//     Particle operator()(Particle &p, size_t idx)
//     {
        
//     }
//     size_t m_countAlive;
// };

// class wake_func
// {
// public:
//     Particle operator()(Particle &p, size_t idx)
//     {

//     }   
// };

// class swapData_func
// {
// public:
//     Particle operator()(Particle &p, size_t idx)
//     {

//     }   
// };

