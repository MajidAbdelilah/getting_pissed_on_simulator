#pragma once
#include <sycl/sycl.hpp>
#include "vector_gpu.hpp"
#include "random.hpp"

using Random = effolkronium::random_static;
sycl::vec<float, 4> random_vec(sycl::vec<float, 4> min, sycl::vec<float, 4> max);
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

template<typename T>
void swap(T &a, T &b)
{
    T temp = a;
    a = b;
    b = temp;
}

class Particle_system
{
public:
    Particle_system(size_t p_count):  q(sycl::gpu_selector_v), m_countAlive(0){
        m_particle = sycl::malloc_device<Particle>(p_count, q);
        q.memset(m_particle, 0, sizeof(Particle) * p_count);
        size = p_count;
    }

    ~Particle_system() {
        sycl::free(m_particle, q);
    }
    
    void kill()
    {
        // if(m_countAlive == 0) return;
        // // if(k == nullptr || kill_count == 0) return;
        // // if(kill_count > m_countAlive) throw std::runtime_error("k size is greater than m_countAlive " + std::to_string(kill_count) + ", " + std::to_string(m_countAlive));
        // // Create SYCL buffer from the vector data
        
        // // size_t k_size = k.size();
        // size_t count_alive = m_countAlive;
        // // size_t thread_count = q.get_device().get_info<sycl::info::device::max_compute_units>() * 128;
        // size_t *kill_count = sycl::malloc_device<size_t>(1, q);
        // size_t fill = 0;
        // q.copy<size_t>(&fill, kill_count, 1);
        // q.wait();
        // q.submit([&](sycl::handler &h){
        //     auto kill_count_reduce = sycl::reduction(kill_count, sycl::plus<>());
        //     auto buf_acc = m_particle;
        //     auto size = this->size;
        //     h.parallel_for(sycl::range<1>(size), kill_count_reduce, [=](sycl::id<1> idx_d, auto &kc_reduce){
        //         auto idx = idx_d.get(0);
        //         if (buf_acc[idx].time.x() < 0.0f && buf_acc[idx].alive == true) {
        //             buf_acc[idx].alive = false;
        //             kc_reduce++;
        //         }
        //     });
        // }).wait();
        // // std::cout << "her 1\n";
        // q.wait();
        // // std::cout << "her 2\n";
        // q.copy<size_t>(kill_count, &fill, 1);
        // q.wait();

        // std::cout << "kill count: " << fill << "\n";
        // this->m_countAlive -= fill;
        
        // sycl::free(kill_count, q);
    }

    void wake(size_t  rev_size)
    {
        // if(w == nullptr) return;
        // if(size == 0) return;
        // if(size > (this->size - m_countAlive)) throw std::runtime_error("w size is greater than m_countAlive " + std::to_string(size) + ", " + std::to_string(m_countAlive));

        // if((m_countAlive + w.size()) > size) return;
        // sycl::buffer<Particle, 1> buf(m_particle.data(), sycl::range<1>(m_particle.size()));
        // sycl::buffer<size_t, 1> w_buf(w);
        // sycl::buffer<Particle, 1> buf(m_particle);
        // size_t w_size = w.size();
        
        // size_t erased = 0;
        // for(size_t idx = 0; idx < w_size; ++idx)
        // {
        //     // std::cout << w[idx] << "\n";
        //     if(w[idx] >= m_countAlive && w[idx] < (m_countAlive + w_size) && m_countAlive > 0)
        //     {
        //         // Swap the elements
        //         m_particle[w[idx]].alive = true;
        //         swap(m_particle[m_countAlive], m_particle[w[idx]]);
        //         m_countAlive++;
        //         // Particle tmp = m_particle[m_countAlive];
        //         // m_particle[m_countAlive] = m_particle[w[idx]];
        //         // m_particle[w[idx]] = tmp;
        //         // m_particle[m_countAlive].alive = true;
        //         // m_particle[w[idx]].alive = false;
                
        //         // size_t size_tmp = w[(w.size() - erased)- 1]; // swap
        //         // w[(w.size() - erased)- 1] = w[idx];
        //         // w[idx] = size_tmp;
        //         // erased++;
        //         // idx--;
        //         w[idx] = SIZE_MAX; // mark as erased
        //         erased++;
        //     }
        // }
        // std::sort(w.begin(), w.end());
        

        // w.resize(w.size() - erased);
        // if(w.size() == 0) return;

        // w_size = w.size();
        // size_t count_alive = m_countAlive;
        // size_t *rev_count = sycl::malloc_device<size_t>(1, q);
        // size_t fill = 0;
        // q.copy<size_t>(&fill, rev_count, 1);
        // q.wait();
        // q.submit([&](sycl::handler &h){
        //     auto buf_acc = m_particle;
        //     // auto w_acc = w; // Updated to use the correct access
        //     // std::cout << "loop_size: " << size - erased << ", " << size << ", " << erased << "\n"; // Updated to use size parameter
        //     h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx_d){
        //         size_t idx = idx_d.get(0);
                
        //         if (buf_acc[idx].alive == false && *rev_count < rev_size) {
                    
        //             buf_acc[idx].alive = true;
                    
                    
        //             {
        //                 sycl::atomic_ref<size_t,
        //                 sycl::memory_order::relaxed, // Relaxed is often sufficient for counters
        //                 sycl::memory_scope::device> // Scope depends on where particles are checked
        //                 atomic_counter(*rev_count);
        //                 atomic_counter.fetch_add(1);
        //             }

        //         }
        //     });
        // }).wait();

        // q.wait();
        // // Update the count of alive particles
        // m_countAlive += rev_size;

    }

    Particle *m_particle;
    size_t size;
    sycl::queue q;
    // sycl::buffer<Particle, 1> buf;
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

