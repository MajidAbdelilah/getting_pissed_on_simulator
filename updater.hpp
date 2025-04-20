#pragma once
#include "particle.hpp"
#include <sycl/sycl.hpp>

sycl::vec<float, 4> random_vec(sycl::vec<float, 4> min, sycl::vec<float, 4> max, unsigned int time);

class ParticleUpdater
{
public:
    ParticleUpdater() { }
     ~ParticleUpdater() { }

     void update(double dt, Particle_system &p);
};

class EulerUpdater
{
public:
    sycl::vec<float, 4> m_globalAcceleration;
    float m_floorY{ 1000.0f };
	float m_bounceFactor{ 2.0f };
    size_t countAlive;
    size_t *buf_countAlive;
    // std::vector<sycl::vec<float, 4>> m_attractors; // .w is force
    sycl::queue q;
public:
    // size_t collectionSize() const { return m_attractors.size(); }
	// void add(const sycl::vec<float, 4> &attr) { m_attractors.push_back(attr); }
	// sycl::vec<float, 4> &get(size_t id) { return m_attractors[id]; }
public:
    EulerUpdater(): q(sycl::gpu_selector_v), countAlive(0), buf_countAlive(nullptr){
        buf_countAlive = sycl::malloc_device<size_t>(1, q);
        q.memset(buf_countAlive, 0, sizeof(size_t)).wait(); 
        // m_attractors.push_back({15, 4, -3, 10}); 
        // m_attractors.push_back({-1, 20, 13, 10});
        // m_attractors.push_back({-10, 0, 0, 10});
    }
    ~EulerUpdater() {
        sycl::free(buf_countAlive, q);
    }
     void update(double dt, Particle_system &p) 
    {
        // if(p.m_countAlive == 0) return;
        unsigned int current_time = time(0);
        m_globalAcceleration = random_vec(sycl::vec<float, 4>(-50.0f), sycl::vec<float, 4>(50.0f), current_time);
        const sycl::vec<float, 4> globalA{ dt * m_globalAcceleration.x(), 
                                 dt * m_globalAcceleration.y(), 
                                 dt * m_globalAcceleration.z(), 
                                 0.0 };
        const float localDT = (float)dt;
    
        const unsigned int endId = p.size;
                
        // if(p.m_countAlive == 0) return;
        float m_floorY = this->m_floorY;
        float m_bounceFactor = this->m_bounceFactor;
        q.memset(buf_countAlive, 0, sizeof(size_t)).wait(); 
        q.submit([&](sycl::handler &h){
            auto buf_acc = p.m_particle;
            auto count_reduce = sycl::reduction(buf_countAlive, sycl::plus<>());
            h.parallel_for(sycl::range<1>(p.size), count_reduce, [=](sycl::id<1> idx_d, auto &acc){
                size_t idx = idx_d.get(0);
                if(buf_acc[idx].alive == false)
                {
                    return ;
                }
                acc++;
                if(buf_acc[idx].time.x() < 0.0f && buf_acc[idx].alive == true)
                {
                    buf_acc[idx].alive = false;
                    // max += -1;
                    return ;
                }

                buf_acc[idx].acc += globalA;
    
                buf_acc[idx].vel += localDT * buf_acc[idx].acc;
        
                buf_acc[idx].pos += localDT * buf_acc[idx].vel;
    
                if (buf_acc[idx].pos.y() > m_floorY)
                {
                    sycl::vec<float, 4> force = buf_acc[idx].acc;
                    
                    float normalFactor = sycl::dot(force, sycl::vec<float, 4>(0.0f, 1.0f, 0.0f, 0.0f));
                    if (normalFactor < 0.0f)
                        force -= sycl::vec<float, 4>(0.0f, 1.0f, 0.0f, 0.0f) * normalFactor;
    
                    float velFactor = sycl::dot(buf_acc[idx].vel, sycl::vec<float, 4>(0.0f, 1.0f, 0.0f, 0.0f));
                    //if (velFactor < 0.0)
                    buf_acc[idx].vel -= sycl::vec<float, 4>(0.0f, 1.0f, 0.0f, 0.0f) * (1.0f + m_bounceFactor) * velFactor;
    
                    buf_acc[idx].acc = force;
                }
                // sycl::vec<float, 4> off;
                // float dist;
                // size_t a = 0;    
                // for (a = 0; a < countAttractors; ++a)
                // {
                //     off.x() = m_attractors_acc[a].x() - buf_acc[idx].pos.x();
                //     off.y() = m_attractors_acc[a].y() - buf_acc[idx].pos.y();
                //     off.z() = m_attractors_acc[a].z() - buf_acc[idx].pos.z();
                //     dist = sycl::dot(off, off);
    
                //     //if (fabs(dist) > 0.00001)
                //     dist = m_attractors_acc[a].w() / dist;
    
                //     buf_acc[idx].acc += off * dist;
                // }
    
                buf_acc[idx].time.x() -= localDT;
                // interpolation: from 0 (start of life) till 1 (end of life)
                buf_acc[idx].time.z() = (float)1.0 - (buf_acc[idx].time.x()*buf_acc[idx].time.w()); // .w is 1.0/max life time		
                
                buf_acc[idx].col = sycl::mix(buf_acc[idx].startCol, buf_acc[idx].endCol, sycl::vec<float, 4>(buf_acc[idx].time.z()));
            });
        }).wait();

        // q.wait();
        // p.m_countAlive = maxBuf.get_host_access()[0];
        q.copy<size_t>(buf_countAlive, &p.m_countAlive, 1).wait();
        // q.wait();

    }
};




// void FloorUpdater::update(double dt, ParticleData *p)
// {

// }


// void AttractorUpdater::update(double dt, ParticleData *p)
// {
// const float localDT = (float)dt;


// }

// void BasicColorUpdater::update(double dt, ParticleData *p)
// {
// const size_t endId = p->m_countAlive;
// for (size_t i = 0; i < endId; ++i)
//     p->m_col[i] = glm::mix(p->m_startCol[i], p->m_endCol[i], p->m_time[i].z);
// }

// void PosColorUpdater::update(double dt, ParticleData *p)
// {
// const size_t endId = p->m_countAlive;
// float scaler, scaleg, scaleb;
// float diffr = m_maxPos.x - m_minPos.x;
// float diffg = m_maxPos.y - m_minPos.y;
// float diffb = m_maxPos.z - m_minPos.z;
// for (size_t i = 0; i < endId; ++i)
// {
//     scaler = (p->m_pos[i].x - m_minPos.x) / diffr;
//     scaleg = (p->m_pos[i].y - m_minPos.y) / diffg;
//     scaleb = (p->m_pos[i].z - m_minPos.z) / diffb;
//     p->m_col[i].r = scaler;// glm::mix(p->m_startCol[i].r, p->m_endCol[i].r, scaler);
//     p->m_col[i].g = scaleg;// glm::mix(p->m_startCol[i].g, p->m_endCol[i].g, scaleg);
//     p->m_col[i].b = scaleb;// glm::mix(p->m_startCol[i].b, p->m_endCol[i].b, scaleb);
//     p->m_col[i].a = glm::mix(p->m_startCol[i].a, p->m_endCol[i].a, p->m_time[i].z);
// }
// }

// void VelColorUpdater::update(double dt, ParticleData *p)
// {
// const size_t endId = p->m_countAlive;
// float scaler, scaleg, scaleb;
// float diffr = m_maxVel.x - m_minVel.x;
// float diffg = m_maxVel.y - m_minVel.y;
// float diffb = m_maxVel.z - m_minVel.z;
// for (size_t i = 0; i < endId; ++i)
// {
//     scaler = (p->m_vel[i].x - m_minVel.x) / diffr;
//     scaleg = (p->m_vel[i].y - m_minVel.y) / diffg;
//     scaleb = (p->m_vel[i].z - m_minVel.z) / diffb;
//     p->m_col[i].r = scaler;// glm::mix(p->m_startCol[i].r, p->m_endCol[i].r, scaler);
//     p->m_col[i].g = scaleg;// glm::mix(p->m_startCol[i].g, p->m_endCol[i].g, scaleg);
//     p->m_col[i].b = scaleb;// glm::mix(p->m_startCol[i].b, p->m_endCol[i].b, scaleb);
//     p->m_col[i].a = glm::mix(p->m_startCol[i].a, p->m_endCol[i].a, p->m_time[i].z);
// }
// }

// void BasicTimeUpdater::update(double dt, ParticleData *p)
// {
// unsigned int endId = p->m_countAlive;
// const float localDT = (float)dt;

// if (endId == 0) return;


// }