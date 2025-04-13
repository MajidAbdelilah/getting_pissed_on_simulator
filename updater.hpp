#pragma once
#include "particle.hpp"
#include <sycl/sycl.hpp>

#include "random.hpp"

using Random = effolkronium::random_static;

sycl::vec<float, 4> random_vec(sycl::vec<float, 4> min, sycl::vec<float, 4> max);

class ParticleUpdater
{
public:
    ParticleUpdater() { }
    virtual ~ParticleUpdater() { }

    virtual void update(double dt, Particle_system &p) = 0;
};

class EulerUpdater : public ParticleUpdater
{
public:
    sycl::vec<float, 4> m_globalAcceleration;
    float m_floorY{ -1.0f };
	float m_bounceFactor{ 2.5f };
    std::vector<sycl::vec<float, 4>> m_attractors; // .w is force
    sycl::queue q;
public:
    size_t collectionSize() const { return m_attractors.size(); }
	void add(const sycl::vec<float, 4> &attr) { m_attractors.push_back(attr); }
	sycl::vec<float, 4> &get(size_t id) { return m_attractors[id]; }
public:
    EulerUpdater(): q(sycl::cpu_selector_v) { 
        m_attractors.push_back({15, 4, -3, 10}); 
        m_attractors.push_back({-1, 20, 13, 10});
        m_attractors.push_back({-10, 0, 0, 10});
    }
    virtual void update(double dt, Particle_system &p) override
    {
        m_globalAcceleration = random_vec(sycl::vec<float, 4>(-50.0f), sycl::vec<float, 4>(50.0f));
        const sycl::vec<float, 4> globalA{ dt * m_globalAcceleration.x(), 
                                 dt * m_globalAcceleration.y(), 
                                 dt * m_globalAcceleration.z(), 
                                 0.0 };
        const float localDT = (float)dt;
    
        const unsigned int endId = p.m_countAlive;
        
        // const size_t countAttractors = m_attractors.size();
        
        std::vector<size_t> to_kill;
        to_kill.reserve(endId); // Reserve space for to_kill
        sycl::buffer<Particle, 1> buf(p.m_particle);
        sycl::buffer<sycl::vec<float, 4>> m_attractors_buf(m_attractors);
        // size_t buf_size = p.m_particle.size();
        float m_floorY = this->m_floorY;
        float m_bounceFactor = this->m_bounceFactor;
        q.submit([&](sycl::handler &h){
            sycl::accessor buf_acc = buf.template get_access<sycl::access_mode::read_write>(h);
            // sycl::accessor m_attractors_acc = m_attractors_buf.template get_access<sycl::access_mode::read>(h);
            h.parallel_for(sycl::range<1>(endId), [=](sycl::id<1> idx_d){
                size_t idx = idx_d.get(0);
                // if(buf_acc[idx].alive == false)
                // {
                //     return ;
                // }
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

        q.wait();
        
        size_t kill_count = 0;

        for (size_t i = 0; i < endId; ++i)
        {
            // p.m_particle[i].acc += globalA;
    
            // p.m_particle[i].vel += localDT * p.m_particle[i].acc;
    
            // p.m_particle[i].pos += localDT * p.m_particle[i].vel;

            // if (p.m_particle[i].pos.y() > m_floorY)
            // {
            //     sycl::vec<float, 4> force = p.m_particle[i].acc;
                
            //     float normalFactor = sycl::dot(force, sycl::vec<float, 4>(0.0f, 1.0f, 0.0f, 0.0f));
            //     if (normalFactor < 0.0f)
            //         force -= sycl::vec<float, 4>(0.0f, 1.0f, 0.0f, 0.0f) * normalFactor;

            //     float velFactor = sycl::dot(p.m_particle[i].vel, sycl::vec<float, 4>(0.0f, 1.0f, 0.0f, 0.0f));
            //     //if (velFactor < 0.0)
            //     p.m_particle[i].vel -= sycl::vec<float, 4>(0.0f, 1.0f, 0.0f, 0.0f) * (1.0f + m_bounceFactor) * velFactor;

            //     p.m_particle[i].acc = force;
            // }
        
            // for (a = 0; a < countAttractors; ++a)
            // {
            //     off.x() = m_attractors[a].x() - p.m_particle[i].pos.x();
            //     off.y() = m_attractors[a].y() - p.m_particle[i].pos.y();
            //     off.z() = m_attractors[a].z() - p.m_particle[i].pos.z();
            //     dist = sycl::dot(off, off);

            //     //if (fabs(dist) > 0.00001)
            //     dist = m_attractors[a].w() / dist;

            //     p.m_particle[i].acc += off * dist;
            // }

            // p.m_particle[i].col = sycl::mix(p.m_particle[i].startCol, p.m_particle[i].endCol, sycl::vec<float, 4>(p.m_particle[i].time.z()));
        
            // p.m_particle[i].time.x() -= localDT;
            // // interpolation: from 0 (start of life) till 1 (end of life)
            // p.m_particle[i].time.z() = (float)1.0 - (p.m_particle[i].time.x()*p.m_particle[i].time.w()); // .w is 1.0/max life time		
            
           
            // std::cout << "time.x(): "<< p.m_particle[i].time.x()  << ", localDt: " << localDT << "\n";
            if (p.m_particle[i].time.x() < 0.0f)
            {
            // std::cout << "kill, time.x(): "<< p.m_particle[i].time.x()  << ", localDt: " << localDT << "\n";

                kill_count++;        
                to_kill.push_back(i);
				// endId = p.m_countAlive < p.m_particle.size() ? p.m_countAlive : p.m_particle.size();
                // p.m_particle[i].time = sycl::vec<float, 4>(20, 20, (float)0.0, (float)1.0 / p.m_particle[i].time.x());

            }
        }
        // std::cout << "to_kill.size(): " << to_kill.size() << "\n";
        to_kill.resize(kill_count);
        p.kill(to_kill);

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