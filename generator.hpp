#pragma once
#include <sycl/sycl.hpp>
#include "particle.hpp"
#include "random.hpp"

using Random = effolkronium::random_static;

sycl::vec<float, 4> random_vec(sycl::vec<float, 4> min, sycl::vec<float, 4> max)
{
    return {Random::get(min.x(), max.x()), Random::get(min.y(), max.y()), Random::get(min.z(), max.z()), Random::get(min.w(), max.w())};
}


class BoxPosGen 
{
public:
    sycl::vec<float, 4> m_pos{ 100.0 };
    sycl::vec<float, 4> m_maxStartPosOffset{ 100.0 };
public:
    BoxPosGen(): m_pos(100.0), m_maxStartPosOffset(100.0) { }

    void generate(double dt, Particle_system &p, size_t startId, size_t endId)
    {
        sycl::vec<float, 4> posMin{ m_pos.x() - m_maxStartPosOffset.x(), m_pos.y() - m_maxStartPosOffset.y(), m_pos.z() - m_maxStartPosOffset.z(), 1.0 };
        sycl::vec<float, 4> posMax{ m_pos.x() + m_maxStartPosOffset.x(), m_pos.y() + m_maxStartPosOffset.y(), m_pos.z() + m_maxStartPosOffset.z(), 1.0 };
        
        for (size_t i = startId; i < endId; ++i)
        {
            p.m_particle[i].pos = random_vec(posMin, posMax);
        }
    }
};

class RoundPosGen 
{
public:
    sycl::vec<float, 4> m_center{ 0.0 };
    float m_radX{ 0.0 };
    float m_radY{ 0.0 };
public:
    RoundPosGen() { }
    RoundPosGen(const sycl::vec<float, 4> &center, double radX, double radY)
        : m_center(center)
        , m_radX((float)radX)
        , m_radY((float)radY)
    { }

    void generate(double dt, Particle_system &p, size_t startId, size_t endId)
    {
        for (size_t i = startId; i < endId; ++i)
        {
            double ang = Random::get(0.0, M_PI*2.0);
            p.m_particle[i].pos = m_center + sycl::vec<float, 4>(m_radX*sin(ang), m_radY*cos(ang), 0.0, 1.0);
        }
    }
};

class BasicColorGen 
{
public:
    sycl::vec<float, 4> m_minStartCol{ 0.0 };
    sycl::vec<float, 4> m_maxStartCol{ 0.0 };
    sycl::vec<float, 4> m_minEndCol{ 0.0 };
    sycl::vec<float, 4> m_maxEndCol{ 0.0 };
public:
    BasicColorGen() : m_minStartCol(20.0), m_maxStartCol(100.0), m_minEndCol(100.0), m_maxEndCol(255.0) { }

    void generate(double dt, Particle_system &p, size_t startId, size_t endId)
    {
        for (size_t i = startId; i < endId; ++i)
        {
            p.m_particle[i].startCol = random_vec(m_minStartCol, m_maxStartCol);
            p.m_particle[i].endCol = random_vec(m_minEndCol, m_maxEndCol);
        }
    }};

class BasicVelGen 
{
public:
    sycl::vec<float, 4> m_minStartVel{ 0.0 };
    sycl::vec<float, 4> m_maxStartVel{ 0.0 };
public:
    BasicVelGen() { }

    void generate(double dt, Particle_system &p, size_t startId, size_t endId)
    {
        for (size_t i = startId; i < endId; ++i)
        {
            p.m_particle[i].vel = random_vec(m_minStartVel, m_maxStartVel);
        }
    }
};

class SphereVelGen 
{
public:
    float m_minVel{ 0.0f };
    float m_maxVel{ 0.0f };
public:
    SphereVelGen() { }

    void generate(double dt, Particle_system &p, size_t startId, size_t endId)
    {
        float phi, theta, v, r;
        for (size_t i = startId; i < endId; ++i)
        {
            phi = Random::get(-M_PI, M_PI);
            theta = Random::get(-M_PI, M_PI);
            v = Random::get(m_minVel, m_maxVel);

            r = v*sinf(phi);
            // p.m_particle[i].vel.z() = v*cosf(phi);
            // p.m_particle[i].vel.x() = r*cosf(theta);
            // p.m_particle[i].vel.y() = r*sinf(theta);
            p.m_particle[i].vel = sycl::vec<float, 4>(r*cosf(theta), r*sinf(theta), v*cosf(phi), 1.0);
        }
    }
};

class BasicTimeGen 
{
public:
    float m_minTime{ 0.0 };
    float m_maxTime{ 0.0 };
public:
    BasicTimeGen() { }

    void generate(double dt, Particle_system &p, size_t startId, size_t endId)
    {
        for (size_t i = startId; i < endId; ++i)
        {
            // p.m_particle[i].time.x() = p.m_particle[i].time.y() = Random::get(m_minTime, m_maxTime);
            // p.m_particle[i].time.z() = (float)0.0;
            // p.m_particle[i].time.w() = (float)1.0 / p.m_particle[i].time.x();
            p.m_particle[i].time = sycl::vec<float, 4>(Random::get(m_minTime, m_maxTime), Random::get(m_minTime, m_maxTime), (float)0.0, (float)1.0 / p.m_particle[i].time.x());
            
        }
    }
};