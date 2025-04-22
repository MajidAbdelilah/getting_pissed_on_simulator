#pragma once
#include <sycl/sycl.hpp>
#define M_PI 3.14159265358979323846
class Mat4x4
{
public:
    // Now m[i] represents row i of the matrix
    sycl::vec<float, 4> m[4];

    Mat4x4()
    {
        setIdentity();
    }
    
    // Constructor now takes rows instead of columns
    Mat4x4(const sycl::vec<float, 4> &row0, const sycl::vec<float, 4> &row1, const sycl::vec<float, 4> &row2, const sycl::vec<float, 4> &row3)
    {
        m[0] = row0;
        m[1] = row1;
        m[2] = row2;
        m[3] = row3;
    }
    
    // Copy constructor remains the same
    Mat4x4(const Mat4x4 &other)
    {
        for (int i = 0; i < 4; ++i)
        {
            m[i] = other.m[i];
        }
    }
    
    // Assignment operator remains the same
    Mat4x4 &operator=(const Mat4x4 &other)
    {
        if (this != &other)
        {
            for (int i = 0; i < 4; ++i)
            {
                m[i] = other.m[i];
            }
        }
        return *this;
    }
    
    Mat4x4 &operator=(Mat4x4 &&other) = default;
    Mat4x4(Mat4x4 &&other) = default;
    ~Mat4x4() = default;
    
    // Matrix multiplication for row-major format
    Mat4x4 operator*(const Mat4x4 &other) const
    {
        Mat4x4 result;
        // Row-major matrix multiplication: C = A * B
        for (int i = 0; i < 4; ++i) { // Row of result/this
            for (int j = 0; j < 4; ++j) { // Column of result/other
                float sum = 0.0f;
                for (int k = 0; k < 4; ++k) { // Column of this / Row of other
                    sum += m[i][k] * other.m[k][j];
                }
                result.m[i][j] = sum;
            }
        }
        return result;
    }
    
    // Matrix-vector multiplication for row-major format
    sycl::vec<float, 4> operator*(const sycl::vec<float, 4> &vec) const
    {
        // Row-major: each row dot product with vector
        float x = sycl::dot(m[0], vec); // Row 0 dot vec
        float y = sycl::dot(m[1], vec); // Row 1 dot vec
        float z = sycl::dot(m[2], vec); // Row 2 dot vec
        float w = sycl::dot(m[3], vec); // Row 3 dot vec
       
        // Perspective divide (check for w=0 before dividing)
        if (w != 0.0f && w != 1.0f) { 
            x /= w; 
            y /= w; 
            z /= w; 
        } 
        return sycl::vec<float, 4>(x, y, z, w);
    }

    // Set projection matrix (row-major format)
    void setProjectionMatrix(float fov, float aspect, float near, float far)
    {
        float f = 1.0f / sycl::tan(fov / 2.0f);
        
        // Set rows for projection matrix
        m[0] = sycl::vec<float, 4>(f / aspect, 0.0f, 0.0f, 0.0f);
        m[1] = sycl::vec<float, 4>(0.0f, f, 0.0f, 0.0f);
        m[2] = sycl::vec<float, 4>(0.0f, 0.0f, (far + near) / (near - far), (2 * far * near) / (near - far));
        m[3] = sycl::vec<float, 4>(0.0f, 0.0f, -1.0f, 0.0f);
    }

    // Set view matrix (row-major format)
    void setViewMatrix(const sycl::vec<float, 3> &eye, const sycl::vec<float, 3> &lookAt, const sycl::vec<float, 3> &up)
    {
        sycl::vec<float, 3> f = sycl::normalize(lookAt - eye);
        sycl::vec<float, 3> s = sycl::normalize(sycl::cross(f, up));
        sycl::vec<float, 3> u = sycl::cross(s, f);

        // Set rows for view matrix
        m[0] = sycl::vec<float, 4>(s.x(), s.y(), s.z(), -sycl::dot(s, eye));
        m[1] = sycl::vec<float, 4>(u.x(), u.y(), u.z(), -sycl::dot(u, eye));
        m[2] = sycl::vec<float, 4>(-f.x(), -f.y(), -f.z(), sycl::dot(f, eye));
        m[3] = sycl::vec<float, 4>(0.0f, 0.0f, 0.0f, 1.0f);
    }
    
    // Set identity matrix (row-major format)
    void setIdentity()
    {
        m[0] = sycl::vec<float, 4>(1.0f, 0.0f, 0.0f, 0.0f);
        m[1] = sycl::vec<float, 4>(0.0f, 1.0f, 0.0f, 0.0f);
        m[2] = sycl::vec<float, 4>(0.0f, 0.0f, 1.0f, 0.0f);
        m[3] = sycl::vec<float, 4>(0.0f, 0.0f, 0.0f, 1.0f);
    }
    
    // Set translation matrix (row-major format)
    void setTranslation(const sycl::vec<float, 3> &translation)
    {
        setIdentity();
        m[0][3] = translation.x();
        m[1][3] = translation.y();
        m[2][3] = translation.z();
    }
    
    // Set rotation matrix (row-major format)
    void setRotation(const sycl::vec<float, 3> &rotation, const float angle)
    {
        setIdentity();
        float rad = angle * (M_PI / 180.0f);
        float cx = sycl::cos(rad);
        float sx = sycl::sin(rad);
        float cy = sycl::cos(rotation.y());
        float sy = sycl::sin(rotation.y());
        float cz = sycl::cos(rotation.z());
        float sz = sycl::sin(rotation.z());
        // Row 0
        m[0][0] = cy * cz;
        m[0][1] = cy * sz;
        m[0][2] = -sy;
        
        // Row 1
        m[1][0] = sx * sy * cz - cx * sz;
        m[1][1] = sx * sy * sz + cx * cz;
        m[1][2] = sx * cy;
        
        // Row 2
        m[2][0] = cx * sy * cz + sx * sz;
        m[2][1] = cx * sy * sz - sx * cz;
        m[2][2] = cx * cy;
    }
    
    // Set scale matrix (row-major format)
    void setScale(const sycl::vec<float, 3> &scale)
    {
        setIdentity();
        m[0][0] = scale.x();
        m[1][1] = scale.y();
        m[2][2] = scale.z();
    }

    // transpose
    Mat4x4 transpose() const
    {
        Mat4x4 result;
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                result.m[i][j] = m[j][i];
            }
        }
        return result;
    }
    
    // Other matrix operations can be added here
};

template<>
struct sycl::is_device_copyable<Mat4x4> : std::true_type {};