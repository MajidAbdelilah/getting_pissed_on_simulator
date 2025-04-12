#pragma once
#include <cstddef>
#include <thread>
#include <vector>
#include <functional>
#include <mutex>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <iostream>

#include <sycl/sycl.hpp>

template<typename T>
class Lp_parallel_vector_GPU: public std::vector<T>
{
public:
    Lp_parallel_vector_GPU(): std::vector<T>() {
        try {
            // Try to select a GPU device
            q = sycl::queue(sycl::gpu_selector_v);
            is_gpu = true;
        } catch (const sycl::exception& e) {
            std::cerr << "GPU not available, falling back to CPU: " << e.what() << std::endl;
            q = sycl::queue(sycl::cpu_selector_v);
            is_gpu = false;
        }
    };
    
    ~Lp_parallel_vector_GPU() {
        // SYCL queue manages resource cleanup automatically
    };

    Lp_parallel_vector_GPU(size_t num_elements) : std::vector<T>(num_elements) {
        try {
            q = sycl::queue(sycl::gpu_selector_v);
            is_gpu = true;
        } catch (const sycl::exception& e) {
            std::cerr << "GPU not available, falling back to CPU: " << e.what() << std::endl;
            q = sycl::queue(sycl::cpu_selector_v);
            is_gpu = false;
        }
    };
    
    Lp_parallel_vector_GPU(const Lp_parallel_vector_GPU& other) : std::vector<T>(other) {
        q = other.q;
    };
    
    Lp_parallel_vector_GPU& operator=(const Lp_parallel_vector_GPU& other) {
        if(this != &other) {
            std::vector<T>::operator=(other);
            q = other.q;
            is_gpu = other.is_gpu;
        }
        return *this;
    }
    
    Lp_parallel_vector_GPU(const std::vector<T>& other) : std::vector<T>(other) {
        try {
            q = sycl::queue(sycl::gpu_selector_v);
            is_gpu = true;
        } catch (const sycl::exception& e) {
            std::cerr << "GPU not available, falling back to CPU: " << e.what() << std::endl;
            q = sycl::queue(sycl::cpu_selector_v);
            is_gpu = false;
        }
    };
    
    Lp_parallel_vector_GPU& operator=(const std::vector<T>& other) {
        std::vector<T>::operator=(other);
        return *this;
    }
    
    Lp_parallel_vector_GPU(const std::initializer_list<T>& init) : std::vector<T>(init) {
        try {
            q = sycl::queue(sycl::gpu_selector_v);
            is_gpu = true;
        } catch (const sycl::exception& e) {
            std::cerr << "GPU not available, falling back to CPU: " << e.what() << std::endl;
            q = sycl::queue(sycl::cpu_selector_v);
            is_gpu = false;
        }
    };
    
    Lp_parallel_vector_GPU& operator=(const std::initializer_list<T>& init) {
        std::vector<T>::operator=(init);
        return *this;
    }

    void fill(const T value) {
        if (this->empty()) return;
        
        // Create SYCL buffer from the vector data
        sycl::buffer<T, 1> buf(this->data(), sycl::range<1>(this->size()));
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffer
            auto acc = buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                acc[idx] = value;
            });
        });
        
        // Wait for operations to complete
        q.wait();
    }

    void fill(T value, size_t size) {
        this->resize(size);
        fill(value);
    }

    template<class fun>
    void fill(const fun& func) {
        if (this->empty()) return;
        
        // Create SYCL buffer from the vector data
        sycl::buffer<T, 1> buf(this->data(), sycl::range<1>(this->size()));
        size_t size = this->size();
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffer
            auto acc = buf.template get_access<sycl::access::mode::read_write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
                acc[idx] = func(acc[idx], idx);
            });
        });
        
        // Wait for operations to complete
        q.wait();
    }

    template<class fun>
    void fill(const fun& func, size_t size) {
        this->resize(size);
        fill(func);
    }

    Lp_parallel_vector_GPU<T> operator+(const Lp_parallel_vector_GPU<T>& other) const {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] + b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator-(const Lp_parallel_vector_GPU<T>& other) const {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] - b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator*(const Lp_parallel_vector_GPU<T>& other) const {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] * b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator/(const Lp_parallel_vector_GPU<T>& other) const {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                // Add a check to avoid division by zero
                if (b_acc[idx] != 0) {
                    res_acc[idx] = a_acc[idx] / b_acc[idx];
                } else {
                    res_acc[idx] = 0; // or some other appropriate value
                }
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator%(const Lp_parallel_vector_GPU<T>& other) const {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                // Check for division by zero
                if (b_acc[idx] != 0) {
                    res_acc[idx] = a_acc[idx] % b_acc[idx];
                } else {
                    res_acc[idx] = 0; // Handle division by zero
                }
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    // Scalar binary arithmetic operators
    Lp_parallel_vector_GPU<T> operator+(const T& scalar) const {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] + scalar;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator-(const T& scalar) const {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] - scalar;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator*(const T& scalar) const {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] * scalar;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator/(const T& scalar) const {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                // Add a check to avoid division by zero
                if (scalar != 0) {
                    res_acc[idx] = a_acc[idx] / scalar;
                } else {
                    res_acc[idx] = 0; // or some other appropriate value
                }
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator%(const T& scalar) const {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                // Add a check to avoid modulo by zero
                if (scalar != 0) {
                    res_acc[idx] = a_acc[idx] % scalar;
                } else {
                    res_acc[idx] = 0; // or some other appropriate value
                }
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    // Scalar bitwise operators
    Lp_parallel_vector_GPU<T> operator&(const T& scalar) const {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] & scalar;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator|(const T& scalar) const {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] | scalar;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator^(const T& scalar) const {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] ^ scalar;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator<<(const T& scalar) const {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] << scalar;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator>>(const T& scalar) const {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] >> scalar;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    // Scalar logical operators
    Lp_parallel_vector_GPU<T> operator&&(const T& scalar) const {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] && scalar;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator||(const T& scalar) const {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] || scalar;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    // Scalar comparison operators
    Lp_parallel_vector_GPU<T> operator==(const T& scalar) const {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] == scalar;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator!=(const T& scalar) const {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] != scalar;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator<(const T& scalar) const {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] < scalar;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator<=(const T& scalar) const {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] <= scalar;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator>(const T& scalar) const {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] > scalar;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator>=(const T& scalar) const {
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        
        if (this->empty()) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] >= scalar;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T>& operator+=(const Lp_parallel_vector_GPU<T>& other) {
        if (this->empty()) return *this;
        if (this->size() != other.size()) throw std::runtime_error("Size mismatch");
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers - using read_write for a_acc
            auto a_acc = a_buf.template get_access<sycl::access::mode::read_write>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                a_acc[idx] += b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return *this;
    }

    Lp_parallel_vector_GPU<T>& operator-=(const Lp_parallel_vector_GPU<T>& other) {
        if (this->empty()) return *this;
        if (this->size() != other.size()) throw std::runtime_error("Size mismatch");
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers - using read_write for a_acc
            auto a_acc = a_buf.template get_access<sycl::access::mode::read_write>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                a_acc[idx] -= b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return *this;
    }

    Lp_parallel_vector_GPU<T>& operator*=(const Lp_parallel_vector_GPU<T>& other) {
        if (this->empty()) return *this;
        if (this->size() != other.size()) throw std::runtime_error("Size mismatch");
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers - using read_write for a_acc
            auto a_acc = a_buf.template get_access<sycl::access::mode::read_write>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                a_acc[idx] *= b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return *this;
    }

    Lp_parallel_vector_GPU<T>& operator/=(const Lp_parallel_vector_GPU<T>& other) {
        if (this->empty()) return *this;
        if (this->size() != other.size()) throw std::runtime_error("Size mismatch");
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers - using read_write for a_acc
            auto a_acc = a_buf.template get_access<sycl::access::mode::read_write>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                // Add a check to avoid division by zero
                if (b_acc[idx] != 0) {
                    a_acc[idx] /= b_acc[idx];
                } else {
                    a_acc[idx] = 0; // or some other appropriate value
                }
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return *this;
    }

    Lp_parallel_vector_GPU<T>& operator%=(const Lp_parallel_vector_GPU<T>& other) {
        if (this->empty()) return *this;
        if (this->size() != other.size()) throw std::runtime_error("Size mismatch");
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers - using read_write for a_acc
            auto a_acc = a_buf.template get_access<sycl::access::mode::read_write>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                // Check to avoid modulo by zero
                if (b_acc[idx] != 0) {
                    a_acc[idx] %= b_acc[idx];
                } else {
                    a_acc[idx] = 0; // or some other appropriate value
                }
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return *this;
    }

    Lp_parallel_vector_GPU<T>& operator&=(const Lp_parallel_vector_GPU<T>& other) {
        if (this->empty()) return *this;
        if (this->size() != other.size()) throw std::runtime_error("Size mismatch");
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers - using read_write for a_acc
            auto a_acc = a_buf.template get_access<sycl::access::mode::read_write>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                a_acc[idx] &= b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return *this;
    }

    Lp_parallel_vector_GPU<T>& operator|=(const Lp_parallel_vector_GPU<T>& other) {
        if (this->empty()) return *this;
        if (this->size() != other.size()) throw std::runtime_error("Size mismatch");
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers - using read_write for a_acc
            auto a_acc = a_buf.template get_access<sycl::access::mode::read_write>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                a_acc[idx] |= b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return *this;
    }

    Lp_parallel_vector_GPU<T>& operator^=(const Lp_parallel_vector_GPU<T>& other) {
        if (this->empty()) return *this;
        if (this->size() != other.size()) throw std::runtime_error("Size mismatch");
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers - using read_write for a_acc
            auto a_acc = a_buf.template get_access<sycl::access::mode::read_write>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                a_acc[idx] ^= b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return *this;
    }

    Lp_parallel_vector_GPU<T>& operator<<=(const Lp_parallel_vector_GPU<T>& other) {
        if (this->empty()) return *this;
        if (this->size() != other.size()) throw std::runtime_error("Size mismatch");
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers - using read_write for a_acc
            auto a_acc = a_buf.template get_access<sycl::access::mode::read_write>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                a_acc[idx] <<= b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return *this;
    }

    Lp_parallel_vector_GPU<T>& operator>>=(const Lp_parallel_vector_GPU<T>& other) {
        if (this->empty()) return *this;
        if (this->size() != other.size()) throw std::runtime_error("Size mismatch");
        
        // Create SYCL buffers
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers - using read_write for a_acc
            auto a_acc = a_buf.template get_access<sycl::access::mode::read_write>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            
            // Create a SYCL kernel to perform the operation
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                a_acc[idx] >>= b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return *this;
    }

    
    
    Lp_parallel_vector_GPU<T> operator==(const Lp_parallel_vector_GPU<T>& other) const {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] == b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator!=(const Lp_parallel_vector_GPU<T>& other) const {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] != b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator<(const Lp_parallel_vector_GPU<T>& other) const {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] < b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator<=(const Lp_parallel_vector_GPU<T>& other) const {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] <= b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator>(const Lp_parallel_vector_GPU<T>& other) const {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] > b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator>=(const Lp_parallel_vector_GPU<T>& other) const {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] >= b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator&&(const Lp_parallel_vector_GPU<T>& other) const {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] && b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator||(const Lp_parallel_vector_GPU<T>& other) const {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] || b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator&(const Lp_parallel_vector_GPU<T>& other) const {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] & b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator|(const Lp_parallel_vector_GPU<T>& other) const {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] | b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator^(const Lp_parallel_vector_GPU<T>& other) const {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] ^ b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator<<(const Lp_parallel_vector_GPU<T>& other) const {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] << b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    Lp_parallel_vector_GPU<T> operator>>(const Lp_parallel_vector_GPU<T>& other) const {
        Lp_parallel_vector_GPU<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        
        if (min_size == 0) return result;
        
        // Create SYCL buffers from the vector data
        sycl::buffer<T, 1> a_buf(this->data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> b_buf(other.data(), sycl::range<1>(min_size));
        sycl::buffer<T, 1> res_buf(result.data(), sycl::range<1>(min_size));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffers
            auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
            auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res_buf.template get_access<sycl::access::mode::write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(min_size), [=](sycl::id<1> idx) {
                res_acc[idx] = a_acc[idx] >> b_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T>& operator++() {
        if (this->empty()) return *this;
        
        // Create SYCL buffer from the vector data
        sycl::buffer<T, 1> buf(this->data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffer
            auto acc = buf.template get_access<sycl::access::mode::read_write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                acc[idx]++;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return *this;
    }
    Lp_parallel_vector_GPU<T>& operator--() {
        if (this->empty()) return *this;
        
        // Create SYCL buffer from the vector data
        sycl::buffer<T, 1> buf(this->data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffer
            auto acc = buf.template get_access<sycl::access::mode::read_write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                acc[idx]--;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return *this;
    }
    Lp_parallel_vector_GPU<T>& operator++(int) {
        if (this->empty()) return *this;
        
        // Create SYCL buffer from the vector data
        sycl::buffer<T, 1> buf(this->data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffer
            auto acc = buf.template get_access<sycl::access::mode::read_write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                acc[idx]++;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return *this;
    }
    Lp_parallel_vector_GPU<T>& operator--(int) {
        if (this->empty()) return *this;
        
        // Create SYCL buffer from the vector data
        sycl::buffer<T, 1> buf(this->data(), sycl::range<1>(this->size()));
        
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffer
            auto acc = buf.template get_access<sycl::access::mode::read_write>(h);
            
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                acc[idx]--;
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return *this;
    }
    Lp_parallel_vector_GPU<T> operator~() {
        if (this->empty()) return *this;
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        // Create SYCL buffer from the vector data
        sycl::buffer<T, 1> a(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res(result.data(), sycl::range<1>(this->size()));

        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffer
            auto a_acc = a.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res.template get_access<sycl::access::mode::write>(h);
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = ~a_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }
    Lp_parallel_vector_GPU<T> operator!() {
        if (this->empty()) return *this;
        Lp_parallel_vector_GPU<T> result;
        result.resize(this->size());
        // Create SYCL buffer from the vector data
        sycl::buffer<T, 1> a(this->data(), sycl::range<1>(this->size()));
        sycl::buffer<T, 1> res(result.data(), sycl::range<1>(this->size()));
        // Submit a command group to the queue
        q.submit([&](sycl::handler& h) {
            // Get access to the buffer
            auto a_acc = a.template get_access<sycl::access::mode::read>(h);
            auto res_acc = res.template get_access<sycl::access::mode::write>(h);
            // Run the kernel in parallel
            h.parallel_for(sycl::range<1>(this->size()), [=](sycl::id<1> idx) {
                res_acc[idx] = !a_acc[idx];
            });
        });
        
        // Wait for operations to complete
        q.wait();
        
        return result;
    }

    mutable sycl::queue q;  // Make q mutable so it can be modified in const methods
    bool is_gpu;
};

template<typename T, class fun>
static void Lp_if_parallel(Lp_parallel_vector_GPU<T>& vec, fun func)
{
    if (vec.empty()) return;
        
    // Create SYCL buffer from the vector data
    sycl::buffer<T, 1> buf(vec.data(), sycl::range<1>(vec.size()));
    size_t size = vec.size();
    
    // Get the queue from the vector object
    sycl::queue& q = vec.q;
    
    // Submit a command group to the queue
    q.submit([&](sycl::handler& h) {
        // Get access to the buffer
        auto acc = buf.template get_access<sycl::access::mode::read_write>(h);
        
        // Run the kernel in parallel
        h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
            // Only apply function if element is truthy
            if(acc[idx])
                func(acc[idx], idx);
        });
    });
    
    // Wait for operations to complete
    q.wait();   
}



template<class fun>
static void Lp_if_parallel(Lp_parallel_vector_GPU<bool>& vec, fun func)
{
    if (vec.empty()) return;
    std::thread threads[128];
    for(size_t i = 0; i < std::thread::hardware_concurrency(); i++)
    {
        threads[i] = std::thread([&vec, i, &func]() {
            for(size_t j = i; j < vec.size(); j+= std::thread::hardware_concurrency())
            if(static_cast<bool>(vec[j])) {
                bool val = true; // Create a temporary variable that can be modified
                func(val, j);
            }
        });
    }
    for(size_t i = 0; i < std::thread::hardware_concurrency(); i++) {
        if(threads[i].joinable()) {
            threads[i].join();
        }
    }
}

