#include <cstddef>
#include <thread>
#include <vector>
#include <functional>
#include <mutex>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <iostream>

template<typename T>
class Lp_parallel_vector: public std::vector<T>
{
public:
    Lp_parallel_vector(): std::vector<T>() {
        num_thread = std::thread::hardware_concurrency();
    };
    ~Lp_parallel_vector() {
        // Only join threads that are joinable (have been started)
        for(size_t i = 0; i < num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
    };

    Lp_parallel_vector(size_t num_elements) : std::vector<T>(num_elements) {
        num_thread = std::thread::hardware_concurrency();
    };
    
    Lp_parallel_vector(const Lp_parallel_vector& other) : std::vector<T>(other) {
        num_thread = other.num_thread;
        // Don't copy threads as they can't be copied
    };
    Lp_parallel_vector& operator=(const Lp_parallel_vector& other) {
        if(this != &other) {
            std::vector<T>::operator=(other);
            num_thread = other.num_thread;
            // Don't copy threads as they can't be copied
        }
        return *this;
    }
    Lp_parallel_vector(const std::vector<T>& other) : std::vector<T>(other) {
        num_thread = std::thread::hardware_concurrency();
    };
    
    Lp_parallel_vector& operator=(const std::vector<T>& other) {
        std::vector<T>::operator=(other);
        return *this;
    }
    Lp_parallel_vector(const std::initializer_list<T>& init) : std::vector<T>(init) {
        num_thread = std::thread::hardware_concurrency();
    };
    
    Lp_parallel_vector& operator=(const std::initializer_list<T>& init) {
        std::vector<T>::operator=(init);
        return *this;
    }

    void fill(T value) {
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, value, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    this->at(j) = value;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
    }

    void fill(T value, size_t size)
    {
        this->resize(size);
        fill(value);
    }

    void fill(std::function<T(T&, size_t)> func) {
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, func, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    this->at(j) = func(this->at(j), j);
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
    }

    void fill(std::function<T(T&, size_t)> func, size_t size)
    {
        this->resize(size);
        fill(func);
    }


    Lp_parallel_vector<T> operator+(const Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) + other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator-(const Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) - other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator*(const Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) * other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator/(const Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) / other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator&&(const Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) && other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator||(const Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) || other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator!()
    {
        Lp_parallel_vector<T> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = !this->at(j);
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<bool> operator==(const Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<bool> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) == other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<bool> operator==(const T& other)
    {
        Lp_parallel_vector<bool> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) == other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<bool> operator!=(const Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<bool> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) != other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<bool> operator!=(const T& other)
    {
        Lp_parallel_vector<bool> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) != other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<bool> operator<(const Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<bool> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) < other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<bool> operator<(const T& other)
    {
        Lp_parallel_vector<bool> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) < other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<bool> operator>(const Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<bool> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) > other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<bool> operator>(const T& other)
    {
        Lp_parallel_vector<bool> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) > other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<bool> operator<=(const Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<bool> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) <= other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<bool> operator<=(const T& other)
    {
        Lp_parallel_vector<bool> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) <= other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<bool> operator>=(const Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<bool> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) >= other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<bool> operator>=(const T& other)
    {
        Lp_parallel_vector<bool> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) >= other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };

    Lp_parallel_vector<T> operator&(Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) & other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator|(Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) | other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator^(Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) ^ other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator%(Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) % other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator<<(Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) << other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator>>(Lp_parallel_vector<T>& other)
    {
        Lp_parallel_vector<T> result;
        auto min_size = std::min(this->size(), other.size());
        result.resize(min_size);
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, &other, min_size, i]() {
                for(size_t j = i; j < min_size; j+= this->num_thread)
                    result[j] = this->at(j) >> other[j];
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator+(const T& other)
    {
        Lp_parallel_vector<T> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) + other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator-(const T& other)
    {
        Lp_parallel_vector<T> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) - other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator*(const T& other)
    {
        Lp_parallel_vector<T> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) * other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator/(const T& other)
    {
        Lp_parallel_vector<T> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) / other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator%(const T& other)
    {
        Lp_parallel_vector<T> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) % other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator&(const T& other)
    {
        Lp_parallel_vector<T> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) & other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator|(const T& other)
    {
        Lp_parallel_vector<T> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) | other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator^(const T& other)
    {
        Lp_parallel_vector<T> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) ^ other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator<<(const T& other)
    {
        Lp_parallel_vector<T> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) << other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator>>(const T& other)
    {
        Lp_parallel_vector<T> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, other, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = this->at(j) >> other;
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };
    Lp_parallel_vector<T> operator~()
    {
        Lp_parallel_vector<T> result;
        result.resize(this->size());
        for(size_t i = 0; i < this->num_thread; i++)
        {
            threads[i] = std::thread([this, &result, i]() {
                for(size_t j = i; j < this->size(); j+= this->num_thread)
                    result[j] = ~this->at(j);
            });
        }
        for(size_t i = 0; i < this->num_thread; i++) {
            if(threads[i].joinable()) {
                threads[i].join();
            }
        }
        return result;        
    };

private:
    size_t num_thread;
    std::thread threads[128];
};


template<typename T, class Func>
static void Lp_if_parallel(Lp_parallel_vector<T> &vec, Func func)
{
    std::thread threads[128];
    for(size_t i = 0; i < std::thread::hardware_concurrency(); i++)
    {
        threads[i] = std::thread([&vec, i, &func]() {
            for(size_t j = i; j < vec.size(); j+= std::thread::hardware_concurrency())
                if(static_cast<bool>(vec[j]))
                    func(static_cast<bool>(vec[j]), j);
        });
    }
    for(size_t i = 0; i < std::thread::hardware_concurrency(); i++) {
        if(threads[i].joinable()) {
            threads[i].join();
        }
    }
}

template<typename T, class Func>
static void Lp_if_single_threaded(Lp_parallel_vector<T>& vec, Func func)
{
    for(size_t j = 0; j < vec.size(); j++)
        if(static_cast<bool>(vec[j]))
            func(static_cast<bool>(vec[j]), j);
}

// Parallel quicksort implementation using a thread pool
template<typename T, typename Func>
void Lp_sort(Lp_parallel_vector<T>& vec, Func comp)
{
    // Check if the vector is empty or has only one element
    if (vec.size() <= 1) {
        return; // Already sorted
    }
    
    // Create a copy of the vector data to work with
    std::vector<T> arr(vec.begin(), vec.end());
    
    // Get number of hardware threads
    size_t num_threads = std::thread::hardware_concurrency();
    
    // Create a thread pool
    std::vector<std::thread> thread_pool;
    
    // Create a mutex for thread synchronization
    std::mutex mutex;
    
    // Create a queue of tasks (ranges to sort)
    std::vector<std::pair<size_t, size_t>> task_queue;
    task_queue.push_back({0, arr.size() - 1});
    
    // Create an atomic counter for active tasks
    std::atomic<size_t> active_tasks(1);
    
    // Create a condition variable for synchronization
    std::condition_variable cv;
    
    // Function to process tasks from the queue
    auto process_tasks = [&]() {
        while (true) {
            // Get a task from the queue
            std::pair<size_t, size_t> task;
            {
                std::unique_lock<std::mutex> lock(mutex);
                
                // Wait until there's a task or all tasks are done
                cv.wait(lock, [&]() {
                    return !task_queue.empty() || active_tasks.load() == 0;
                });
                
                // If all tasks are done, exit
                if (task_queue.empty() && active_tasks.load() == 0) {
                    break;
                }
                
                // Get a task from the queue
                if (!task_queue.empty()) {
                    task = task_queue.back();
                    task_queue.pop_back();
                } else {
                    continue;
                }
            }
            
            // Process the task
            size_t low = task.first;
            size_t high = task.second;
            
            // If the range is small, use sequential sort
            if (high - low < 1000) {
                std::sort(arr.begin() + low, arr.begin() + high + 1, comp);
                active_tasks--;
                cv.notify_all();
                continue;
            }
            
            // Partition the array
            size_t pivot_idx;
            
            // Simple partitioning
            {
                // Choose pivot (middle element)
                T pivot = arr[low + (high - low) / 2];
                
                // Initialize indices
                size_t i = low;
                size_t j = high;
                
                // Partition the array
                while (true) {
                    // Find element on left that should be on right
                    while (i < arr.size() && comp(arr[i], pivot)) i++;
                    
                    // Find element on right that should be on left
                    while (j > 0 && comp(pivot, arr[j])) j--;
                    
                    // If indices crossed, break
                    if (i >= j) {
                        pivot_idx = j;
                        break;
                    }
                    
                    // Swap elements
                    std::swap(arr[i], arr[j]);
                    i++;
                    j--;
                }
            }
            
            // Add new tasks to the queue
            {
                std::lock_guard<std::mutex> lock(mutex);
                
                // Add left sub-array to the queue
                if (pivot_idx > 0 && low < pivot_idx) {
                    task_queue.push_back({low, pivot_idx});
                    active_tasks++;
                }
                
                // Add right sub-array to the queue
                if (pivot_idx < high) {
                    task_queue.push_back({pivot_idx + 1, high});
                    active_tasks++;
                }
            }
            
            // Decrement active tasks counter
            active_tasks--;
            
            // Notify waiting threads
            cv.notify_all();
        }
    };
    
    // Start worker threads
    for (size_t i = 0; i < num_threads; i++) {
        thread_pool.push_back(std::thread(process_tasks));
    }
    
    // Wait for all threads to finish
    for (auto& thread : thread_pool) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    // Copy the sorted data back to the original vector
    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] = arr[i];
    }
}

