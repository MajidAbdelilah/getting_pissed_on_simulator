
#include "my_random.hpp"

sycl::vec<unsigned int, 4> my_srand (sycl::vec<unsigned int, 4> newseed) {
    sycl::vec<unsigned int, 4> seed(1);
    seed = newseed.convert<unsigned int>() & sycl::vec<unsigned int, 4>(0x7fffffffU);
    return seed;
}


sycl::vec<int, 4> my_rand (sycl::vec<unsigned int, 4> seed) {
    seed = (seed * sycl::vec<unsigned int, 4>(1103515245U) + sycl::vec<unsigned int, 4>(12345U)) & sycl::vec<unsigned int, 4>(0x7fffffffU);
    return seed.convert<int>();
}

sycl::vec<float, 4> my_randf (sycl::vec<unsigned int, 4> seed) {
    return my_rand(seed).convert<float>() / sycl::vec<float, 4>((float)0x7fffffffU);
}

unsigned int my_srand (unsigned int newseed) {
    unsigned int seed = (1);
    seed = (unsigned int)newseed & (0x7fffffffU);
    return seed;
}
int my_rand (unsigned int seed) {
    seed = (seed * (1103515245U) + (12345U)) & (0x7fffffffU);
    return seed;
}
float my_randf (unsigned int seed) {
    return (float)my_rand(seed) / ((float)0x7fffffffU);
}

int random_range(int min, int max, unsigned int time) {
    if (min > max) {
        std::swap(min, max);
    }
    return my_rand(my_srand((time))) % (max - min + 1) + min;
}

float random_rangef(float min, float max, unsigned int time) {
    if (min > max) {
        std::swap(min, max);
    }
    
    return my_randf((time)) * (max - min) + min;
}

sycl::vec<int, 4> random_range(sycl::vec<int, 4> min, sycl::vec<int, 4> max, unsigned int time){
    for(int i = 0; i < 4; ++i) {
        if (min[i] > max[i]) {
            std::swap(min[i], max[i]);
        }
    }
    return my_rand(my_srand(sycl::vec<unsigned int, 4>(time, time * 1000, time * 2000, time * 3000))) % (max - min + 1) + min;
}

sycl::vec<float, 4> random_rangef(sycl::vec<float, 4> min, sycl::vec<float, 4> max, unsigned int time){
    for(int i = 0; i < 4; ++i) {
        if (min[i] > max[i]) {
            std::swap(min[i], max[i]);
        }
    }
    
    return my_randf(my_srand(sycl::vec<unsigned int, 4>(time, time * 1000, time * 2000, time * 3000))) * (max - min) + min;
}
