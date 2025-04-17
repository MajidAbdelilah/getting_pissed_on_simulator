
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

sycl::vec<int, 4> random_range(int min, int max, unsigned int time) {
    if (min > max) {
        std::swap(min, max);
    }
    return my_rand(my_srand(sycl::vec<unsigned int, 4>(time, time * 1000, time * 2000, time * 3000))) % (max - min + 1) + min;
}

sycl::vec<float, 4> random_rangef(float min, float max, unsigned int time) {
    if (min > max) {
        std::swap(min, max);
    }
    
    return my_randf(my_srand(sycl::vec<unsigned int, 4>(time, time * 1000, time * 2000, time * 3000))) * (max - min) + min;
}
