#include <sycl/sycl.hpp>

sycl::vec<unsigned int, 4> my_srand (sycl::vec<unsigned int, 4> newseed);
sycl::vec<int, 4> my_rand (sycl::vec<unsigned int, 4> seed);
sycl::vec<float, 4> my_randf (sycl::vec<unsigned int, 4> seed);
sycl::vec<int, 4> random_range(int min, int max, unsigned int time);
sycl::vec<float, 4> random_rangef(float min, float max, unsigned int time);