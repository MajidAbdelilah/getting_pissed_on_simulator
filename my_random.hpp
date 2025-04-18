#include <sycl/sycl.hpp>

sycl::vec<unsigned int, 4> my_srand (sycl::vec<unsigned int, 4> newseed);
sycl::vec<int, 4> my_rand (sycl::vec<unsigned int, 4> seed);
sycl::vec<float, 4> my_randf (sycl::vec<unsigned int, 4> seed);
unsigned int my_srand (unsigned int newseed);
int my_rand (unsigned int seed);
float my_randf (unsigned int seed);
int random_range(int min, int max, unsigned int time);
#ifdef icpx
extern SYCL_EXTERNAL 
#endif
float random_rangef(float min, float max, unsigned int time);
sycl::vec<int, 4> random_range(sycl::vec<int, 4> min, sycl::vec<int, 4> max, unsigned int time);
#ifdef icpx
extern SYCL_EXTERNAL 
#endif
sycl::vec<float, 4> random_rangef(sycl::vec<float, 4> min, sycl::vec<float, 4> max, unsigned int time);
