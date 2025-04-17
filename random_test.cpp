#include <sycl/sycl.hpp>

#include <random>


class RandomFiller {

public:

  RandomFiller(int *ptr) : ptr_{ptr} {

    std::random_device hwRand;

    std::uniform_int_distribution<> r{1, 100};

    randomNum_ = r(hwRand);

  }

  void operator()(sycl::item<1> item) const {

    ptr_[item.get_id()] = get_random();

  }

  int get_random() const { return randomNum_; }


private:

  int *ptr_;

  int randomNum_;

};

// extern SYCL_EXTERNAL unsigned int time(unsigned int *tloc);


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


int main(int argc, char **argv) {
    
    sycl::queue q(sycl::gpu_selector_v);
    size_t size = 10;
    sycl::vec<float, 4> *p = sycl::malloc_shared<sycl::vec<float, 4>>(size, q);
    for(size_t i = 0; i < size; ++i) {
        p[i] = 0;
    }
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<int> distribution(0,10);
    // *p = 0;
    distribution(rng);
    unsigned int my_time = time(0);
    q.submit([&](sycl::handler &h) {
        
        h.parallel_for<class random_kernel>(sycl::range<1>(size), [=](sycl::id<1> id) {
            // unsigned int seed = my_srand(my_time + id.get(0) * 2);
            // std::cout << "Random number from kernel: " << p[id.get(0)] << std::endl;
            p[id.get(0)] = random_rangef(0, 100, my_time + id.get(0) * 1000);
        });
    }).wait();
    for(size_t i = 0; i < size; ++i) {
        std::cout << "Random number from kernel: " << p[i][0] << ", " << p[i][1] << ", " << p[i][2] << ", " << p[i][3] << std::endl;
    }
    sycl::free(p, q);

    sycl::device device = sycl::device(sycl::gpu_selector_v);
    std::cout << "Device: " << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Device vendor: " << device.get_info<sycl::info::device::vendor>() << std::endl;
    std::cout << "Device version: " << device.get_info<sycl::info::device::version>() << std::endl;
    std::cout << "Device driver version: " << device.get_info<sycl::info::device::driver_version>() << std::endl;
    std::cout << "Device max compute units: " << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
    std::cout << "Device max work group size: " << device.get_info<sycl::info::device::max_work_group_size>() << std::endl;
    sycl::vec<unsigned int, 4> my_time2(time(0), time(0) + 1000, time(0) + 2000, time(0) + 3000);
    sycl::vec<int, 4> random_number = my_rand(my_time2) % 100; // Generate a random number between 0 and 99
    std::cout << "Random number: " << random_number[0] << ", " << random_number[1] << ", " << random_number[2] << ", " << random_number[3] << std::endl;
    return 0;
}