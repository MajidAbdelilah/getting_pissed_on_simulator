// #include <sycl/sycl.hpp>
// #include <cmath>
// #include <vector>
// #include <algorithm>

// // Helper function for parallel merge - fixed version
// template <typename T>
// void parallel_merge(sycl::queue &q, T *start, size_t mid, size_t size) {
//     // Create a temporary buffer on the device
//     T* temp = sycl::malloc_device<T>(size, q);
    
//     // Copy data to temporary buffer
//     q.memcpy(temp, start, size * sizeof(T)).wait();
    
//     // Create a buffer to store the merged result
//     T* result = sycl::malloc_device<T>(size, q);
    
//     // Perform parallel merge
//     q.submit([&](sycl::handler &h) {
//         h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
//             size_t i = idx[0];
//             size_t k;
            
//             if (i < mid) {
//                 // Element from first subarray
//                 T val = temp[i];
                
//                 // Count elements in second subarray that are smaller than val
//                 size_t count = 0;
//                 for (size_t j = mid; j < size; j++) {
//                     if (temp[j] < val) {
//                         count++;
//                     }
//                 }
                
//                 // Final position
//                 k = i + count;
//             } else {
//                 // Element from second subarray
//                 T val = temp[i];
                
//                 // Count elements in first subarray that are smaller than or equal to val
//                 size_t count = 0;
//                 for (size_t j = 0; j < mid; j++) {
//                     if (temp[j] <= val) {
//                         count++;
//                     }
//                 }
                
//                 // Final position
//                 k = count + (i - mid);
//             }
            
//             // Store the value in its correct position
//             result[k] = temp[i];
//         });
//     }).wait();
    
//     // Copy result back to the original array
//     q.memcpy(start, result, size * sizeof(T)).wait();
    
//     // Free temporary buffers
//     sycl::free(temp, q);
//     sycl::free(result, q);
// }

// // Improved parallel merge function optimized for GPU
// template <typename T>
// void merge(sycl::queue &q, T *start, size_t mid, size_t size) {
//     // For small arrays, use simpler approach to avoid overhead
//     if (size < 1024) {
//         // Create temporary buffer
//         T* temp = sycl::malloc_device<T>(size, q);
        
//         // Copy data to temporary buffer
//         q.memcpy(temp, start, size * sizeof(T)).wait();
        
//         // Sequential merge on the device using a single work-item
//         q.single_task([=]() {
//             size_t left = 0;
//             size_t right = mid;
//             size_t k = 0;
            
//             // Merge the two sorted subarrays
//             while (left < mid && right < size) {
//                 if (temp[left] <= temp[right]) {
//                     start[k++] = temp[left++];
//                 } else {
//                     start[k++] = temp[right++];
//                 }
//             }
            
//             // Copy remaining elements
//             while (left < mid) {
//                 start[k++] = temp[left++];
//             }
            
//             while (right < size) {
//                 start[k++] = temp[right++];
//             }
//         }).wait();
        
//         // Free temporary buffer
//         sycl::free(temp, q);
//     } else {
//         // For larger arrays use the parallel merge implementation
//         parallel_merge(q, start, mid, size);
//     }
// }

// // Bottom-up parallel merge sort implementation
// template <typename T>
// void parallel_sort(T *start, T *end, sycl::queue &q) {
//     size_t size = end - start;
    
//     // Handle base cases
//     if (size <= 1) {
//         return;
//     }
    
//     // Use bottom-up merge sort algorithm which is more GPU-friendly
//     for (size_t width = 1; width < size; width *= 2) {
//         for (size_t i = 0; i < size; i += 2 * width) {
//             // Determine the boundaries of subarrays to merge
//             size_t mid = std::min(i + width, size);
//             size_t right = std::min(i + 2 * width, size);
//             size_t current_size = right - i;
            
//             // Skip if there's only one subarray
//             if (mid < right && mid > i) {  // Added additional check to avoid infinite loop
//                 // Merge the two subarrays
//                 merge(q, start + i, mid - i, current_size);
//             }
//         }
//     }
// }