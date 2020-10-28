//  author: Felice Pantaleo, CERN, 2018
#include <cassert>
#include <iostream>
#include <new>

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "SYCLCore/GPUSimpleVector.h"

void vector_pushback(GPU::SimpleVector<int> *foo, sycl::nd_item<3> item) {
  auto index = item.get_local_id(2) + item.get_group(2) * item.get_local_range().get(2);
  foo->push_back(index);
}

void vector_reset(GPU::SimpleVector<int> *foo) { foo->reset(); }

void vector_emplace_back(GPU::SimpleVector<int> *foo, sycl::nd_item<3> item) {
  auto index = item.get_local_id(2) + item.get_group(2) * item.get_local_range().get(2);
  foo->emplace_back(index);
}

int main() {
  auto maxN = 10000;
  GPU::SimpleVector<int> *obj_ptr = nullptr;
  GPU::SimpleVector<int> *d_obj_ptr = nullptr;
  GPU::SimpleVector<int> *tmp_obj_ptr = nullptr;
  int *data_ptr = nullptr;
  int *d_data_ptr = nullptr;

  sycl::queue queue = dpct::get_default_queue();

  obj_ptr = sycl::malloc_host<GPU::SimpleVector<int>>(1, queue);
  ((data_ptr = sycl::malloc_host<int>(maxN * sizeof(int), queue), 0));
  ((d_data_ptr = sycl::malloc_device<int>(maxN * sizeof(int), queue), 0));

  auto v = GPU::make_SimpleVector(obj_ptr, maxN, data_ptr);

  tmp_obj_ptr = sycl::malloc_host<GPU::SimpleVector<int>>(1, queue);
  GPU::make_SimpleVector(tmp_obj_ptr, maxN, d_data_ptr);
  //assert(tmp_obj_ptr->size() == 0);
  //assert(tmp_obj_ptr->capacity() == static_cast<int>(maxN));

  d_obj_ptr = sycl::malloc_device<GPU::SimpleVector<int>>(1, queue);
  // ... and copy the object to the device.
  queue.memcpy(d_obj_ptr, tmp_obj_ptr, sizeof(GPU::SimpleVector<int>)).wait();

  int max_work_group_size = queue.get_device().get_info<sycl::info::device::max_work_group_size>();
  int numThreadsPerBlock = std::min(256, max_work_group_size);
  int numBlocks = 5;
  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range(sycl::range(1, 1, numBlocks * numThreadsPerBlock), sycl::range(1, 1, numThreadsPerBlock)),
        [=](sycl::nd_item<3> item) { vector_pushback(d_obj_ptr, item); });
  });
  queue.wait_and_throw();

  queue.memcpy(obj_ptr, d_obj_ptr, sizeof(GPU::SimpleVector<int>)).wait();

  //assert(obj_ptr->size() == (numBlocks * numThreadsPerBlock < maxN ? numBlocks * numThreadsPerBlock : maxN));
  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range(sycl::range(1, 1, numBlocks * numThreadsPerBlock), sycl::range(1, 1, numThreadsPerBlock)),
        [=](sycl::nd_item<3> item) { vector_reset(d_obj_ptr); });
  });
  queue.wait_and_throw();

  queue.memcpy(obj_ptr, d_obj_ptr, sizeof(GPU::SimpleVector<int>)).wait();

  //assert(obj_ptr->size() == 0);

  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range(sycl::range(1, 1, numBlocks * numThreadsPerBlock), sycl::range(1, 1, numThreadsPerBlock)),
        [=](sycl::nd_item<3> item) { vector_emplace_back(d_obj_ptr, item); });
  });
  queue.wait_and_throw();

  queue.memcpy(obj_ptr, d_obj_ptr, sizeof(GPU::SimpleVector<int>)).wait();

  //assert(obj_ptr->size() == (numBlocks * numThreadsPerBlock < maxN ? numBlocks * numThreadsPerBlock : maxN));

  queue.memcpy(data_ptr, d_data_ptr, obj_ptr->size() * sizeof(int)).wait();
  sycl::free(obj_ptr, queue);
  sycl::free(data_ptr, queue);
  sycl::free(tmp_obj_ptr, queue);
  sycl::free(d_data_ptr, queue);
  sycl::free(d_obj_ptr, queue);
  std::cout << "TEST PASSED" << std::endl;
  return 0;
}
