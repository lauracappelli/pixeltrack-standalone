//  author: Felice Pantaleo, CERN, 2018
#include <cassert>
#include <iostream>
#include <new>

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "SYCLCore/GPUSimpleVector.h"

void vector_pushback(GPU::SimpleVector<int> *foo, sycl::nd_item<3> item_ct1) {
  auto index = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
  foo->push_back(index);
}

void vector_reset(GPU::SimpleVector<int> *foo) { foo->reset(); }

void vector_emplace_back(GPU::SimpleVector<int> *foo, sycl::nd_item<3> item_ct1) {
  auto index = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
  foo->emplace_back(index);
}

int main() {
  auto maxN = 10000;
  GPU::SimpleVector<int> *obj_ptr = nullptr;
  GPU::SimpleVector<int> *d_obj_ptr = nullptr;
  GPU::SimpleVector<int> *tmp_obj_ptr = nullptr;
  int *data_ptr = nullptr;
  int *d_data_ptr = nullptr;

  obj_ptr = sycl::malloc_host<GPU::SimpleVector<int>>(1, dpct::get_default_queue());
  ((data_ptr = sycl::malloc_host<int>(maxN * sizeof(int), dpct::get_default_queue()), 0));
  ((d_data_ptr = sycl::malloc_device<int>(maxN * sizeof(int), dpct::get_default_queue()), 0));

  auto v = GPU::make_SimpleVector(obj_ptr, maxN, data_ptr);

  tmp_obj_ptr = sycl::malloc_host<GPU::SimpleVector<int>>(1, dpct::get_default_queue());
  GPU::make_SimpleVector(tmp_obj_ptr, maxN, d_data_ptr);
  assert(tmp_obj_ptr->size() == 0);
  assert(tmp_obj_ptr->capacity() == static_cast<int>(maxN));

  d_obj_ptr = sycl::malloc_device<GPU::SimpleVector<int>>(1, dpct::get_default_queue());
  // ... and copy the object to the device.
  dpct::get_default_queue().memcpy(d_obj_ptr, tmp_obj_ptr, sizeof(GPU::SimpleVector<int>)).wait();

  int numBlocks = 5;
  int numThreadsPerBlock = 256;
  /*
  DPCT1049:162: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, numBlocks) * sycl::range(1, 1, numThreadsPerBlock),
                                    sycl::range(1, 1, numThreadsPerBlock)),
                     [=](sycl::nd_item<3> item_ct1) { vector_pushback(d_obj_ptr, item_ct1); });
  });
  dpct::get_current_device().queues_wait_and_throw();

  dpct::get_default_queue().memcpy(obj_ptr, d_obj_ptr, sizeof(GPU::SimpleVector<int>)).wait();

  assert(obj_ptr->size() == (numBlocks * numThreadsPerBlock < maxN ? numBlocks * numThreadsPerBlock : maxN));
  /*
  DPCT1049:164: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, numBlocks) * sycl::range(1, 1, numThreadsPerBlock),
                                    sycl::range(1, 1, numThreadsPerBlock)),
                     [=](sycl::nd_item<3> item_ct1) { vector_reset(d_obj_ptr); });
  });
  dpct::get_current_device().queues_wait_and_throw();

  dpct::get_default_queue().memcpy(obj_ptr, d_obj_ptr, sizeof(GPU::SimpleVector<int>)).wait();

  assert(obj_ptr->size() == 0);

  /*
  DPCT1049:166: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, numBlocks) * sycl::range(1, 1, numThreadsPerBlock),
                                    sycl::range(1, 1, numThreadsPerBlock)),
                     [=](sycl::nd_item<3> item_ct1) { vector_emplace_back(d_obj_ptr, item_ct1); });
  });
  dpct::get_current_device().queues_wait_and_throw();

  dpct::get_default_queue().memcpy(obj_ptr, d_obj_ptr, sizeof(GPU::SimpleVector<int>)).wait();

  assert(obj_ptr->size() == (numBlocks * numThreadsPerBlock < maxN ? numBlocks * numThreadsPerBlock : maxN));

  dpct::get_default_queue().memcpy(data_ptr, d_data_ptr, obj_ptr->size() * sizeof(int)).wait();
  sycl::free(obj_ptr, dpct::get_default_queue());
  sycl::free(data_ptr, dpct::get_default_queue());
  sycl::free(tmp_obj_ptr, dpct::get_default_queue());
  sycl::free(d_data_ptr, dpct::get_default_queue());
  sycl::free(d_obj_ptr, dpct::get_default_queue());
  std::cout << "TEST PASSED" << std::endl;
  return 0;
}
