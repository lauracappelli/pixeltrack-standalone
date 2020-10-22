//  author: Felice Pantaleo, CERN, 2018
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cassert>
#include <iostream>
#include <new>

#include "SYCLCore/GPUSimpleVector.h"
#include "SYCLCore/cudaCheck.h"

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

  cudaCheck((obj_ptr = sycl::malloc_host<GPU::SimpleVector<int>>(1, dpct::get_default_queue()), 0));
  /*
  DPCT1003:160: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  ((data_ptr = sycl::malloc_host<int>(maxN * sizeof(int), dpct::get_default_queue()), 0));
  ((d_data_ptr = sycl::malloc_device<int>(maxN * sizeof(int), dpct::get_default_queue()), 0));

  auto v = GPU::make_SimpleVector(obj_ptr, maxN, data_ptr);

  /*
  DPCT1003:161: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  cudaCheck((tmp_obj_ptr = sycl::malloc_host<GPU::SimpleVector<int>>(1, dpct::get_default_queue()), 0));
  GPU::make_SimpleVector(tmp_obj_ptr, maxN, d_data_ptr);
  assert(tmp_obj_ptr->size() == 0);
  assert(tmp_obj_ptr->capacity() == static_cast<int>(maxN));

  cudaCheck((d_obj_ptr = sycl::malloc_device<GPU::SimpleVector<int>>(1, dpct::get_default_queue()), 0));
  // ... and copy the object to the device.
  cudaCheck((dpct::get_default_queue().memcpy(d_obj_ptr, tmp_obj_ptr, sizeof(GPU::SimpleVector<int>)).wait(), 0));

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
  /*
  DPCT1010:163: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  */
  cudaCheck(0);
  cudaCheck((dpct::get_current_device().queues_wait_and_throw(), 0));

  cudaCheck((dpct::get_default_queue().memcpy(obj_ptr, d_obj_ptr, sizeof(GPU::SimpleVector<int>)).wait(), 0));

  assert(obj_ptr->size() == (numBlocks * numThreadsPerBlock < maxN ? numBlocks * numThreadsPerBlock : maxN));
  /*
  DPCT1049:164: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, numBlocks) * sycl::range(1, 1, numThreadsPerBlock),
                                    sycl::range(1, 1, numThreadsPerBlock)),
                     [=](sycl::nd_item<3> item_ct1) { vector_reset(d_obj_ptr); });
  });
  /*
  DPCT1010:165: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  */
  cudaCheck(0);
  cudaCheck((dpct::get_current_device().queues_wait_and_throw(), 0));

  cudaCheck((dpct::get_default_queue().memcpy(obj_ptr, d_obj_ptr, sizeof(GPU::SimpleVector<int>)).wait(), 0));

  assert(obj_ptr->size() == 0);

  /*
  DPCT1049:166: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, numBlocks) * sycl::range(1, 1, numThreadsPerBlock),
                                    sycl::range(1, 1, numThreadsPerBlock)),
                     [=](sycl::nd_item<3> item_ct1) { vector_emplace_back(d_obj_ptr, item_ct1); });
  });
  /*
  DPCT1010:167: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
  */
  cudaCheck(0);
  cudaCheck((dpct::get_current_device().queues_wait_and_throw(), 0));

  cudaCheck((dpct::get_default_queue().memcpy(obj_ptr, d_obj_ptr, sizeof(GPU::SimpleVector<int>)).wait(), 0));

  assert(obj_ptr->size() == (numBlocks * numThreadsPerBlock < maxN ? numBlocks * numThreadsPerBlock : maxN));

  /*
  DPCT1003:168: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  cudaCheck((dpct::get_default_queue().memcpy(data_ptr, d_data_ptr, obj_ptr->size() * sizeof(int)).wait(), 0));
  /*
  DPCT1003:169: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  cudaCheck((sycl::free(obj_ptr, dpct::get_default_queue()), 0));
  /*
  DPCT1003:170: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  cudaCheck((sycl::free(data_ptr, dpct::get_default_queue()), 0));
  /*
  DPCT1003:171: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  cudaCheck((sycl::free(tmp_obj_ptr, dpct::get_default_queue()), 0));
  /*
  DPCT1003:172: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  cudaCheck((sycl::free(d_data_ptr, dpct::get_default_queue()), 0));
  /*
  DPCT1003:173: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
  */
  cudaCheck((sycl::free(d_obj_ptr, dpct::get_default_queue()), 0));
  std::cout << "TEST PASSED" << std::endl;
  return 0;
}
