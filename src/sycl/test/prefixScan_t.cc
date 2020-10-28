#include <cassert>
#include <ios>
#include <iostream>

#include <CL/sycl.hpp>

#include "SYCLCore/prefixScan.h"
using namespace cms::sycltools;

template <typename T>
void testPrefixScan(sycl::nd_item<3> item, uint32_t size, sycl::stream out, T *ws, T *c, T *co, int subgroupSize) {
  auto first = item.get_local_id(2);
  for (auto i = first; i < size; i += item.get_local_range().get(2))
    c[i] = 1;
  item.barrier();

  cms::sycltools::blockPrefixScan(item, c, co, size, ws, out, subgroupSize);
  cms::sycltools::blockPrefixScan(item, c, size, ws, out, subgroupSize);

  //assert(1 == c[0]);
  if (1 != c[0]) {
    out << "failed (testPrefixScan): 1 != c[0] " << sycl::endl;
    return;
  }
  //assert(1 == co[0]);
  if (1 != co[0]) {
    out << "failed (testPrefixScan): 1 != co[0] " << sycl::endl;
    return;
  }

  for (auto i = first + 1; i < size; i += item.get_local_range().get(2)) {
    if (c[i] != c[i - 1] + 1)
      out << "failed " << size << " " << i << " " << item.get_local_range(2) << " " << c[i] << " " << c[i - 1]
          << sycl::endl;

    //assert(c[i] == c[i - 1] + 1);
    if (c[i] != c[i - 1] + 1) {
      out << "failed (testPrefixScan): c[i] != c[i - 1] + 1 " << sycl::endl;
      return;
    }
    //assert(c[i] == i + 1);
    if (c[i] != i + 1) {
      out << "failed (testPrefixScan): c[i] != i + 1 " << sycl::endl;
      return;
    }
    //assert(c[i] == co[i]);
    if (c[i] != co[i]) {
      out << "failed (testPrefixScan): c[i] != co[i] " << sycl::endl;
      return;
    }
  }
}

template <typename T>
void testWarpPrefixScan(sycl::nd_item<3> item, uint32_t size, sycl::stream out, T *c, T *co, int subgroupSize) {
  //assert(size <= 32);
  if (size > 32) {
    out << "failed (testWarpPrefixScan): size > 32 " << sycl::endl;
    return;
  }

  auto i = item.get_local_id(2);
  c[i] = 1;
  item.barrier();

  warpPrefixScan(item, c, co, i, subgroupSize);
  warpPrefixScan(item, c, i, subgroupSize);
  item.barrier();

  //assert(1 == c[0]);
  if (1 != c[0]) {
    out << "failed (testWarpPrefixScan): 1 != c[0] " << sycl::endl;
    return;
  }
  //assert(1 == co[0]);
  if (1 != co[0]) {
    out << "failed (testWarpPrefixScan): 1 != co[0] " << sycl::endl;
    return;
  }

  if (i != 0) {
    if (c[i] != c[i - 1] + 1)
      out << "failed " << size << " " << i << " " << item.get_local_range(2) << " " << c[i] << " " << c[i - 1]
          << sycl::endl;

    //assert(c[i] == c[i - 1] + 1);
    if (c[i] != c[i - 1] + 1) {
      out << "failed (testWarpPrefixScan): c[i] != c[i - 1] + 1 " << sycl::endl;
      return;
    }
    //assert(c[i] == i + 1);
    if (c[i] != (T)i + 1) {
      out << "failed (testWarpPrefixScan): c[i] != i + 1 " << sycl::endl;
      return;
    }
    //assert(c[i] == co[i]);
    if (c[i] != co[i]) {
      out << "failed (testWarpPrefixScan): c[i] != co[i] " << sycl::endl;
      return;
    }
  }
}

void init(sycl::nd_item<3> item, uint32_t *v, uint32_t val, uint32_t n, sycl::stream out) {
  auto i = item.get_group(2) * item.get_local_range().get(2) + item.get_local_id(2);
  if (i < n)
    v[i] = val;
  if (i == 0)
    out << "  init" << sycl::endl;
}

void verify(sycl::nd_item<3> item, uint32_t const *v, uint32_t n, sycl::stream out) {
  auto i = item.get_group(2) * item.get_local_range().get(2) + item.get_local_id(2);
  if (i < n) {
    //assert(v[i] == i + 1);
    if (v[i] != i + 1) {
      out << "failed (verify): v[i] != i + 1 " << sycl::endl;
      return;
    }
  }
  if (i == 0)
    out << "  verify" << sycl::endl;
}

void sycl_exception_handler(sycl::exception_list exceptions) {
  std::ostringstream msg;
  msg << "Caught asynchronous SYCL exception:";
  for (auto const &exc_ptr : exceptions) {
    try {
      std::rethrow_exception(exc_ptr);
    } catch (sycl::exception const &e) {
      msg << '\n' << e.what();
    }
    throw std::runtime_error(msg.str());
  }
}

int main() try {
  sycl::default_selector device_selector;
  sycl::queue queue(device_selector, sycl_exception_handler, sycl::property::queue::in_order());

  // query the device for the maximum workgroup size
  auto workgroupSize = queue.get_device().get_info<sycl::info::device::max_work_item_sizes>();
  // FIXME the OpenCL CPU device reports a maximum workgroup size of 8192,
  // but workgroups bigger than 4096 result in a CL_OUT_OF_RESOURCES error
  const unsigned int workgroupSizeLimit = 4096;
  const unsigned int maxWorkgroupSize = std::min<unsigned int>(workgroupSizeLimit, workgroupSize[2]);
  std::cout << "max workgroup size: " << maxWorkgroupSize << std::endl;

  // query the device for the maximum subgroup size, up to 16
  auto subgroupSizes = queue.get_device().get_info<sycl::info::device::sub_group_sizes>();
  auto subgroupSize = std::min<int>(16, *std::max_element(std::begin(subgroupSizes), std::end(subgroupSizes)));

  std::cout << "warp level" << std::endl;
  // std::cout << "warp 32" << std::endl;
  queue.submit([&](sycl::handler &cgh) {
    sycl::stream out(64 * 1024, 80, cgh);
    sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::local> c_acc(1024, cgh);
    sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::local> co_acc(1024, cgh);

    cgh.parallel_for(
        sycl::nd_range(sycl::range(1, 1, subgroupSize), sycl::range(1, 1, subgroupSize)),
        [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(16)]] {
          testWarpPrefixScan<int>(item, 32, out, c_acc.get_pointer(), co_acc.get_pointer(), subgroupSize);
        });
  });
  queue.wait_and_throw();

  // std::cout << "warp 16" << std::endl;
  queue.submit([&](sycl::handler &cgh) {
    sycl::stream out(64 * 1024, 80, cgh);
    sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::local> c_acc(1024, cgh);
    sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::local> co_acc(1024, cgh);

    cgh.parallel_for(
        sycl::nd_range(sycl::range(1, 1, subgroupSize), sycl::range(1, 1, subgroupSize)),
        [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(16)]] {
          testWarpPrefixScan<int>(item, 16, out, c_acc.get_pointer(), co_acc.get_pointer(), subgroupSize);
        });
  });
  queue.wait_and_throw();

  // std::cout << "warp 5" << std::endl;
  queue.submit([&](sycl::handler &cgh) {
    sycl::stream out(64 * 1024, 80, cgh);
    sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::local> c_acc(1024, cgh);
    sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::local> co_acc(1024, cgh);

    cgh.parallel_for(
        sycl::nd_range(sycl::range(1, 1, subgroupSize), sycl::range(1, 1, subgroupSize)),
        [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(16)]] {
          testWarpPrefixScan<int>(item, 5, out, c_acc.get_pointer(), co_acc.get_pointer(), subgroupSize);
        });
  });
  queue.wait_and_throw();

  std::cout << "block level" << std::endl;
  for (unsigned int bs = 32; bs <= maxWorkgroupSize; bs += 32) {
    for (unsigned int j = 1; j <= maxWorkgroupSize; ++j) {
      queue.submit([&](sycl::handler &cgh) {
        sycl::stream out(64 * 1024, 80, cgh);

        sycl::accessor<uint16_t, 1, sycl::access_mode::read_write, sycl::target::local> ws_acc(32, cgh);
        sycl::accessor<uint16_t, 1, sycl::access_mode::read_write, sycl::target::local> c_acc(1024, cgh);
        sycl::accessor<uint16_t, 1, sycl::access_mode::read_write, sycl::target::local> co_acc(1024, cgh);

        cgh.parallel_for(
            sycl::nd_range(sycl::range(1, 1, bs), sycl::range(1, 1, bs)),
            [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(16)]] {
              testPrefixScan<uint16_t>(
                  item, j, out, ws_acc.get_pointer(), c_acc.get_pointer(), co_acc.get_pointer(), subgroupSize);
            });
      });
      queue.wait_and_throw();

      queue.submit([&](sycl::handler &cgh) {
        sycl::stream out(64 * 1024, 80, cgh);

        sycl::accessor<float, 1, sycl::access_mode::read_write, sycl::target::local> ws_acc(32, cgh);
        sycl::accessor<float, 1, sycl::access_mode::read_write, sycl::target::local> c_acc(1024, cgh);
        sycl::accessor<float, 1, sycl::access_mode::read_write, sycl::target::local> co_acc(1024, cgh);

        cgh.parallel_for(
            sycl::nd_range(sycl::range(1, 1, bs), sycl::range(1, 1, bs)),
            [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(16)]] {
              testPrefixScan<float>(
                  item, j, out, ws_acc.get_pointer(), c_acc.get_pointer(), co_acc.get_pointer(), subgroupSize);
            });
      });
      queue.wait_and_throw();
    }
  }

  // empiric limit
  auto max_items = maxWorkgroupSize * maxWorkgroupSize;
  unsigned int num_items = 10;
  for (int ksize = 1; ksize < 5; ++ksize) {
    // test multiblock
    std::cout << "multiblock" << std::endl;
    num_items *= 10;
    if (num_items > max_items) {
      std::cout << "Error: too many work items requested: " << num_items << " vs " << max_items << std::endl;
      break;
    }

    // declare, allocate, and initialize device-accessible pointers for input and output
    uint32_t *d_in = sycl::malloc_device<uint32_t>(num_items, queue);
    uint32_t *d_out = sycl::malloc_device<uint32_t>(num_items, queue);

    auto nthreads = std::min((int)maxWorkgroupSize, 256);
    auto nblocks = (num_items + nthreads - 1) / nthreads;

    queue.submit([&](sycl::handler &cgh) {
      sycl::stream out(64 * 1024, 80, cgh);

      cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, nblocks * nthreads), sycl::range(1, 1, nthreads)),
                       [=](sycl::nd_item<3> item) { init(item, d_in, 1, num_items, out); });
    });
    queue.wait_and_throw();

    // the block counter
    int32_t *d_pc = sycl::malloc_device<int32_t>(1, queue);

    nthreads = maxWorkgroupSize;
    nblocks = (num_items + nthreads - 1) / nthreads;
    std::cout << "  nthreads: " << nthreads << " nblocks " << nblocks << " numitems " << num_items << std::endl;

    queue.submit([&](sycl::handler &cgh) {
      sycl::stream out(64 * 1024, 80, cgh);

      sycl::accessor<unsigned int, 1, sycl::access_mode::read_write, sycl::target::local> psum_acc(4 * nblocks, cgh);
      sycl::accessor<unsigned int, 1, sycl::access_mode::read_write, sycl::target::local> ws_acc(32, cgh);
      sycl::accessor<bool, 0, sycl::access_mode::read_write, sycl::target::local> isLastBlockDone_acc(cgh);

      cgh.parallel_for(
          sycl::nd_range(sycl::range(1, 1, nblocks * nthreads), sycl::range(1, 1, nthreads)),
          [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(16)]] {
            multiBlockPrefixScan<uint32_t>(item,
                                           d_in,
                                           d_out,
                                           num_items,
                                           d_pc,
                                           ws_acc.get_pointer(),
                                           isLastBlockDone_acc.get_pointer(),
                                           psum_acc.get_pointer(),
                                           out,
                                           subgroupSize);
          });
    });
    queue.wait_and_throw();

    queue.submit([&](sycl::handler &cgh) {
      sycl::stream out(64 * 1024, 80, cgh);

      cgh.parallel_for(sycl::nd_range(sycl::range(1, 1, nblocks * nthreads), sycl::range(1, 1, nthreads)),
                       [=](sycl::nd_item<3> item) { verify(item, d_out, num_items, out); });
    });
    queue.wait_and_throw();
  }  // ksize

  return 0;
} catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
