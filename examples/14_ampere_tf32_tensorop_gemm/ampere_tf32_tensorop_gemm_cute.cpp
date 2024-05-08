/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include <cstdlib>
#include <cstdio>

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/device_memory.h"
#include "helper.h"

using namespace cute;

using TileShape = Shape<_128, _128, _32>;

using TiledMma = TiledMMA<
        MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>,
        Layout<Shape<_2,_2,_1>, Stride<_2, _1, _1>>, // 2x2x1 thread group
        Tile<_32,_32,_8>>;                           // 32x32x8 MMA for LDSM, 1x2x1 value group

// Smem
using SmemLayoutAtomA = decltype(
composition(Swizzle <2,3,2> {},
            Layout<Shape<_32, _8>,
                    Stride<_1, _32>>{}));
using SmemCopyAtomA = Copy_Atom<UniversalCopy<float>, float>;
// Gmem
using GmemTiledCopyA = decltype(
make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, float>{},
                Layout<Shape<_16, _8>,
                        Stride<_1, _16>>{},
                Layout<Shape<_4, _1>>{}));

// Smem
using SmemLayoutAtomB = decltype(
composition(Swizzle <2,3,2> {},
            Layout<Shape<_32, _8>,
                    Stride<_1, _32>>{}));
using SmemCopyAtomB = Copy_Atom<UniversalCopy<float>, float>;
// Gmem
using GmemTiledCopyB = decltype(
make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, float>{},
                Layout<Shape<_16, _8>,
                        Stride<_1, _16>>{},
                Layout<Shape<_4, _1>>{}));

using Stages = Int<3>;

using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtomA{},
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Stages{})));
using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtomB{},
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Stages{})));

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = float;  // <- data type of epilogue operations
using ElementInputA = float;                        // <- data type of elements in input matrix A
using ElementInputB = float;                        // <- data type of elements in input matrix B
using ElementOutput = float;                        // <- data type of elements in output matrix D

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

//// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                     // <- data type of output matrix
        128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
        // memory access. For a byte, it's 16
        // elements. This becomes the vector width of
        // math instructions in the epilogue too
        ElementAccumulator,                                // <- data type of accumulator
        ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

using DispatchPolicy = cutlass::gemm::MainloopSm80CpAsync<Stages{}>;

template <typename Gemm_Op>
void
run(Gemm_Op gemm_op)
{
  gemm_op();
}

void test_gemm(int m, int n, int k)
{

  std::cout << "M = " << m << std::endl;
  std::cout << "N = " << n << std::endl;
  std::cout << "K = " << k << std::endl;

  using TA = float;
  using TB = float;
  using TC = float;
  using TI = float;

  auto h_A = std::vector<TA>(m*k);
  auto h_B = std::vector<TB>(n*k);
  auto h_C = std::vector<TC>(m*n);

  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<tfloat32_t>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<tfloat32_t>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>(-1);

  auto d_A = syclcompat::malloc<TA>(h_A.size());
  auto d_B = syclcompat::malloc<TB>(h_B.size());
  auto d_C = syclcompat::malloc<TC>(h_C.size());

  syclcompat::memcpy<TA>(d_A, h_A.data(), h_A.size());
  syclcompat::memcpy<TB>(d_B, h_B.data(), h_B.size());
  syclcompat::memcpy<TC>(d_C, h_C.data(), h_C.size());

  TI alpha = 1.0;
  TI beta  = 0.0;

  double tflops = (2.0*m*n*k) * 1e-12;

  const int timing_iterations = 100;
  GPU_Clock timer;

  //
  // CuTe
  //

  // Define strides (mixed)
  auto dA = make_stride(Int<1>{}, m, Int<1>{});
  auto dB = make_stride(Int<1>{}, n, Int<1>{});
  auto dC = make_stride(Int<1>{}, m, Int<1>{});

  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
          decltype(dC),
          decltype(dC),
          EpilogueOp,
          cutlass::gemm::EpilogueDefault>;

// Mainloop
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
          DispatchPolicy,
          TileShape,
          ElementInputA,
          decltype(dA),
          ElementInputB,
          decltype(dB),
          TiledMma,
          GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,  // A
          GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity   // B
  >;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
          Shape<int, int, int, int>,
          CollectiveMainloop,
          CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  ProblemShapeType cute_problem_size = ProblemShapeType{m, n, k, 1};

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{
          cutlass::gemm::GemmUniversalMode::kGemm,
          cute_problem_size,  // <- problem size of matrix multiplication
          {  d_A, dA, d_B, dB },
          {
                  { alpha, beta },
                  d_C, dC, d_C, dC
          }
  };

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not
  gemm_op.can_implement(arguments);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  gemm_op.initialize(arguments);

  syclcompat::memcpy<TC>(d_C, h_C.data(), h_C.size());

  // Run once (and check)
  run(gemm_op);

  syclcompat::wait_and_throw();

  auto cute_result = std::vector<TC>(h_C.size());
  syclcompat::memcpy<TC>(cute_result.data(), d_C, h_C.size());

  // Timing iterations
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    run(gemm_op);
  }
  syclcompat::wait_and_throw();

  double cute_time = timer.seconds() / timing_iterations;
  printf("CUTLASS_GEMM:     [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / cute_time, cute_time*1000);
}

int main(int argc, char** argv)
{
  int m = 5120;
  if (argc >= 2)
    sscanf(argv[1], "%d", &m);

  int n = 5120;
  if (argc >= 3)
    sscanf(argv[2], "%d", &n);

  int k = 4096;
  if (argc >= 4)
    sscanf(argv[3], "%d", &k);

  test_gemm(m, n, k);

  return 0;
}
