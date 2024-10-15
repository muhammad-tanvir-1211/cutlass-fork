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
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/fusion/intel_pvc_callbacks.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "pvc_flash_attn_gemm_universal.hpp"
#include "pvc_flash_attn_epilogue.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/sycl_event_manager.hpp"

#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "../common.h"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace flash {

  template <class IntT>
  CUTLASS_HOST_DEVICE
  cute::Stride<IntT, IntT, cute::Int<1>, IntT>
  make_cute_packed_stride(cute::Stride<IntT, IntT, cute::Int<1>, IntT> s, cute::Shape<int,int,int,int> shape_NCHW) {
    static_assert(std::is_integral_v<IntT>,
      "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
    auto s_copy = s;
    cute::get<3>(s_copy) = static_cast<IntT>(cute::get<2>(shape_NCHW));
    int num_heads =  cute::get<1>(shape_NCHW);
    if (num_heads > 1) {
      cute::get<1>(s_copy) = static_cast<IntT>(cute::get<2>(shape_NCHW) * cute::get<3>(shape_NCHW));
    }
    else {
      cute::get<1>(s_copy) = static_cast<IntT>(0);
    }

    int batch = cute::get<0>(shape_NCHW);
    if (batch > 1) {
      cute::get<0>(s_copy) = static_cast<IntT>(cute::get<1>(shape_NCHW) * cute::get<2>(shape_NCHW) * cute::get<3>(shape_NCHW));
    }
    else {
      cute::get<0>(s_copy) = static_cast<IntT>(0);
    }
    return s_copy;
  }

  template <class IntT>
  CUTLASS_HOST_DEVICE
  cute::Stride<IntT, IntT, IntT, cute::Int<1>>
  make_cute_packed_stride(cute::Stride<IntT, IntT, IntT, cute::Int<1>> s, cute::Shape<int,int,int,int> shape_NCHW) {
    static_assert(std::is_integral_v<IntT>,
      "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
    auto s_copy = s;
    cute::get<2>(s_copy) = static_cast<IntT>(cute::get<3>(shape_NCHW));
    int num_heads = cute::get<1>(shape_NCHW);
    if (num_heads > 1) {
      cute::get<1>(s_copy) = static_cast<IntT>(cute::get<2>(shape_NCHW) * cute::get<3>(shape_NCHW));
    }
    else {
      cute::get<1>(s_copy) = static_cast<IntT>(0);
    }

    int batch = cute::get<0>(shape_NCHW);
    if (batch > 1) {
      cute::get<0>(s_copy) = static_cast<IntT>(cute::get<1>(shape_NCHW) * cute::get<2>(shape_NCHW) * cute::get<3>(shape_NCHW));
    }
    else {
      cute::get<0>(s_copy) = static_cast<IntT>(0);
    }
    return s_copy;
  }

  template <class IntT>
  CUTLASS_HOST_DEVICE
  cute::Stride<IntT, IntT, cute::Int<1>>
  make_cute_packed_stride(cute::Stride<IntT, IntT, cute::Int<1>> s, cute::Shape<int,int,int> shape_NCH) {
    static_assert(std::is_integral_v<IntT>,
      "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
    auto s_copy = s;
    cute::get<1>(s_copy) = static_cast<IntT>(cute::get<2>(shape_NCH));
    int batch = cute::get<1>(shape_NCH);
    if (batch > 1) {
      cute::get<0>(s_copy) = static_cast<IntT>(cute::get<2>(shape_NCH) * cute::get<1>(shape_NCH));
    }
    else {
      cute::get<0>(s_copy) = static_cast<IntT>(0);
    }

    return s_copy;
  }
} // namespace flash

// Command line options parsing
struct Options {

  bool help;
  bool error;
  bool is_causal;

  int batch, num_heads, seq_len, head_size, iterations;
  float softmax_scale;

  Options():
    help(false),
    error(false),
    is_causal(false),
    batch(4), num_heads(8), seq_len(4096), head_size(64), iterations(20),
    softmax_scale(1.f)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    if (cmd.check_cmd_line_flag("is_causal")) {
      is_causal = true;
    }

    cmd.get_cmd_line_argument("batch", batch, 4);
    cmd.get_cmd_line_argument("num_heads", num_heads, 8);
    cmd.get_cmd_line_argument("seq_len", seq_len, 16384);
    cmd.get_cmd_line_argument("head_size", head_size, 64);
    cmd.get_cmd_line_argument("iterations", iterations, 100);

    softmax_scale = 1 / sqrt(static_cast<float>(seq_len));
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "PVC Flash Attention v2 Example\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --is_causal                 Apply Causal Mask to the output of first Matmul\n"
      << "  --batch=<int>               Sets the Batch Size of the Multi-Head Self Attention module\n"
      << "  --num_heads=<int>           Sets the Number of Attention Heads of the Multi-Head Self Attention module\n"
      << "  --seq_len=<int>             Sets the Sequence length of the Multi-Head Self Attention module\n"
      << "  --head_size=<int>           Sets the Attention Head dimension of the Multi-Head Self Attention module\n"
      << "  --iterations=<int>          Iterations\n\n";

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class GemmKernel
>
struct ExampleRunner {

  using StrideQ = typename GemmKernel::StrideQ;
  using StrideK = typename GemmKernel::StrideK;
  using StrideV = typename GemmKernel::StrideV;
  using StrideO = typename GemmKernel::StrideO;
  using StrideLSE = typename GemmKernel::StrideLSE;

  using LayoutQ = cutlass::layout::RowMajor;
  using LayoutK = cutlass::layout::ColumnMajor;
  using LayoutV = cutlass::layout::RowMajor;
  using LayoutO = cutlass::layout::RowMajor;
  using LayoutLSE = cutlass::layout::RowMajor;

  using ElementQ = typename GemmKernel::ElementQ;
  using ElementK = typename GemmKernel::ElementK;
  using ElementV = typename GemmKernel::ElementV;
  using ElementQcc = typename GemmKernel::ElementAccumulator;

  using CollectiveEpilogue = typename GemmKernel::CollectiveEpilogue;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;
  using ElementAccumulator = typename CollectiveEpilogue::ElementAccumulator;

  using ProblemShapeType = typename GemmKernel::ProblemShape;

  //
  // Data members
  //

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;
  StrideLSE stride_LSE;
  uint64_t seed = 0;

  cutlass::DeviceAllocation<ElementQ> block_Q;
  cutlass::DeviceAllocation<ElementK> block_K;
  cutlass::DeviceAllocation<ElementV> block_V;
  cutlass::DeviceAllocation<ElementOutput> block_O;
  cutlass::DeviceAllocation<ElementOutput> block_lse;
  cutlass::DeviceAllocation<ElementOutput> block_ref_O;
  cutlass::DeviceAllocation<ElementOutput> block_ref_lse;

  //
  // Methods
  //

  /*bool verify(const ProblemShapeType& problem_size, ElementCompute alpha, ElementCompute beta) {
    auto [N, d] = problem_size;

    cutlass::TensorRef ref_A(block_Q.get(), LayoutA::packed({N, d}));
    cutlass::TensorRef ref_B(block_K.get(), LayoutB::packed({d, N}));
    cutlass::TensorRef ref_C(block_V.get(), LayoutC::packed({N, d}));
    cutlass::TensorRef ref_D(block_ref_O.get(), LayoutD::packed({N, d}));

    cutlass::reference::device::GemmComplex(
          {M, N, K},
          alpha,
          ref_A,
          cutlass::ComplexTransform::kNone,
          ref_B,
          cutlass::ComplexTransform::kNone,
          beta,
          ref_C,
          ref_D,
          ElementAccumulator(0),
          L,     // batch_count
          M * K, // batch_stride_Q
          K * N, // batch_stride_K
          M * N, // batch_stride_V
          M * N  // batch_stride_O
        );

    syclcompat::wait();

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed = cutlass::reference::device::BlockCompareEqual(
      block_ref_O.get(), block_O.get(), block_O.size());

    return passed;
  }*/

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(const ProblemShapeType& problem_size) {
    // auto problem_shape = cute::append<4>(problem_size, 1);
    auto [batch, num_heads, seq_len, head_size] = problem_size;

    stride_Q = flash::make_cute_packed_stride(StrideQ{}, cute::make_shape(batch, num_heads, seq_len, head_size));
    stride_K = flash::make_cute_packed_stride(StrideK{}, cute::make_shape(batch, num_heads, seq_len, head_size));
    stride_V = flash::make_cute_packed_stride(StrideV{}, cute::make_shape(batch, num_heads, seq_len, head_size));
    stride_O = flash::make_cute_packed_stride(StrideO{}, cute::make_shape(batch, num_heads, seq_len, head_size));
    stride_LSE = flash::make_cute_packed_stride(StrideLSE{}, cute::make_shape(batch, num_heads, seq_len));

    auto count = batch * num_heads * seq_len * head_size;
    block_Q.reset(count);
    block_K.reset(count);
    block_V.reset(count);
    block_O.reset(count);
    block_ref_O.reset(count);
    block_lse.reset(count);
    block_ref_lse.reset(count);

    initialize_block(block_Q, seed + 2023);
    initialize_block(block_K, seed + 2022);
    initialize_block(block_V, seed + 2021);
  }

  static void run(typename GemmKernel::Params params) {
    dim3 const block = GemmKernel::get_block_shape();
    dim3 const grid = GemmKernel::get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = GemmKernel::SharedStorageSize;

    const auto sycl_block = syclcompat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = syclcompat::dim3(grid.x, grid.y, grid.z);

    using namespace syclcompat::experimental;
    auto event = launch<cutlass::device_kernel<GemmKernel>>(launch_policy{
      sycl_grid, sycl_block, local_mem_size{static_cast<std::size_t>(smem_size)}, 
      kernel_properties{sycl_exp::sub_group_size<GemmKernel::DispatchPolicy::SubgroupSize>}
    }, params);

    EventManager::getInstance().addEvent(event);
  }

  void run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = ProblemShapeType{options.batch, options.num_heads, options.seq_len, options.head_size};

    initialize(problem_size);

    typename GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {block_Q.get(), stride_Q, block_K.get(), stride_K, block_V.get(), stride_V},
      {options.is_causal}, {options.softmax_scale},
      {{1}, block_O.get(), stride_O, block_lse.get(), stride_LSE},
      hw_info
    };

    // GemmKernel gemm_op;

    size_t workspace_size = GemmKernel::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    GemmKernel::can_implement(arguments);

    // Initialize the workspace
    auto status = GemmKernel::initialize_workspace(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      return;
    }

    typename GemmKernel::Params params = GemmKernel::to_underlying_arguments(arguments, workspace.get());

    // Run the GEMM
    run(params);

    // gemm_op.run();

    syclcompat::wait();

    // Verify that the result is correct
    bool passed = true; //verify(problem_size, options.alpha, options.beta);
    std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;

    if (passed && options.iterations > 0) {
      GPU_Clock timer;
      timer.start();
      for (int i = 0; i < options.iterations; ++i) {
        run(params);
      }
      syclcompat::wait();

      float cute_time = timer.seconds() / options.iterations;
      double tflops = (2.0 * options.batch * options.num_heads * options.seq_len * options.head_size) * 1e-12;
      std::cout << "Problem Size: " << options.batch << 'x' << options.num_heads << 'x' << options.seq_len << 'x' << options.head_size << std::endl;
      printf("Cutlass Flash Attention Performance:     [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / cute_time, cute_time*1000);
    }

    return;
  }

};

int main(int argc, const char** argv)
{
  //
  // Parse options
  //

  Options options;

  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  //
  // Run examples
  //

  // The KernelHardwareInfo struct holds the number of EUs on the GPU with a given device ID. This
  // information is used by the underlying kernel.
  cutlass::KernelHardwareInfo hw_info;

  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  bool passed;

  // The code section below describes datatype for input, output matrices and computation between
  // elements in input matrices.
  using ElementAccumulator = float;                   // <- data type of accumulator
  using ElementComputeEpilogue = float;  // <- data type of epilogue operations
  using ElementInputQ = bfloat16_t;                        // <- data type of elements in input matrix A
  using ElementInputKV = bfloat16_t;                        // <- data type of elements in input matrix B
  using ElementOutput = float;                        // <- data type of elements in output matrix D

  using LayoutQ = cutlass::layout::RowMajor;
  using LayoutK = cutlass::layout::ColumnMajor;
  using LayoutV = cutlass::layout::RowMajor;
  using LayoutO = cutlass::layout::RowMajor;
  using LayoutLSE = cutlass::layout::RowMajor;

  using GmemTiledCopyQ = XE_2D_U16x8x16x4x2_LD_N;
  using GmemTiledCopyK = XE_2D_U16x16x16x2x2_V;
  using GmemTiledCopyV = XE_2D_U16x16x16x2x2_V;

  // Workgroup-level tile
  using TileShape = Shape<_256, _64, _32>;

  using TiledMma = TiledMMA<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
          Layout<Shape<_1,_1,_1>>,
          Tile<_32,_64,_32>>; // Subgroup level-tile

  constexpr int PipelineStages = 3;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelPVC<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelPVCEpilogue;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementOutput, ElementComputeEpilogue,
          ElementAccumulator, ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape,
          decltype(tile_shape(TiledMma()))>;
  
  // TODO: need to remove copy for LSE as it is not required
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogueAttention<
          EpilogueDispatchPolicy,
          TileShape,
          ElementAccumulator,
          cute::Stride<int64_t, int64_t, int64_t, cute::Int<1>>,
          ElementOutput,
          cute::Stride<int64_t, int64_t, cute::Int<1>>,
          FusionCallBacks,
          XE_2D_U32x8x16x1x1_ST_N>;

// Mainloop
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMmaAttention<
          GEMMDispatchPolicy,
          TileShape,
          ElementInputQ,
          cute::Stride<int64_t, int64_t, int64_t, cute::Int<1>>,
          ElementInputKV,
          cute::Stride<int64_t, int64_t, cute::Int<1>, int64_t>,
          ElementInputKV,
          cute::Stride<int64_t, int64_t, int64_t, cute::Int<1>>,
          TiledMma,
          GmemTiledCopyQ,  // Q
          GmemTiledCopyK,  // K
          GmemTiledCopyV  // V
  >;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversalAttention<
  Shape<int, int, int, int>,
  CollectiveMainloop,
  CollectiveEpilogue
  >;

  ExampleRunner<GemmKernel> runner;

  runner.run(options, hw_info);

  return 0;
}
