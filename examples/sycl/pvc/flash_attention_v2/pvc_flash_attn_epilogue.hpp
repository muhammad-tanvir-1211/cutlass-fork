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
/*! \file
  \brief Functor performing elementwise operations used by epilogues.
*/

#pragma once

#include <sycl/sycl.hpp>
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/detail/layout.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class DispatchPolicy,
  class... Args
>
class CollectiveEpilogueAttention {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy>, "Could not find an epilogue specialization.");
};

template <
  class CtaTileMNK_,
  class ElementO_,
  class StrideO_,
  class ElementLSE_,
  class StrideLSE_,
  class FusionCallbacks_,
  class CopyOpO_
>
class CollectiveEpilogueAttention<
    IntelPVCEpilogue,
    CtaTileMNK_,
    ElementO_,
    StrideO_,
    ElementLSE_,
    StrideLSE_,
    FusionCallbacks_,
    CopyOpO_
> {
public:
  //
  // Type Aliases
  //
  using DispatchPolicy = IntelPVCEpilogue;
  using CtaTileMNK = CtaTileMNK_;
  using FusionCallbacks = FusionCallbacks_;
  using ElementO = ElementO_;
  using ElementAccumulator = ElementO_;
  using StrideO = StrideO_;
  using ElementLSE = ElementLSE_;
  using StrideLSE = StrideLSE_;
  using CopyOpO = CopyOpO_;

  using ThreadEpilogueOp = typename fusion::FusionCallbacksTraits<FusionCallbacks>::Operation;
  using GmemTiledCopyO = CopyOpO;
  using ElementOutput = typename FusionCallbacks::ElementOutput;
  using ElementCompute = typename FusionCallbacks::ElementCompute;

  static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;

  static_assert(cute::rank(CtaTileMNK{}) == 3, "CtaTileMNK must be rank-3: [CTA_M, CTA_N, CTA_K]");
  static_assert(cute::rank(StrideO{}) == 4, "StrideO must be rank-4: [batch, num_heads, seq_len, head_size]");
  static_assert(cute::rank(StrideLSE{}) == 3, "StrideLSE must be rank-3: [batch, num_heads, seq_len]");

private:
  constexpr static bool is_destination_supported = not cute::is_void_v<ElementO>;

public:

  using EmptyType = cute::tuple<>;

  struct TensorStorageImpl: cute::tuple<EmptyType, EmptyType> {
    using FusionStorage = typename FusionCallbacks::SharedStorage;
    FusionStorage thread;
  };

  struct SharedStorage {
    using TensorStorage = TensorStorageImpl;

    TensorStorage tensors;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;

  // Host side epilogue arguments
  struct Arguments {
    typename FusionCallbacks::Arguments thread{};
    ElementO const* ptr_O;
    StrideO dO;
    ElementLSE* ptr_LSE;
    StrideLSE dLSE;
  };

  // Device side epilogue params
  struct Params {
    using XE_Copy_O = decltype(make_xe_2d_copy<GmemTiledCopyO>(
        make_tensor(static_cast<ElementO const*>(nullptr),
            repeat_like(StrideO{}, int32_t(0)), StrideO{})));

    typename FusionCallbacks::Params thread{};
    XE_Copy_O xe_store_o;
    ElementLSE* ptr_LSE;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {
    auto [batch, num_heads, seq_len, head_size] = problem_shape;

    typename Params::XE_Copy_O xe_store_o = {};

    if constexpr (is_destination_supported) {
      Tensor tensor_O = make_tensor(args.ptr_O, make_layout(make_shape(batch, num_heads, seq_len, head_size), args.dO));
      xe_store_o = make_xe_2d_copy<CopyOpO>(tensor_O);
    }

    return {
      FusionCallbacks::to_underlying_arguments(problem_shape, args.thread, workspace),
      xe_store_o,
      args.ptr_LSE
    };
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream, 
    CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  CollectiveEpilogueAttention(Params const& params_, TensorStorage const& shared_storage_)
      : params(params_), fusion_callbacks(params_.thread, shared_storage_.thread) {}

  template <
  class ProblemShape,
  class TileCoord,
  class FragOut,
  class FragMax,
  class FragSum,
  class TiledMma
  >
  CUTLASS_DEVICE void
  operator() (
    ProblemShape problem_shape,
    TileCoord tile_coord,
    FragOut &out,
    FragMax const &max,
    FragSum const &sum,
    TiledMma tiled_mma,
    ElementLSE const& softmax_scale
  ) {

    Tensor tLSEr = make_fragment_like(sum);

    using MmaAtomShape = typename TiledMma::AtomShape_MNK;
    using SubgroupTileShape = decltype(tile_shape(TiledMma()));

    static constexpr int FragsM = get<0>(SubgroupTileShape{}) / get<0>(MmaAtomShape());
    static constexpr int FragsN = get<1>(SubgroupTileShape{}) / get<1>(MmaAtomShape());

    static constexpr int FragmentSize = (get<0>(MmaAtomShape()) * get<1>(MmaAtomShape())) / SubgroupSize;

    CUTLASS_PRAGMA_UNROLL
    for (int x = 0; x < FragmentSize; x++) {
      CUTLASS_PRAGMA_UNROLL
      for (int y = 0; y < FragsM; y++) {
        
        ElementLSE curr_sum = sum(x, y);
        ElementO scale = curr_sum == 0.f ? 1.f : 1.f / curr_sum;
        
        tLSEr(x, y) = curr_sum == 0.f ? INFINITY : max(x, y) * softmax_scale + logf(curr_sum);
        
        CUTLASS_PRAGMA_UNROLL
        for (int z = 0; z < FragsN; z++) {
          out(x, y, z) *= scale;
        }
      }
    }

    // Indexing variables
    auto [batch, num_heads, seq_len, head_size] = problem_shape;
    auto [batch_coord, num_head_coord, seq_coord, head_size_coord] = tile_coord;

    Tensor tOuti = params.xe_store_o.get_pvc_flash_tensor(
            make_coord(0, 0, seq_coord, 0),
            make_shape(batch, num_heads, Int<FragsM>{}, Int<FragsN>{}),
            make_stride(Int<get<0>(MmaAtomShape{})>{}, Int<get<1>(MmaAtomShape{})>{}));
    auto tOi = tOuti(_, batch_coord, num_head_coord, _, _);

    copy(params.xe_store_o, out, tOi);

    const int lse_offset = seq_coord + 
                          (static_cast<int>(BlockIdxY()) + 
                                static_cast<int>(BlockIdxZ()) * num_heads) * seq_len;

    auto lse_ptr = params.ptr_LSE + lse_offset;

    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    const int lane_id = static_cast<int>(sg.get_local_linear_id());

    // use only 1 work item per sub_group to write
    // lse since all work items within subgroup have
    // the same sum() data stored in registers
    if(lane_id == 0) {
      int count = 0;
      CUTLASS_PRAGMA_UNROLL
      for (int x = 0; x < FragsM; x++) {
        CUTLASS_PRAGMA_UNROLL
        for (int y = 0; y < FragmentSize; y++) {
          *(lse_ptr + count++) = tLSEr(y, x);
        }
      }
    }
  }

private:
  Params const& params;
  FusionCallbacks fusion_callbacks;
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
