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
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class DispatchPolicy,
  class TileShape_,
  class ElementQ_,
  class StrideQ_,
  class ElementK_,
  class StrideK_,
  class ElementV_,
  class StrideV_,
  class TiledMma_,
  class GmemTiledCopyQ_,
  class GmemTiledCopyK_,
  class GmemTiledCopyV_>
struct CollectiveMmaAttention {
  static_assert(cutlass::detail::dependent_false<ElementQ_>, "Could not find a mainloop specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Stages,
  class TileShape_,
  class ElementQ_,
  class StrideQ_,
  class ElementK_,
  class StrideK_,
  class ElementV_,
  class StrideV_,
  class TiledMma_,
  class GmemTiledCopyQ_,
  class GmemTiledCopyK_,
  class GmemTiledCopyV_>
struct CollectiveMmaAttention<
    MainloopIntelPVC<Stages>,
    TileShape_,
    ElementQ_,
    StrideQ_,
    ElementK_,
    StrideK_,
    ElementV_,
    StrideV_,
    TiledMma_,
    GmemTiledCopyQ_,
    GmemTiledCopyK_,
    GmemTiledCopyV_>
{
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopIntelPVC<Stages>;
  using WorkgroupTileShape = TileShape_;
  using ElementQ = ElementQ_;
  using StrideQ = StrideQ_;
  using ElementK = ElementK_;
  using StrideK = StrideK_;
  using ElementV = ElementV_;
  using StrideV = StrideV_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyQ = GmemTiledCopyQ_;
  using GmemTiledCopyK = GmemTiledCopyK_;
  using GmemTiledCopyV = GmemTiledCopyV_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;

  using MmaAtomShape = typename TiledMma::AtomShape_MNK;
  using SubgroupTileShape = decltype(tile_shape(TiledMma()));

  static constexpr auto sg_per_wg_m = get<0>(WorkgroupTileShape{}) / get<0>(SubgroupTileShape{});
  static constexpr auto sg_per_wg_n = get<1>(WorkgroupTileShape{}) / get<1>(SubgroupTileShape{});

  static constexpr uint32_t MaxThreadsPerBlock =
          cute::size(WorkgroupTileShape{}) / cute::size(SubgroupTileShape{}) * SubgroupSize;

  static constexpr int FragsM = get<0>(SubgroupTileShape{}) / get<0>(MmaAtomShape()); // A frags per sub_group
  static constexpr int FragsN = get<1>(SubgroupTileShape{}) / get<1>(MmaAtomShape()); // B frags per sub_group
  static constexpr int FragsK = get<2>(SubgroupTileShape{}) / get<2>(MmaAtomShape());

  // Calculate the vector width based on the amount of registers 
  // required per work item by dividing the total fragment size by 
  // the sub_group size.
  static constexpr int VecC = (get<1>(MmaAtomShape()) * get<0>(MmaAtomShape())) / SubgroupSize;
  static constexpr int VecA = (get<0>(MmaAtomShape()) * get<2>(MmaAtomShape())) / SubgroupSize;
  static constexpr int VecB = (get<1>(MmaAtomShape()) * get<2>(MmaAtomShape())) / SubgroupSize;

  // Host side kernel arguments
  struct Arguments {
    ElementQ const* ptr_Q;
    StrideQ dQ;
    ElementK const* ptr_K;
    StrideK dK;
    ElementV const* ptr_V;
    StrideV dV;
  };

  struct Params {
    using XE_Copy_Q = decltype(make_xe_2d_copy<GmemTiledCopyQ>(make_tensor(static_cast<ElementQ const*>(nullptr), 
                                repeat_like(StrideQ{}, int32_t(0)), StrideQ{})));
    using XE_Copy_K = decltype(make_xe_2d_copy<GmemTiledCopyK>(make_tensor(static_cast<ElementK const*>(nullptr), 
                                repeat_like(StrideK{}, int32_t(0)), StrideK{})));
    using XE_Copy_V = decltype(make_xe_2d_copy<GmemTiledCopyV>(make_tensor(static_cast<ElementV const*>(nullptr), 
                                repeat_like(StrideV{}, int32_t(0)), StrideV{})));
    XE_Copy_Q gmem_tiled_copy_q;
    XE_Copy_K gmem_tiled_copy_k;
    XE_Copy_V gmem_tiled_copy_v;
  };

  //
  // Methods
  //

  CollectiveMmaAttention() = default;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    (void) workspace;

    auto [batch, num_heads, seq_len, head_size] = problem_shape;

    Tensor tensorQ = make_tensor(args.ptr_Q, make_layout(make_shape(batch, num_heads, seq_len, head_size), args.dQ));
    Tensor tensorK = make_tensor(args.ptr_K, make_layout(make_shape(batch, num_heads, seq_len, head_size), args.dK));
    Tensor tensorV = make_tensor(args.ptr_V, make_layout(make_shape(batch, num_heads, seq_len, head_size), args.dV));

    typename Params::XE_Copy_Q copyQ = make_xe_2d_copy<GmemTiledCopyQ>(tensorQ);
    typename Params::XE_Copy_K copyK = make_xe_2d_copy<GmemTiledCopyK>(tensorK);
    typename Params::XE_Copy_V copyV = make_xe_2d_copy<GmemTiledCopyK>(tensorV);
    return Params{copyQ, copyK, copyV};
  }

  template <
    class CopyAtom,
    class Tensor
  >
  CUTLASS_DEVICE void
  prefetch(CopyAtom const &copy_atom, Tensor gT) {
    cute::prefetch(copy_atom, gT);
  }

  template <
    class CopyAtom,
    class Tensor,
    class Fragment
  >
  CUTLASS_DEVICE void
  load(CopyAtom const &copy_atom, Tensor gT, Fragment& frag) {
    copy(copy_atom, gT, frag);
  }

  template <
  class FragAccum,
  class TensorQ,
  class TensorK,
  class FragSrc
  >
  CUTLASS_DEVICE void
  mmaQK(
    FragAccum& accum,
    TensorQ gQ,
    TensorK gK,
    FragSrc const &frag_src,
    int const &head_size,
    Params const &params) {

    TiledMma tiled_mma;

    constexpr int version = is_same_v<GmemTiledCopyK, XE_2D_U16x16x16x2x1_V> ? 1 : 2;

    Tensor tQr = make_tensor<typename TiledMma::ValTypeA>(Shape<Int<get<0>(SubgroupTileShape{}) * FragsK>, Int<1>>{});
    Tensor tKr = make_tensor<typename TiledMma::ValTypeB>(Shape<Int<get<2>(SubgroupTileShape{}) * version>, Int<FragsN / version>>{});

    Tensor tQr_view = make_tensor(static_cast<decltype(tQr) &&>(tQr).data(),
                            Shape<Int<VecA>, Int<FragsM>, Int<FragsK>>{});
    Tensor tKr_view = make_tensor(static_cast<decltype(tKr) &&>(tKr).data(),
                            Shape<Int<VecB>, Int<FragsN>, Int<FragsK>>{},
                            Stride<_1, Int<VecB * FragsK>, Int<VecB>>{});

    // Prefetch K
    int prefetch_idx = 0;
    // for(int i = 0; i < DispatchPolicy::Stages; i++, prefetch_idx += get<2>(SubgroupTileShape{})) {
    //   prefetch(params.gmem_tiled_copy_q, gQ(_, _, prefetch_idx));
    //   prefetch(params.gmem_tiled_copy_k, gK(_, _, prefetch_idx));
    // }

    CUTLASS_PRAGMA_UNROLL
    for (int head_tile = 0; head_tile < head_size; head_tile += get<2>(SubgroupTileShape{}), prefetch_idx += get<2>(SubgroupTileShape{})) {
      load(params.gmem_tiled_copy_q, gQ(_, _, head_tile), tQr);
      load(params.gmem_tiled_copy_k, gK(_, _, head_tile), tKr);

      cute::gemm(tiled_mma, accum, tQr_view, tKr_view, frag_src);

      // prefetch(params.gmem_tiled_copy_q, gQ(_, _, prefetch_idx));
      // prefetch(params.gmem_tiled_copy_k, gK(_, _, prefetch_idx));
    }
  }

  template <
  class FragAccum,
  class FragP,
  class TensorV,
  class FragSrc
  >
  CUTLASS_DEVICE void
  mmaPV(
    FragAccum& accum,
    FragP const &tPr,
    TensorV gV,
    FragSrc const &frag_src,
    int const &reduce_dim,
    Params const &params) {

    TiledMma tiled_mma;

    constexpr int version = is_same_v<GmemTiledCopyV, XE_2D_U16x16x16x2x1_V> ? 1 : 2;

    Tensor tVr = make_tensor<typename TiledMma::ValTypeB>(Shape<Int<get<2>(SubgroupTileShape{}) * version>, 
                                                                Int<FragsN / version>>{});

    Tensor tVr_view = make_tensor(static_cast<decltype(tVr) &&>(tVr).data(),
                            Shape<Int<VecB>, Int<FragsN>, Int<FragsK>>{},
                            Stride<_1, Int<VecB * FragsK>, Int<VecB>>{});

    // Prefetch K
    int prefetch_idx = 0;
    // for(int i = 0; i < DispatchPolicy::Stages; i++, prefetch_idx += get<2>(SubgroupTileShape{})) {
    //   prefetch(params.gmem_tiled_copy_v, gV(_, _, prefetch_idx));
    // }

    Tensor tPr_view = make_tensor<typename TiledMma::ValTypeA>(Shape<Int<VecA>, Int<FragsM>, Int<FragsK>>{});

    CUTLASS_PRAGMA_UNROLL
    for (int idx = 0, z2 = 0; idx < reduce_dim; z2 += FragsK) {
      load(params.gmem_tiled_copy_v, gV(_, _, idx), tVr);

      // initialize the P view
      for(int x = 0; x < VecC; x++) {
        for (int y = 0; y < FragsM; y++) {
          for (int z = 0; z < FragsK; z++) {
            tPr_view(x, y, z) = tPr(x, y, z2);
          }
        }
      }

      cute::gemm(tiled_mma, accum, tPr_view, tVr_view, frag_src);

      // prefetch(params.gmem_tiled_copy_v, gV(_, _, prefetch_idx));

      idx += get<2>(SubgroupTileShape{});
      prefetch_idx += get<2>(SubgroupTileShape{});
    }
  }

};

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
