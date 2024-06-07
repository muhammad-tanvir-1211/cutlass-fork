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
  \brief Fusion callbacks specializations for the Intel PVC epilogue
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"

#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_store_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class ElementOutput_,
  class ElementCompute_,
  class ElementScalar_,
  FloatRoundStyle RoundStyle_,
  class CtaTileShapeMNK_,
  class EpilogueTile_
>
struct FusionCallbacks<
    epilogue::IntelPVCEpilogue,
    fusion::ScaledAcc<ElementOutput_, ElementCompute_, ElementScalar_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_
> : Sm90EVT<Sm90Compute<multiplies, ElementOutput_, ElementCompute_, RoundStyle_>,
      Sm90ScalarBroadcast<ElementScalar_>,
      Sm90AccFetch
    > {

  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementScalar = ElementScalar_;

  using Impl = 
    Sm90EVT<Sm90Compute<multiplies, ElementOutput, ElementCompute, RoundStyle_>,
      Sm90ScalarBroadcast<ElementScalar>,
      Sm90AccFetch
    >;
  using Operation = fusion::ScaledAcc<ElementOutput, ElementCompute, ElementScalar, RoundStyle_>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar const* alpha_ptr = nullptr;

    operator typename Impl::Arguments() const {
      return
        { // binary op : alpha * acc
          {{alpha}, {alpha_ptr}}, // leaf args : alpha
          {},                     // leaf args : acc
          {}                  // binary args : multiplies
        };   // end binary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

template <
  class ElementOutput_,
  class ElementCompute_,
  class ElementSource_,
  class ElementScalar_,
  FloatRoundStyle RoundStyle_,
  class CtaTileShapeMNK_,
  class EpilogueTile_
>
struct FusionCallbacks<
    epilogue::IntelPVCEpilogue,
    fusion::ScaledC<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_
> : Sm90EVT<Sm90Compute<multiplies, ElementOutput_, ElementCompute_, RoundStyle_>,
      Sm90ScalarBroadcast<ElementScalar_>,
      Sm90SrcFetch<ElementSource_>
    > {

  using Impl = 
    Sm90EVT<Sm90Compute<multiplies, ElementOutput_, ElementCompute_, RoundStyle_>,
      Sm90ScalarBroadcast<ElementScalar_>,
      Sm90SrcFetch<ElementSource_>
    >;
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementSource = ElementSource_;
  using ElementScalar = ElementScalar_;
  using Operation = fusion::ScaledC<ElementOutput, ElementCompute, ElementSource_, ElementScalar, RoundStyle_>;

  struct Arguments {
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* beta_ptr = nullptr;

    operator typename Impl::Arguments() const {
      return
        { // binary op : beta * C
          {{beta}, {beta_ptr}}, // leaf args : beta
          {},                   // leaf args : C
          {}                  // binary args : multiplies
        };   // end binary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;
};

template <
  class ElementOutput_,
  class ElementCompute_,
  class ElementSource_,
  class ElementScalar_,
  FloatRoundStyle RoundStyle_,
  class CtaTileShapeMNK_,
  class EpilogueTile_
>
struct FusionCallbacks<
    epilogue::IntelPVCEpilogue,
    fusion::LinearCombination<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_>,
    CtaTileShapeMNK_,
    EpilogueTile_
> : Sm90LinearCombination<typename cutlass::detail::get_unpacked_element_type<ElementOutput_>::type, 
                          ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_> {

  using Impl = Sm90LinearCombination<typename cutlass::detail::get_unpacked_element_type<ElementOutput_>::type, 
                                    ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_>;
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementSource = ElementSource_;
  using ElementScalar = ElementScalar_;
  using Operation = fusion::LinearCombination<ElementOutput, ElementCompute, ElementSource_, ElementScalar, RoundStyle_>;

  struct Arguments {
    ElementScalar alpha = ElementScalar(1);
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;

    operator typename Impl::Arguments() const {
      return
        {    // ternary op : beta * C + (alpha * acc)
          {{beta}, {beta_ptr}}, // leaf args : beta
          {},                   // leaf args : C
          {                     // binary op : alpha * acc
            {{alpha}, {alpha_ptr}}, // leaf args : alpha
            {},                     // leaf args : acc
            {}                  // binary args : multiplies
          },                    // end binary op
          {} // ternary args : multiply_add
        };   // end ternary op
    }
  };

  // Ctor inheritance
  using Impl::Impl;

  using SharedStorage = typename Impl::SharedStorage;

  using Prologue = FusionCallbacks<epilogue::IntelPVCEpilogue, 
                                  fusion::ScaledC<ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle_>,
                                  CtaTileShapeMNK_,
                                  EpilogueTile_>;
  using PrologueParams = typename Prologue::Params;
  using PrologueSharedStorage = typename Prologue::SharedStorage;
  using PrologueArgs = typename Prologue::Arguments;

  using Epilogue = FusionCallbacks<epilogue::IntelPVCEpilogue, 
                                  fusion::ScaledAcc<ElementOutput, ElementCompute, ElementScalar, RoundStyle_>,
                                  CtaTileShapeMNK_,
                                  EpilogueTile_>;
  using EpilogueParams = typename Epilogue::Params;
  using EpilogueSharedStorage = typename Epilogue::SharedStorage;
  using EpilogueArgs = typename Epilogue::Arguments;

  static constexpr PrologueArgs
  get_prologue_args(Arguments const& args) {
    return PrologueArgs{args.beta / args.alpha, args.beta_ptr};
  }

  static constexpr EpilogueArgs
  get_epilogue_args(Arguments const& args) {
    return EpilogueArgs{args.alpha, args.alpha_ptr};
  }

  template <typename StorageType>
  static constexpr StorageType
  get_storage() {
    StorageType storage;
    return storage;
  }

  struct Params {
    PrologueParams prologue_params;
    EpilogueParams epilogue_params;
    typename Impl::Params params;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return {
      Prologue::to_underlying_arguments(problem_shape, get_prologue_args(args), workspace),
      Epilogue::to_underlying_arguments(problem_shape, get_epilogue_args(args), workspace),
      Impl::to_underlying_arguments(problem_shape, args, workspace)
    };
  };

  FusionCallbacks(Params const& params, SharedStorage const& shared_storage) 
                  : Impl(params.params, shared_storage),
                    prologue(params.prologue_params, get_storage<PrologueSharedStorage>()),
                    epilogue(params.epilogue_params, get_storage<EpilogueSharedStorage>())
                    {}

  Prologue prologue;
  Epilogue epilogue;
};

} // namespace cutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////
