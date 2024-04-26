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

#include <cute/atom/copy_traits.hpp>
#include <cute/atom/copy_atom.hpp>

#include <cute/arch/copy_xe.hpp>

namespace cute
{
    template <class GTensor>
    struct Copy_Traits<XE_2D_LOAD, GTensor>
    {
        using ThrID = Layout<_1>;
        using NumBits = Int<sizeof(typename GTensor::engine_type::value_type) * 8>;
        // Map from (src-thr,src-val) to bit
        using SrcLayout = Layout<Shape<_1, NumBits>>;
        // Map from (dst-thr,dst-val) to bit
        using DstLayout = Layout<Shape<_1, NumBits>>;
        // Reference map from (thr,val) to bit
        using RefLayout = SrcLayout;

        GTensor tensor;

        template <class TS, class SLayout,
                  class TD, class DLayout>
        CUTE_HOST_DEVICE friend constexpr void
        copy_unpack(Copy_Traits const &traits,
                    Tensor<ViewEngine<ArithmeticTupleIterator<TS>>, SLayout> const &src,
                    Tensor<TD, DLayout> &dst)
        {
            static_assert(is_rmem<TD>::value);
            int H = size<0>(traits.tensor);
            int W = size<1>(traits.tensor) * sizeof(typename TD::value_type);
            auto [y, x, z] = src.data().coord_;
            XE_2D_LOAD::copy(traits.tensor.data() + z, W, H, W, int2_{static_cast<int>(x), static_cast<int>(y)}, &*dst.data());
        }
    };

    template <class GTensor>
    struct Copy_Traits<XE_2D_SAVE, GTensor>
    {
        using ThrID = Layout<_1>;
        using NumBits = Int<sizeof(typename GTensor::engine_type::value_type) * 8>;
        // Map from (src-thr,src-val) to bit
        using SrcLayout = Layout<Shape<_1, NumBits>>;
        // Map from (dst-thr,dst-val) to bit
        using DstLayout = Layout<Shape<_1, NumBits>>;
        // Reference map from (thr,val) to bit
        using RefLayout = SrcLayout;

        GTensor tensor;

        template <class TS, class SLayout,
                  class TD, class DLayout>
        CUTE_HOST_DEVICE friend constexpr void
        copy_unpack(Copy_Traits const &traits,
                    Tensor<TS, SLayout> const &src,
                    Tensor<ViewEngine<ArithmeticTupleIterator<TD>>, DLayout> &dst)
        {
            static_assert(is_rmem<TS>::value);
            int H = size<0>(traits.tensor);
            int W = size<1>(traits.tensor) * sizeof(typename decltype(traits.tensor)::engine_type::value_type);
            auto [y, x, z] = dst.data().coord_;
            XE_2D_SAVE::copy(traits.tensor.data() + z, W, H, W, int2_{static_cast<int>(x), static_cast<int>(y)}, &*src.data());
        }
    };

    template <class Copy, class GEngine, class GLayout>
    auto make_xe_2d_copy(Tensor<GEngine, GLayout> gtensor)
    {
        using GTensor = Tensor<GEngine, GLayout>;
        using Traits = Copy_Traits<Copy, GTensor>;
        Traits traits{gtensor};
        return Copy_Atom<Traits, typename GEngine::value_type>{traits};
    }
}
