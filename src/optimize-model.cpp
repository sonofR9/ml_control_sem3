/* Copyright (C) 2023-2024 Novak Alexander
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "optimize-model.h"

namespace optimization {
template Tensor<double, RepetitiveAllocator<double>> modelTestEvolution<
    RepetitiveAllocator, std::function<void(std::size_t, double)>>(
    const optimization::GlobalOptions&,
    std::function<void(std::size_t, double)>);
template Tensor<double, RepetitiveAllocator<double>>
modelTestGray<RepetitiveAllocator, std::function<void(std::size_t, double)>>(
    const optimization::GlobalOptions&,
    std::function<void(std::size_t, double)>);

template Tensor<double, RepetitiveAllocator<double>>
modelTestEvolution<RepetitiveAllocator>(const optimization::GlobalOptions&,
                                        decltype(&coutPrint));
template Tensor<double, RepetitiveAllocator<double>>
modelTestGray<RepetitiveAllocator>(const optimization::GlobalOptions&,
                                   decltype(&coutPrint));
}  // namespace optimization