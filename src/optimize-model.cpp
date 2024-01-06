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