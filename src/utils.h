#pragma once

#include <ranges>

namespace optimization {
template <typename R, typename T>
concept ConvertibleInputRangeTo =
    std::ranges::input_range<R> &&
    std::convertible_to<std::ranges::range_value_t<R>, T>;
}
