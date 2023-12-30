#include "function-approximation.h"
#include "model.h"

#include <gtest/gtest.h>

using namespace optimization;
using namespace two_wheeled_robot;

TEST(PiecewiseLinearApproximation, ValidTest) {
  const Tensor<20, double> vec{1, 1, 2, 2, 3, 3, 4, 4, 5,  5,
                               6, 6, 7, 7, 8, 8, 9, 9, 10, 10};
  auto vector2d{approximationFrom1D<10>(vec)};

  Tensor<2, double> expected{1, 1};
  ASSERT_EQ(vector2d[0], expected);
  expected = {10, 10};
  ASSERT_EQ(vector2d[9], expected);

  const PiecewiseLinearApproximation<2, double>& approx{1, vector2d.begin(),
                                                        vector2d.end()};

  for (int i{0}; i < 20; ++i) {
    const double time{1.0 * i / 2};
    const double val{1.0 + time};
    expected = {val, val};
    const auto actual{approx(time)};
    ASSERT_EQ(actual, expected)
        << "actual: " << actual << " expected: " << expected
        << " time: " << time;
  }
}