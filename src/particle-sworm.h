#include "global.h"

#include <algorithm>
#include <cstddef>

namespace optimization {
/**
 * @brief
 *
 * @tparam N number of parameters
 * @tparam Fit fitness function
 * @tparam P number of in population
 * @tparam B number of best wolfs
 */
template <int N, Regular1OutFunction<Vector<N, double>> Fit, int P = 100,
          int B = 3>
class GrayWolfAlgorithm {
  using Specimen = Vector<N, double>;
  using Population = std::array<Specimen, P>;

 public:
  GrayWolfAlgorithm(Fit fit) : fit_{fit} {
  }

  Vector<N, double> solve(int numIterations) {
    auto population{GrayWolfAlgorithm::generatePopulation()};

    for (int i{0}; i < numIterations; ++i) {
      const auto& best{getBest(population)};
      const double alpha{2.0 * (1 - 1.0 * i / numIterations)};

      for (auto& spec : population) {
        const auto& ksi{GrayWolfAlgorithm::generateKsi()};
        for (int j{0}; j < N; ++j) {
          double qj{spec[j]};
          double res{0};
          for (int k{0}; k < B; ++k) {
            res += best[k][j] -
                   2.0 * ksi[2 * k + 1] *
                       std::abs((2 * ksi[2 * k] - 1) * alpha * best[k][j] - qj);
          }
          qj = res / B;
        }
      }
    }

    return *std::max_element(population.begin(), population.end(),
                             [this](const auto& lhs, const auto& rhs) {
                               return fit_(lhs) > fit_(rhs);
                             });
  }

 private:
  static std::array<double, 2ZU * B> generateKsi() {
    std::array<double, 2ZU * B> result{};
    std::generate(result.begin(), result.end(),
                  []() -> double { return Probability::get(); });
    return result;
  }

  std::array<Specimen, B> getBest(const Population& population) {
    using CalcSpecimen = std::pair<typename Specimen::Iterator, double>;
    std::array<CalcSpecimen, P> calcPop{{population.begin(), 0}};
    for (int i{0}; i < P; ++i) {
      calcPop[i].first = population.begin() + i;
      calcPop[i].second = fit_(population[i]);
    }

    std::array<CalcSpecimen, B> best;
    std::partial_sort_copy(
        population.begin(), population.end(), best.begin(), best.end(),
        [this](const CalcSpecimen& lhs, const CalcSpecimen& rhs) {
          return lhs.second > rhs.second;
        });

    std::array<Specimen, B> result;
    for (int i{0}; i < B; ++i) {
      result[i] = *best[i].first;
    }
    return result;
  }

  static Population generatePopulation() {
    // generate random population (P chromosomes of size N each)
    std::array<Vector<N, double>, P> population;
    std::generate(population.begin(), population.end(),
                  []() -> Vector<N, double> {
                    Vector<N, double> chromosome;
                    std::generate(chromosome.begin(), chromosome.end(),
                                  []() { return DoubleGenerator::get(); });
                    return chromosome;
                  });
    return population;
  }

  Fit fit_;
};
}  // namespace optimization
