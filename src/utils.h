#pragma once

#include "tensor.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>

#include <cassert>
#include <fstream>
#include <ostream>
#include <iostream>
#include <ranges>
#include <string>
#include <vector>

namespace optimization {
constexpr double kEps = 1e-10;

template <class Alloc, class VectorAlloc>
void writeTrajectoryToFiles(
    const std::vector<std::vector<double, Alloc>, VectorAlloc>& trajectory) {
  assert((trajectory.size() == 4));

  std::ofstream fileX("trajectory_x.txt");
  std::ofstream fileY("trajectory_y.txt");

  if (!fileX.is_open() || !fileY.is_open()) {
    throw std::runtime_error("Error opening files");
  }

  for (size_t i = 0; i < trajectory[0].size(); ++i) {
    fileX << trajectory[0][i] << "\n";
    fileY << trajectory[1][i] << "\n";
  }

  fileX.close();
  fileY.close();
}

template <typename T, class Alloc>
optimization::Tensor<T, Alloc> readSaveFile(const std::string& fileName) try {
  // create file if it does not exist
  optimization::Tensor<T, Alloc> result{};
  { std::ofstream file(fileName, std::ofstream::app); }
  std::ifstream file(fileName);
  boost::archive::text_iarchive ia(file);
  ia >> result;
  return result;
} catch (const boost::archive::archive_exception& e) {
  std::cout << e.what() << "\n";
  return {};
}

template <typename T, class Alloc>
void writeToSaveFile(const std::string& fileName,
                     const optimization::Tensor<T, Alloc>& save) try {
  std::ofstream file(fileName, std::ofstream::out | std::ofstream::trunc);
  boost::archive::text_oarchive oa(file);
  oa << save;
} catch (const boost::archive::archive_exception& e) {
  std::cout << e.what() << "\n";
}

}  // namespace optimization

namespace std {
template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
  for (auto item : vec) {
    os << item << ",";
  }
  return os;
}
}  // namespace std
