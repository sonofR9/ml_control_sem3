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

#pragma once

#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace optimization {

/// @throws std::runtime_error if file does not exist
template <class Tens>
std::pair<Tens, std::size_t> readDataFromFile(
    const std::filesystem::path& inputFilePath) {
  std::ifstream inFile(inputFilePath);

  if (!inFile.is_open()) {
    throw std::runtime_error("Could not open file: " + inputFilePath.string());
  }

  // Skip first line (timestamp)
  inFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  Tens data;
  std::size_t sequentialNumber = 0;

  std::string line;
  while (std::getline(inFile, line)) {
    if (line.empty()) {
      continue;
    }

    std::stringstream ss(line);
    std::vector<double> dataRow;
    double value{};
    while (ss >> value) {
      dataRow.push_back(value);
    }

    if (dataRow.size() == 1) {
      sequentialNumber = static_cast<std::size_t>(dataRow[0]);
    } else {
      for (std::size_t i{0}; i < dataRow.size(); ++i) {
        data.back().push_back(dataRow[i]);
      }
    }
  }

  inFile.close();
  return {data, sequentialNumber};
}

/// @throws std::runtime_error if file does not exist
template <class Tens>
void writeDataToFile(const std::filesystem::path& outputFilePath,
                     const Tens& data, std::size_t sequentialNumber) {
  std::ofstream outFile(outputFilePath);

  if (!outFile.is_open()) {
    throw std::runtime_error("Could not open file: " + outputFilePath.string());
  }

  // Write timestamp
  auto now{std::chrono::system_clock::now()};
  auto inTimeT{std::chrono::system_clock::to_time_t(now)};
  outFile << std::put_time(std::localtime(&inTimeT), "%Y-%m-%d %H:%M:%S")
          << "\n";

  // Write data
  for (const auto& row : data) {
    for (size_t i = 0; i < row.size(); ++i) {
      outFile << std::fixed << std::setprecision(2) << row[i];
      if (i != row.size() - 1) {
        outFile << ",";
      }
    }
    outFile << "\n";
  }

  // Write sequential number
  outFile << sequentialNumber << std::endl;  // NOLINT(performance-avoid-endl)

  outFile.close();
}

}  // namespace optimization
