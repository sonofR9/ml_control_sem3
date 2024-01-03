#pragma once

// #define COLLECT_ALLOCATOR_STATS

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#ifdef COLLECT_ALLOCATOR_STATS
#include <format>
#include <fstream>
#include <thread>
#include <unordered_map>
#endif

namespace optimization {

class BadDealloc : public std::bad_alloc {
 public:
  BadDealloc(std::string msg) noexcept : msg_{std::move(msg)} {
  }

  [[nodiscard]] const char* what() const noexcept override {
    return msg_.c_str();
  }

 private:
  std::string msg_;
};

#ifdef COLLECT_ALLOCATOR_STATS
template <typename T>
class AllocatorStatCollector {
 public:
  AllocatorStatCollector() = default;
  ~AllocatorStatCollector() {
    // append to file
    std::ofstream file{"allocator_stat.txt", std::ios::app};
    if (file.is_open()) {
      file << std::this_thread::get_id()
           << std::format(" type size: {}\n", sizeof(T));
      for (const auto& [k, v] : allocs_) {
        file << std::format("size:{} count:{}\n", k, v);
      }
    }
  }

  void insert(std::size_t size) {
    ++allocs_[size];
  }

 private:
  std::unordered_map<std::size_t, std::size_t> allocs_{};
};
#endif

constexpr std::size_t kMaxSize{100'000};

/**
 * @brief takes ~3Mb of space (kMaxSize * sizeof(vector<T*>) to be precise).
 * Fast if you have repetitive allocations of the same size.
 */
template <typename T>
class RepetitiveAllocator {
 public:
  using value_type = T;
  using size_type = size_t;
  using difference_type = ptrdiff_t;

  constexpr RepetitiveAllocator() noexcept = default;
  constexpr RepetitiveAllocator(const RepetitiveAllocator&) noexcept = default;
  constexpr RepetitiveAllocator& operator=(const RepetitiveAllocator&) =
      default;

  template <class Other>
  /*implicit*/ constexpr RepetitiveAllocator(
      const RepetitiveAllocator<Other>&) noexcept {
  }
  constexpr ~RepetitiveAllocator() = default;

  [[nodiscard]] constexpr T* allocate(const std::size_t n) {
    if (n > kMaxSize) [[unlikely]] {
      return allocator_.allocate(n);
    }

    T* result{nullptr};

    auto& freeSize{free_[n]};
    if (!freeSize.empty()) {
      result = freeSize.back();
      freeSize.pop_back();
    } else {
      result = allocator_.allocate(n);
      allocated_.push_back(result);
#ifdef COLLECT_ALLOCATOR_STATS
      statCollector_.insert(n);
#endif
    }
    return result;
  }

  constexpr void deallocate(T* p, std::size_t n) {
    if (n > kMaxSize) [[unlikely]] {
      return allocator_.deallocate(p, n);
    }
    free_[n].push_back(p);
  }

  constexpr void deallocateAll() {
    free_.clear();
    for (auto& p : allocated_) {
      allocator_.deallocate(p);
    }
    allocated_.clear();
    // if (smthInUse) {
    //   throw BadDealloc("Some objects were still in use when deallocated");
    // }
  }

 private:
  static thread_local std::allocator<T> allocator_;

  static thread_local std::vector<std::vector<T*>> free_;
  static thread_local std::vector<T*> allocated_;

#ifdef COLLECT_ALLOCATOR_STATS
  static thread_local AllocatorStatCollector<T> statCollector_;
#endif
};

template <typename T>
thread_local std::allocator<T> RepetitiveAllocator<T>::allocator_{};

template <typename T>
thread_local std::vector<std::vector<T*>> RepetitiveAllocator<T>::free_ =
    std::vector<std::vector<T*>>(kMaxSize);
template <typename T>
thread_local std::vector<T*> RepetitiveAllocator<T>::allocated_{};

#ifdef COLLECT_ALLOCATOR_STATS
template <typename T>
thread_local AllocatorStatCollector<T> RepetitiveAllocator<T>::statCollector_{};
#endif
}  // namespace optimization
