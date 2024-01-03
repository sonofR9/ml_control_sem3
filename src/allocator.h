#pragma once

#include <cstddef>
#include <exception>
#include <memory>
#include <string>
#include <unordered_map>

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
    T* result{nullptr};
    if (free_.contains(n)) {
      auto [begin, end] = free_.equal_range(n);
      T* result{begin->second};
      free_.erase(begin);
    } else {
      T* result{allocator_.allocate(n)};
    }
    borrowed_.insert({result, n});
    return result;
  }

  constexpr void deallocate(T* p, std::size_t n) {
    borrowed_.erase(p);
    free_.insert({n, p});
  }

  constexpr void deallocateAll() {
    for (auto& [p, n] : free_) {
      allocator_.deallocate(p, n);
    }
    free_.clear();

    bool smthInUse{!borrowed_.empty()};
    for (auto& [p, n] : borrowed_) {
      allocator_.deallocate(p, n);
    }
    borrowed_.clear();
    if (smthInUse) {
      throw BadDealloc("Some objects were still in use when deallocated");
    }
  }

 private:
  static std::allocator<T> allocator_;

  static thread_local std::unordered_multimap<std::size_t, T*> free_;
  static thread_local std::unordered_map<T*, std::size_t> borrowed_;
};

template <typename T>
std::allocator<T> RepetitiveAllocator<T>::allocator_{};

template <typename T>
thread_local std::unordered_multimap<std::size_t, T*>
    RepetitiveAllocator<T>::free_{};
template <typename T>
thread_local std::unordered_map<T*, std::size_t>
    RepetitiveAllocator<T>::borrowed_{};
}  // namespace optimization
