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

#include <string>
#include <tuple>
#include <utility>

namespace optimization {
template <typename Sig>
struct signature;
template <typename Ret, typename... Args>
struct signature<Ret(Args...)> {
  using type = std::tuple<Args...>;
  using ret = Ret;
};

template <typename Ret, typename Obj, typename... Args>
struct signature<Ret (Obj::*)(Args...)> {
  using type = std::tuple<Args...>;
  using ret = Ret;
};

template <typename Ret, typename Obj, typename... Args>
struct signature<Ret (Obj::*)(Args...) const> {
  using type = std::tuple<Args...>;
  using ret = Ret;
};

template <typename Fun>
concept is_fun = std::is_function_v<Fun>;

template <typename Fun>
concept is_mem_fun = std::is_member_function_pointer_v<std::decay_t<Fun>>;

template <typename Fun>
concept is_functor = std::is_class_v<std::decay_t<Fun>> &&
                     requires(Fun&& t) { &std::decay_t<Fun>::operator(); };

template <is_functor T>
auto arguments(T&& t)
    -> signature<decltype(&std::decay_t<T>::operator())>::type;

template <is_functor T>
auto arguments(const T& t)
    -> signature<decltype(&std::decay_t<T>::operator())>::type;

template <is_fun T>
auto arguments(const T& t) -> signature<T>::type;

template <is_mem_fun T>
auto arguments(T&& t) -> signature<std::decay_t<T>>::type;

template <is_mem_fun T>
auto arguments(const T& t) -> signature<std::decay_t<T>>::type;
}  // namespace optimization
