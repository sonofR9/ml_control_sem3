#pragma once

#include "global.h"
#include "tensor.h"

#include <cmath>
#include <concepts>

namespace two_wheeled_robot {
template <typename T, class Alloc>
concept ControlFunctionFullLvalue = requires(
    T fun, const optimization::Tensor<double, Alloc>& state, double time) {
  { fun(state, time) } -> std::same_as<optimization::Tensor<double, Alloc>>;
};

template <typename T, class Alloc>
concept ControlFunctionTimeOnly = requires(T fun, double time) {
  { fun(time) } -> std::same_as<optimization::Tensor<double, Alloc>>;
};

template <typename T, class Alloc>
concept ControlFunctionFullRvalue =
    requires(T fun, optimization::Tensor<double, Alloc>&& state, double time) {
      {
        fun(std::move(state), time)
      } -> std::same_as<optimization::Tensor<double, Alloc>>;
    };

template <typename T, class Alloc>
concept ControlFunctionFullLvaluePreallocated =
    requires(T fun, const optimization::Tensor<double, Alloc>& state,
             double time, optimization::Tensor<double, Alloc>& result) {
      { fun(state, time, result) } -> std::same_as<void>;
    };

template <typename T, class Alloc>
concept ControlFunctionTimeOnlyPreallocated =
    requires(T fun, double time, optimization::Tensor<double, Alloc>& result) {
      { fun(time, result) } -> std::same_as<void>;
    };

template <typename T, class Alloc>
concept ControlFunctionFullRvaluePreallocated =
    requires(T fun, optimization::Tensor<double, Alloc>&& state, double time,
             optimization::Tensor<double, Alloc>& result) {
      { fun(std::move(state), time, result) } -> std::same_as<void>;
    };

template <typename T, class Alloc>
concept ControlFunction =
    ControlFunctionFullLvalue<T, Alloc> || ControlFunctionTimeOnly<T, Alloc> ||
    ControlFunctionFullRvalue<T, Alloc> ||
    // preallocated variants
    ControlFunctionFullLvaluePreallocated<T, Alloc> ||
    ControlFunctionTimeOnlyPreallocated<T, Alloc> ||
    ControlFunctionFullRvaluePreallocated<T, Alloc>;

template <class Alloc, ControlFunction<Alloc> C>
class Model {
 public:
  /**
   * @brief Construct a new Model object
   *
   * @param leftWheelControl control function for left wheel. Can accept either
   * state and time or just time (state as rvalue or lvalue)
   * @param rightWheelControl control function for right wheel. Can accept
   * either state and time or just time (state as rvalue or lvalue)
   * @param r radius of wheels
   * @param a distance between wheels
   */
  explicit Model(C control, double r = 1, double a = 1);

  /**
   * @brief models equations system of robot
   *
   * @param state current state
   * @param time current time
   * @return optimization::StateDerivativesPoint<3> left-hand side of equations
   * system (derivatives of state variables)
   */
  optimization::StateDerivativesPoint<double, Alloc> operator()(
      optimization::Tensor<double, Alloc> state, double time);
  void operator()(
      const optimization::Tensor<double, Alloc>& state, double time,
      optimization::StateDerivativesPoint<double, Alloc>& preallocatedResult);
};

template <class Alloc, ControlFunctionTimeOnly<Alloc> C>
class Model<Alloc, C> {
 public:
  explicit Model(C control, double r = 1, double a = 1)
      : u_{control}, rdiv2_{r / 2}, rdiva_{r / a} {
    assert((u_(0).size() == 2));
    assert((u_(0).size() == 2));
  }

  /**
   * @brief models equations system of robot (preferred version)
   *
   * @param state current state
   * @param time current time
   * @return optimization::StateDerivativesPoint left-hand side of equations
   * system (derivatives of state variables)
   */
  optimization::StateDerivativesPoint<double, Alloc> operator()(
      const optimization::Tensor<double, Alloc>& state, double time) {
    thread_local static auto res =
        optimization::StateDerivativesPoint<double, Alloc>(3);
    res = u_(time);
    const auto xyCommon{rdiv2_ * (res[0] + res[1])};
    return {xyCommon * std::cos(state[2]), xyCommon * std::sin(state[2]),
            (res[0] - res[1]) * rdiva_};
  }

  void operator()(
      const optimization::Tensor<double, Alloc>& state, double time,
      optimization::StateDerivativesPoint<double, Alloc>& preallocatedResult) {
    thread_local static auto res =
        optimization::StateDerivativesPoint<double, Alloc>(3);
    res = u_(time);
    const auto xyCommon{rdiv2_ * (res[0] + res[1])};
    preallocatedResult[0] = xyCommon * std::cos(state[2]);
    preallocatedResult[1] = xyCommon * std::sin(state[2]);
    preallocatedResult[1] = (res[0] - res[1]) * rdiva_;
  }

 private:
  C u_;

  double rdiv2_;
  double rdiva_;
};

template <class Alloc, ControlFunctionFullLvalue<Alloc> C>
class Model<Alloc, C> {
 public:
  explicit Model(C control, double r = 2, double a = 1)
      : u_{control}, rdiv2_{r / 2}, rdiva_{r / a} {
    assert((u_({0, 0, 0}, 0).size() == 2));
    assert((u_({0, 0, 0}, 0).size() == 2));
  }

  /**
   * @brief models equations system of robot (preferred version)
   *
   * @param state current state
   * @param time current time
   * @return optimization::StateDerivativesPoint<3> left-hand side of equations
   * system (derivatives of state variables)
   */
  optimization::StateDerivativesPoint<double, Alloc> operator()(
      const optimization::Tensor<double, Alloc>& state, double time) {
    thread_local static auto res =
        optimization::StateDerivativesPoint<double, Alloc>(3);
    res = u_(state, time);
    const auto xyCommon{rdiv2_ * (res[0] + res[1])};
    return {xyCommon * std::cos(state[2]), xyCommon * std::sin(state[2]),
            (res[0] - res[1]) * rdiva_};
  }

  void operator()(
      const optimization::Tensor<double, Alloc>& state, double time,
      optimization::StateDerivativesPoint<double, Alloc>& preallocatedResult) {
    thread_local static auto res =
        optimization::StateDerivativesPoint<double, Alloc>(3);
    res = u_(state, time);
    const auto xyCommon{rdiv2_ * (res[0] + res[1])};
    preallocatedResult[0] = xyCommon * std::cos(state[2]);
    preallocatedResult[1] = xyCommon * std::sin(state[2]);
    preallocatedResult[1] = (res[0] - res[1]) * rdiva_;
  }

 private:
  C u_;

  double rdiv2_;
  double rdiva_;
};

template <class Alloc, ControlFunctionTimeOnlyPreallocated<Alloc> C>
class Model<Alloc, C> {
 public:
  explicit Model(C control, double r = 1, double a = 1)
      : u_{control}, rdiv2_{r / 2}, rdiva_{r / a} {
    assert((u_(0).size() == 2));
    assert((u_(0).size() == 2));
  }

  /**
   * @brief models equations system of robot (preferred version)
   *
   * @param state current state
   * @param time current time
   * @return optimization::StateDerivativesPoint left-hand side of equations
   * system (derivatives of state variables)
   */
  optimization::StateDerivativesPoint<double, Alloc> operator()(
      const optimization::Tensor<double, Alloc>& state, double time) {
    thread_local static auto res =
        optimization::StateDerivativesPoint<double, Alloc>(3);
    u_(time, res);
    const auto xyCommon{rdiv2_ * (res[0] + res[1])};
    return {xyCommon * std::cos(state[2]), xyCommon * std::sin(state[2]),
            (res[0] - res[1]) * rdiva_};
  }

  void operator()(
      const optimization::Tensor<double, Alloc>& state, double time,
      optimization::StateDerivativesPoint<double, Alloc>& preallocatedResult) {
    thread_local static auto res =
        optimization::StateDerivativesPoint<double, Alloc>(3);
    u_(time, res);
    const auto xyCommon{rdiv2_ * (res[0] + res[1])};
    preallocatedResult[0] = xyCommon * std::cos(state[2]);
    preallocatedResult[1] = xyCommon * std::sin(state[2]);
    preallocatedResult[1] = (res[0] - res[1]) * rdiva_;
  }

 private:
  C u_;

  double rdiv2_;
  double rdiva_;
};

template <class Alloc, ControlFunctionFullLvaluePreallocated<Alloc> C>
class Model<Alloc, C> {
 public:
  explicit Model(C control, double r = 2, double a = 1)
      : u_{control}, rdiv2_{r / 2}, rdiva_{r / a} {
    assert((u_({0, 0, 0}, 0).size() == 2));
    assert((u_({0, 0, 0}, 0).size() == 2));
  }

  /**
   * @brief models equations system of robot (preferred version)
   *
   * @param state current state
   * @param time current time
   * @return optimization::StateDerivativesPoint<3> left-hand side of equations
   * system (derivatives of state variables)
   */
  optimization::StateDerivativesPoint<double, Alloc> operator()(
      const optimization::Tensor<double, Alloc>& state, double time) {
    thread_local static auto res =
        optimization::StateDerivativesPoint<double, Alloc>(3);
    u_(state, time, res);
    const auto xyCommon{rdiv2_ * (res[0] + res[1])};
    return {xyCommon * std::cos(state[2]), xyCommon * std::sin(state[2]),
            (res[0] - res[1]) * rdiva_};
  }

  void operator()(
      const optimization::Tensor<double, Alloc>& state, double time,
      optimization::StateDerivativesPoint<double, Alloc>& preallocatedResult) {
    thread_local static auto res =
        optimization::StateDerivativesPoint<double, Alloc>(3);
    u_(state, time, res);
    const auto xyCommon{rdiv2_ * (res[0] + res[1])};
    preallocatedResult[0] = xyCommon * std::cos(state[2]);
    preallocatedResult[1] = xyCommon * std::sin(state[2]);
    preallocatedResult[1] = (res[0] - res[1]) * rdiva_;
  }

 private:
  C u_;

  double rdiv2_;
  double rdiva_;
};
}  // namespace two_wheeled_robot
