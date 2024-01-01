#pragma once

#include "runge-kutte.h"

#include <cmath>
#include <concepts>
// import optimization;

namespace two_wheeled_robot {

template <typename T>
concept ControlFunctionFullLvalue =
    requires(T fun, const optimization::StaticTensor<3>& state, double time) {
      { fun(state, time) } -> std::same_as<optimization::StaticTensor<2>>;
    };

template <typename T>
concept ControlFunctionTimeOnly = requires(T fun, double time) {
  { fun(time) } -> std::same_as<optimization::StaticTensor<2>>;
};

template <typename T>
concept ControlFunctionFullRvalue =
    requires(T fun, optimization::StaticTensor<3>&& state, double time) {
      {
        fun(std::move(state), time)
      } -> std::same_as<optimization::StaticTensor<2>>;
    };

template <typename T>
concept ControlFunction =
    ControlFunctionFullLvalue<T> || ControlFunctionTimeOnly<T> ||
    ControlFunctionFullRvalue<T>;

template <ControlFunction C>
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
  Model(C control, double r = 1, double a = 1);

  /**
   * @brief models equations system of robot
   *
   * @param state current state
   * @param time current time
   * @return optimization::StateDerivativesPoint<3> left-hand side of equations
   * system (derivatives of state variables)
   */
  optimization::StateDerivativesPoint<3> operator()(
      optimization::StaticTensor<3> state, double time);
};

template <ControlFunctionTimeOnly C>
class Model<C> {
 public:
  Model(C control, double r = 1, double a = 1)
      : u_{control}, rdiv2_{r / 2}, rdiva_{r / a} {
  }

  /**
   * @brief models equations system of robot (preferred version)
   *
   * @param state current state
   * @param time current time
   * @return optimization::StateDerivativesPoint<3> left-hand side of equations
   * system (derivatives of state variables)
   */
  optimization::StateDerivativesPoint<3> operator()(
      const optimization::StaticTensor<3>& state, double time) {
    const auto res{u_(time)};
    const auto xyCommon{rdiv2_ * (res[0] + res[1])};
    return {xyCommon * std::cos(state[2]), xyCommon * std::sin(state[2]),
            (res[0] - res[1]) * rdiva_};
  }

 private:
  C u_;

  double rdiv2_;
  double rdiva_;
};

template <ControlFunctionFullLvalue C>
class Model<C> {
 public:
  Model(C control, double r = 2, double a = 1)
      : u_{control}, rdiv2_{r / 2}, rdiva_{r / a} {
  }

  /**
   * @brief models equations system of robot (preferred version)
   *
   * @param state current state
   * @param time current time
   * @return optimization::StateDerivativesPoint<3> left-hand side of equations
   * system (derivatives of state variables)
   */
  optimization::StateDerivativesPoint<3> operator()(
      const optimization::StaticTensor<3>& state, double time) {
    auto res{u_(state, time)};
    const auto xyCommon{rdiv2_ * (res[0] + res[1])};
    return {time * std::cos(state[2]), time * std::sin(state[2]),
            (res[0] - res[1]) * rdiva_};
  }

 private:
  C u_;

  double rdiv2_;
  double rdiva_;
};

// template <ControlFunctionFullRvalue C>
// class Model<C> {
//  public:
//   Model(C control, double r = 2, double a = 1) : u_{control}, r_{r}, a_{a} {
//   }

//   /**
//    * @brief models equations system of robot (preferred version)
//    *
//    * @param state current state
//    * @param time current time
//    * @return optimization::StateDerivativesPoint<3> left-hand side of
//    equations
//    * system (derivatives of state variables)
//    */
//   optimization::StaticTensor<3> operator()(optimization::StaticTensor<3>&&
//   state,
//                                      double time) {
//     auto res{u_(state, time)};
//     return {r_ / 2 * (res[0] + res[1]) * std::cos(state[2]),
//             r_ / 2 * (res[0] + res[1]) * std::sin(state[2]),
//             (res[0] - res[1]) * r_ / a_};
//   }
//
//  private:
//   C u_;

//   double r_;
//   double a_;
// };
}  // namespace two_wheeled_robot
