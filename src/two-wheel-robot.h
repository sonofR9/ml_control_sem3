#pragma once

#include "runge_kutte.h"

#include <cmath>
#include <concepts>
// import runge_kutte;

namespace two_wheeled_robot {

template <typename T>
concept ControlFunctionFullLvalue =
    requires(T fun, const runge_kutte::StatePoint<3>& state, double time) {
      { fun(state, time) } -> std::same_as<double>;
    };

template <typename T>
concept ControlFunctionTimeOnly = requires(T fun, double time) {
  { fun(time) } -> std::same_as<double>;
};

template <typename T>
concept ControlFunctionFullRvalue =
    requires(T fun, runge_kutte::StatePoint<3>&& state, double time) {
      { fun(std::move(state), time) } -> std::same_as<double>;
    };

template <typename T>
concept ControlFunction =
    ControlFunctionFullLvalue<T> || ControlFunctionTimeOnly<T> ||
    ControlFunctionFullRvalue<T>;

template <ControlFunction C1, ControlFunction C2>
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
  Model(C1 leftWheelControl, C2 rightWheelControl, double r = 2, double a = 1);

  /**
   * @brief models equations system of robot
   *
   * @param state current state
   * @param time current time
   * @return runge_kutte::StateDerivativesPoint<3> left-hand side of equations
   * system (derivatives of state variables)
   */
  runge_kutte::StateDerivativesPoint<3>
  operator()(runge_kutte::StatePoint<3> state, double time);
};

template <ControlFunctionTimeOnly C1, ControlFunctionTimeOnly C2>
class Model<C1, C2> {
 public:
  Model(C1 leftWheelControl, C2 rightWheelControl, double r = 2, double a = 1)
      : u1_{leftWheelControl}, u2_{rightWheelControl}, r_{r}, a_{a} {
  }

  /**
   * @brief models equations system of robot (preferred version)
   *
   * @param state current state
   * @param time current time
   * @return runge_kutte::StateDerivativesPoint<3> left-hand side of equations
   * system (derivatives of state variables)
   */
  runge_kutte::StateDerivativesPoint<3>
  operator()(const runge_kutte::StatePoint<3>& state, double time) {
    return {(u1_(time) + u2_(time)) * std::cos(state[2]),
            (u1_(time) + u2_(time)) * std::sin(state[2]),
            u1_(time) - u2_(time)};
  }

  /**
   * @brief Overload of models equations system (So it could accept rvalues too)
   */
  runge_kutte::StatePoint<3> operator()(runge_kutte::StatePoint<3>&& state,
                                        double time) {
    return {(u1_(time) + u2_(time)) * std::cos(state[2]),
            (u1_(time) + u2_(time)) * std::sin(state[2]),
            u1_(time) - u2_(time)};
  }

 private:
  C1 u1_;
  C2 u2_;

  double r_;
  double a_;
};

template <ControlFunctionFullLvalue C1, ControlFunctionFullLvalue C2>
class Model<C1, C2> {
 public:
  Model(C1 leftWheelControl, C2 rightWheelControl, double r = 2, double a = 1)
      : u1_{leftWheelControl}, u2_{rightWheelControl}, r_{r}, a_{a} {
  }

  /**
   * @brief models equations system of robot (preferred version)
   *
   * @param state current state
   * @param time current time
   * @return runge_kutte::StateDerivativesPoint<3> left-hand side of equations
   * system (derivatives of state variables)
   */
  runge_kutte::StateDerivativesPoint<3>
  operator()(const runge_kutte::StatePoint<3>& state, double time) {
    return {r_ / 2 * (u1_(state, time) + u2_(state, time)) * std::cos(state[2]),
            r_ / 2 * (u1_(state, time) + u2_(state, time)) * std::sin(state[2]),
            (u1_(state, time) - u2_(state, time)) * r_ / a_};
  }

  /**
   * @brief Overload of models equations system (So it could accept rvalues too)
   * (not preferred)
   */
  runge_kutte::StatePoint<3> operator()(runge_kutte::StatePoint<3> state,
                                        double time) {
    return {r_ / 2 * (u1_(state, time) + u2_(state, time)) * std::cos(state[2]),
            r_ / 2 * (u1_(state, time) + u2_(state, time)) * std::sin(state[2]),
            (u1_(state, time) - u2_(state, time)) * r_ / a_};
  }

 private:
  C1 u1_;
  C2 u2_;

  double r_;
  double a_;
};

template <ControlFunctionFullRvalue C1, ControlFunctionFullRvalue C2>
class Model<C1, C2> {
 public:
  Model(C1 leftWheelControl, C2 rightWheelControl, double r = 2, double a = 1)
      : u1_{leftWheelControl}, u2_{rightWheelControl}, r_{r}, a_{a} {
  }

  /**
   * @brief models equations system of robot (preferred version)
   *
   * @param state current state
   * @param time current time
   * @return runge_kutte::StateDerivativesPoint<3> left-hand side of equations
   * system (derivatives of state variables)
   */
  runge_kutte::StatePoint<3> operator()(runge_kutte::StatePoint<3>&& state,
                                        double time) {
    return {r_ / 2 * (u1_(state, time) + u2_(state, time)) * std::cos(state[2]),
            r_ / 2 * (u1_(state, time) + u2_(state, time)) * std::sin(state[2]),
            (u1_(state, time) - u2_(state, time)) * r_ / a_};
  }

  /**
   * @brief Overload of models equations system (So it could accept lvalues too)
   * (not preferred)
   */
  runge_kutte::StatePoint<3> operator()(runge_kutte::StatePoint<3> state,
                                        double time) {
    return {r_ / 2 * (u1_(state, time) + u2_(state, time)) * std::cos(state[2]),
            r_ / 2 * (u1_(state, time) + u2_(state, time)) * std::sin(state[2]),
            (u1_(state, time) - u2_(state, time)) * r_ / a_};
  }

 private:
  C1 u1_;
  C2 u2_;

  double r_;
  double a_;
};
} // namespace two_wheeled_robot
