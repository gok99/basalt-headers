#pragma once
#include <cmath>
#include <Eigen/Dense>

namespace basalt::autodiff {

// See http://ceres-solver.org/automatic_derivatives.html#implementing-jets
template<typename S, int N> struct Jet {
  S a;
  Eigen::Matrix<S, 1, N> v;

  Jet() {}
  Jet(S a, const Eigen::Matrix<S, 1, N> &v) : a(a), v(v) {}

  static Jet<S, N> castFrom(S s) {
    Jet<S, N> jet;
    jet.a = s;
    jet.v.setZero();
    return jet;
  }

  static Jet<S, N> castFrom(const Jet<S, N> &jet) {
    return jet;
  }

  Jet<S, N> operator+(const Jet<S, N>& g) const {
    return Jet<S, N>(a + g.a, v + g.v);
  }

  Jet<S, N> operator-(const Jet<S, N>& g) const {
    return Jet<S, N>(a - g.a, v - g.v);
  }

  Jet<S, N> operator*(const Jet<S, N>& g) const {
    return Jet<S, N>(a * g.a, a * g.v + v * g.a);
  }

  Jet<S, N> operator/(const Jet<S, N>& g) const {
    return Jet<S, N>(a / g.a, v / g.a - a * g.v / (g.a * g.a));
  }

  Jet<S, N> operator+(S s) const {
    return Jet<S, N>(a + s, v);
  }

  Jet<S, N> operator-(S s) const {
    return Jet<S, N>(a - s, v);
  }

  Jet<S, N> operator*(S s) const {
    return Jet<S, N>(a * s, v * s);
  }

  Jet<S, N> operator/(S s) const {
    return Jet<S, N>(a / s, v / s);
  }

  Jet<S, N> operator-() const {
    return Jet<S, N>(-a, -v);
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

using std::sqrt;
using std::acos;

template<typename S, int N> Jet<S, N> sqrt(const Jet<S, N> &f) {
  const S x = std::sqrt(f.a);
  return Jet<S, N>(x, S(0.5) * f.v / x);
}

template<typename S, int N> Jet<S, N> acos(const Jet<S, N> &f) {
  return Jet<S, N>(std::acos(f.a), f.v * (S(-1) / std::sqrt(S(1) - f.a*f.a)));
}

// TODO: add more mathematical functions as needed

template<typename S, typename J> inline S asScalar(const J &f) { return f.a; }
template<typename T, typename S> inline T maybeJet(const S &t) { return T::castFrom(t); }

#define X(x) \
  template<> inline x maybeJet(const x &s) { return s; } \
  template<> inline x asScalar(const x &s) { return s; }
X(float)
X(double)
X(long double)
#undef X

}

#define JET basalt::autodiff::Jet

// scalar first
template<typename S, int N> JET<S, N> operator+(S s, const JET<S, N>& f) {
  return JET<S, N>(f.a + s, f.v);
}

template<typename S, int N> JET<S, N> operator-(S s, const JET<S, N>& f) {
  return JET<S, N>(s - f.a, -f.v);
}

template<typename S, int N> JET<S, N> operator*(S s, const JET<S, N>& f) {
  return JET<S, N>(f.a * s, f.v * s);
}

template<typename S, int N> JET<S, N> operator/(S s, const JET<S, N>& g) {
  return JET<S, N>(s / g.a, -s * g.v / (g.a * g.a));
}

#undef JET