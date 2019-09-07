/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt-headers.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

@file
@brief Uniform cumulative B-spline for SO(3)
*/

#pragma once

#include <basalt/spline/spline_common.h>
#include <basalt/utils/assert.h>
#include <basalt/utils/sophus_utils.hpp>

#include <Eigen/Dense>
#include <sophus/so3.hpp>

#include <array>

namespace basalt {

/// @brief Uniform cummulative B-spline for SO(3) of order N
///
/// For example, in the particular case scalar values and order N=5, for a time
/// \f$t \in [t_i, t_{i+1})\f$ the value of \f$p(t)\f$ depends only on 5 control
/// points at \f$[t_i, t_{i+1}, t_{i+2}, t_{i+3}, t_{i+4}]\f$. To
/// simplify calculations we transform time to uniform representation \f$s(t) =
/// (t - t_0)/\Delta t \f$, such that control points transform into \f$ s_i \in
/// [0,..,N] \f$. We define function \f$ u(t) = s(t)-s_i \f$ to be a time since
/// the start of the segment. Following the cummulative matrix representation of
/// De Boor - Cox formula, the value of the function can be evaluated as
/// follows: \f{align}{
///    R(u(t)) &= R_i
///    \prod_{j=1}^{4}{\exp(k_{j}\log{(R_{i+j-1}^{-1}R_{i+j})})},
///    \\ \begin{pmatrix} k_0 \\ k_1 \\ k_2 \\ k_3 \\ k_4 \end{pmatrix}^T &=
///    M_{c5} \begin{pmatrix} 1 \\ u \\ u^2 \\ u^3 \\ u^4
///    \end{pmatrix},
/// \f}
/// where \f$ R_{i} \in SO(3) \f$ are knots and \f$ M_{c5} \f$ is a cummulative
/// blending matrix computed using \ref computeBlendingMatrix \f{align}{
///    M_{c5} = \frac{1}{4!}
///    \begin{pmatrix} 24 & 0 & 0 & 0 & 0 \\ 23 & 4 & -6 & 4 & -1 \\ 12 & 16 & 0
///    & -8 & 3 \\ 1 & 4 & 6 & 4 & -3 \\ 0 & 0 & 0 & 0 & 1 \end{pmatrix}.
/// \f}
template <int _N, typename _Scalar = double>
class So3Spline {
 public:
  static constexpr int N = _N;        ///< Order of the spline.
  static constexpr int DEG = _N - 1;  ///< Degree of the spline.

  static constexpr _Scalar ns_to_s = 1e-9;  ///< Nanosecond to second conversion
  static constexpr _Scalar s_to_ns = 1e9;   ///< Second to nanosecond conversion

  using MatN = Eigen::Matrix<_Scalar, _N, _N>;
  using VecN = Eigen::Matrix<_Scalar, _N, 1>;

  using VecD = Eigen::Matrix<_Scalar, 3, 1>;
  using MatD = Eigen::Matrix<_Scalar, 3, 3>;

  using SO3 = Sophus::SO3<_Scalar>;

  /// @brief Struct to store the Jacobian of the spline
  ///
  /// Since B-spline of order N has local support (only N knots infuence the
  /// value) the Jacobian is zero for all knots except maximum N for value and
  /// all derivatives.
  struct JacobianStruct {
    size_t start_idx;
    std::array<MatD, _N> d_val_d_knot;
  };

  /// @brief Constructor with knot interval and start time
  ///
  /// @param[in] time_interval_ns knot time interval in nanoseconds
  /// @param[in] start_time_ns start time of the spline in nanoseconds
  So3Spline(int64_t time_interval_ns, int64_t start_time_ns = 0)
      : dt_ns(time_interval_ns), start_t_ns(start_time_ns) {
    pow_inv_dt[0] = 1.0;
    pow_inv_dt[1] = s_to_ns / dt_ns;
    pow_inv_dt[2] = pow_inv_dt[1] * pow_inv_dt[1];
  }

  /// @brief Maximum time represented by spline
  ///
  /// @return maximum time represented by spline in nanoseconds
  int64_t maxTimeNs() const {
    return start_t_ns + (knots.size() - N + 1) * dt_ns - 1;
  }

  /// @brief Minimum time represented by spline
  ///
  /// @return minimum time represented by spline in nanoseconds
  int64_t minTimeNs() const { return start_t_ns; }

  /// @brief Gererate random trajectory
  ///
  /// @param[in] n number of knots to generate
  /// @param[in] static_init if true the first N knots will be the same
  /// resulting in static initial condition
  void genRandomTrajectory(int n, bool static_init = false) {
    if (static_init) {
      VecD rnd = VecD::Random() * M_PI;

      for (int i = 0; i < N; i++) knots.push_back(SO3::exp(rnd));

      for (int i = 0; i < n - N; i++)
        knots.push_back(SO3::exp(VecD::Random() * M_PI));

    } else {
      for (int i = 0; i < n; i++)
        knots.push_back(SO3::exp(VecD::Random() * M_PI));
    }
  }

  /// @brief Set start time for spline
  ///
  /// @param[in] start_time_ns start time of the spline in nanoseconds
  inline void setStartTimeNs(int64_t s) { start_t_ns = s; }

  /// @brief Add knot to the end of the spline
  ///
  /// @param[in] knot knot to add
  inline void knots_push_back(const SO3& knot) { knots.push_back(knot); }

  /// @brief Remove knot from the back of the spline
  inline void knots_pop_back() { knots.pop_back(); }

  /// @brief Return the first knot of the spline
  ///
  /// @return first knot of the spline
  inline const SO3& knots_front() const { return knots.front(); }

  /// @brief Remove first knot of the spline and increase the start time
  inline void knots_pop_front() {
    start_t_ns += dt_ns;
    knots.pop_front();
  }

  /// @brief Resize containter with knots
  ///
  /// @param[in] n number of knots
  inline void resize(size_t n) { knots.resize(n); }

  /// @brief Return reference to the knot with index i
  ///
  /// @param i index of the knot
  /// @return reference to the knot
  inline SO3& getKnot(int i) { return knots[i]; }

  /// @brief Return const reference to the knot with index i
  ///
  /// @param i index of the knot
  /// @return const reference to the knot
  inline const SO3& getKnot(int i) const { return knots[i]; }

  /// @brief Return const reference to deque with knots
  ///
  /// @return const reference to deque with knots
  const Eigen::deque<SO3>& getKnots() const { return knots; }

  /// @brief Return time interval in nanoseconds
  ///
  /// @return time interval in nanoseconds
  int64_t getTimeIntervalNs() const { return dt_ns; }

  /// @brief Evaluate SO(3) B-spline
  ///
  /// @param[in] time_ns time for evaluating the value of the spline in
  /// nanoseconds
  /// @param[out] J if not nullptr, return the Jacobian of the value with
  /// respect to knots
  /// @return SO(3) value of the spline
  SO3 evaluate(int64_t time_ns, JacobianStruct* J = nullptr) const {
    int64_t st_ns = (time_ns - start_t_ns);

    BASALT_ASSERT_STREAM(st_ns >= 0, "st_ns " << st_ns << " time_ns " << time_ns
                                              << " start_t_ns " << start_t_ns);

    int64_t s = st_ns / dt_ns;
    double u = double(st_ns % dt_ns) / double(dt_ns);

    BASALT_ASSERT_STREAM(s >= 0, "s " << s);
    BASALT_ASSERT_STREAM(size_t(s + N) <= knots.size(), "s " << s << " N " << N
                                                             << " knots.size() "
                                                             << knots.size());

    VecN p;
    baseCoeffsWithTime<0>(p, u);

    VecN coeff = blending_matrix_ * p;

    SO3 res = knots[s];

    MatD J_helper;

    if (J) {
      J->start_idx = s;
      J_helper.setIdentity();
    }

    for (int i = 0; i < DEG; i++) {
      const SO3& p0 = knots[s + i];
      const SO3& p1 = knots[s + i + 1];

      SO3 r01 = p0.inverse() * p1;
      VecD delta = r01.log();
      VecD kdelta = delta * coeff[i + 1];

      if (J) {
        MatD Jl_inv_delta, Jl_k_delta;

        Sophus::leftJacobianInvSO3(delta, Jl_inv_delta);
        Sophus::leftJacobianSO3(kdelta, Jl_k_delta);

        J->d_val_d_knot[i] = J_helper;
        J_helper = coeff[i + 1] * res.matrix() * Jl_k_delta * Jl_inv_delta *
                   p0.inverse().matrix();
        J->d_val_d_knot[i] -= J_helper;
      }
      res *= SO3::exp(kdelta);
    }

    if (J) J->d_val_d_knot[DEG] = J_helper;

    return res;
  }

  /// @brief Evaluate rotational velocity (first time derivative) of SO(3)
  /// B-spline in the body frame

  /// First, let's note that for scalars \f$ k, \Delta k \f$ the following
  /// holds: \f$ \exp((k+\Delta k)\phi) = \exp(k\phi)\exp(\Delta k\phi), \phi
  /// \in \mathbb{R}^3\f$. This is due to the fact that rotations around the
  /// same axis are commutative.
  ///
  /// Let's take SO(3) B-spline with N=3 as an example. The evolution in time of
  /// rotation from the body frame to the world frame is described with \f[
  ///  R_{wb}(t) = R(t) = R_i \exp(k_1(t) \log(R_{i}^{-1}R_{i+1})) \exp(k_2(t)
  ///  \log(R_{i+1}^{-1}R_{i+2})), \f] where \f$ k_1, k_2 \f$ are spline
  ///  coefficients (see detailed description of \ref So3Spline). Since
  ///  expressions under logmap do not depend on time we can rename them to
  ///  constants.
  /// \f[ R(t) = R_i \exp(k_1(t) ~ d_1) \exp(k_2(t) ~ d_2). \f]
  ///
  /// With linear approximation of the spline coefficient evolution over time
  /// \f$ k_1(t_0 + \Delta t) = k_1(t_0) + k_1'(t_0)\Delta t \f$ we can write
  /// \f{align}
  ///  R(t_0 + \Delta t) &= R_i \exp( (k_1(t_0) + k_1'(t_0) \Delta t) ~ d_1)
  ///  \exp((k_2(t_0) + k_2'(t_0) \Delta t) ~ d_2)
  ///  \\ &= R_i \exp(k_1(t_0) ~ d_1) \exp(k_1'(t_0)~ d_1 \Delta t )
  ///  \exp(k_2(t_0) ~ d_2) \exp(k_2'(t_0) ~ d_2 \Delta t )
  ///  \\ &= R_i \exp(k_1(t_0) ~ d_1)
  ///  \exp(k_2(t_0) ~ d_2) \exp(R_{a}^T k_1'(t_0)~ d_1 \Delta t )
  ///  \exp(k_2'(t_0) ~ d_2 \Delta t )
  ///  \\ &= R_i \exp(k_1(t_0) ~ d_1)
  ///  \exp(k_2(t_0) ~ d_2) \exp((R_{a}^T k_1'(t_0)~ d_1 +
  ///  k_2'(t_0) ~ d_2) \Delta t )
  ///  \\ &= R(t_0) \exp((R_{a}^T k_1'(t_0)~ d_1 +
  ///  k_2'(t_0) ~ d_2) \Delta t )
  ///  \\ &= R(t_0) \exp( \omega \Delta t ),
  /// \f} where \f$ \Delta t \f$ is small, \f$ R_{a} \in SO(3) = \exp(k_2(t_0) ~
  /// d_2) \f$ and \f$ \omega \f$ is the rotational velocity in the body frame.
  /// More explicitly we have the formula for rotational velocity in the body
  /// frame \f[ \omega = R_{a}^T k_1'(t_0)~ d_1 +  k_2'(t_0) ~ d_2. \f]
  /// Derivatives of spline coefficients can be computed with \ref
  /// baseCoeffsWithTime similar to \ref RdSpline (detailed description). With
  /// the recursive formula computations generalize to different orders of
  /// spline N.
  ///
  /// @param[in] time_ns time for evaluating velocity of the spline in
  /// nanoseconds
  /// @return rotational velocity (3x1 vector)
  VecD velocityBody(int64_t time_ns) const {
    int64_t st_ns = (time_ns - start_t_ns);

    BASALT_ASSERT_STREAM(st_ns >= 0, "st_ns " << st_ns << " time_ns " << time_ns
                                              << " start_t_ns " << start_t_ns);

    int64_t s = st_ns / dt_ns;
    double u = double(st_ns % dt_ns) / double(dt_ns);

    BASALT_ASSERT_STREAM(s >= 0, "s " << s);
    BASALT_ASSERT_STREAM(size_t(s + N) <= knots.size(), "s " << s << " N " << N
                                                             << " knots.size() "
                                                             << knots.size());

    VecN p;
    baseCoeffsWithTime<0>(p, u);
    VecN coeff = blending_matrix_ * p;

    baseCoeffsWithTime<1>(p, u);
    VecN dcoeff = pow_inv_dt[1] * blending_matrix_ * p;

    SO3 r_accum;

    VecD rot_vel;
    rot_vel.setZero();

    for (int i = DEG - 1; i >= 0; i--) {
      const SO3& p0 = knots[s + i];
      const SO3& p1 = knots[s + i + 1];

      SO3 r01 = p0.inverse() * p1;
      VecD delta = r01.log();

      rot_vel += r_accum * delta * dcoeff[i + 1];
      r_accum *= SO3::exp(-delta * coeff[i + 1]);
    }

    return rot_vel;
  }

  /// @brief Evaluate rotational velocity (first time derivative) of SO(3)
  /// B-spline in the body frame
  ///
  /// @param[in] time_ns time for evaluating velocity of the spline in
  /// nanoseconds
  /// @param[out] J if not nullptr, return the Jacobian of the rotational
  /// velocity in body frame with respect to knots
  /// @return rotational velocity (3x1 vector)
  VecD velocityBody(int64_t time_ns, JacobianStruct* J) const {
    int64_t st_ns = (time_ns - start_t_ns);

    BASALT_ASSERT_STREAM(st_ns >= 0, "st_ns " << st_ns << " time_ns " << time_ns
                                              << " start_t_ns " << start_t_ns);

    int64_t s = st_ns / dt_ns;
    double u = double(st_ns % dt_ns) / double(dt_ns);

    BASALT_ASSERT_STREAM(s >= 0, "s " << s);
    BASALT_ASSERT_STREAM(size_t(s + N) <= knots.size(), "s " << s << " N " << N
                                                             << " knots.size() "
                                                             << knots.size());

    VecN p;
    baseCoeffsWithTime<0>(p, u);
    VecN coeff = blending_matrix_ * p;

    baseCoeffsWithTime<1>(p, u);
    VecN dcoeff = pow_inv_dt[1] * blending_matrix_ * p;

    VecD rot_vel;
    rot_vel.setZero();

    VecD delta_vec[DEG];

    MatD R_tmp[DEG];
    SO3 accum;
    SO3 exp_k_delta[DEG];

    MatD Jr_delta_inv[DEG], Jr_kdelta[DEG];

    for (int i = DEG - 1; i >= 0; i--) {
      const SO3& p0 = knots[s + i];
      const SO3& p1 = knots[s + i + 1];

      SO3 r01 = p0.inverse() * p1;
      delta_vec[i] = r01.log();

      Sophus::rightJacobianInvSO3(delta_vec[i], Jr_delta_inv[i]);
      Jr_delta_inv[i] *= p1.inverse().matrix();

      VecD k_delta = coeff[i + 1] * delta_vec[i];
      Sophus::rightJacobianSO3(-k_delta, Jr_kdelta[i]);

      R_tmp[i] = accum.matrix();
      rot_vel += R_tmp[i] * delta_vec[i] * dcoeff[i + 1];
      exp_k_delta[i] = Sophus::SO3d::exp(-k_delta);
      accum *= exp_k_delta[i];
    }

    MatD d_vel_d_delta[DEG];

    d_vel_d_delta[0] = dcoeff[1] * R_tmp[0] * Jr_delta_inv[0];
    VecD v = delta_vec[0] * dcoeff[1];
    for (int i = 1; i < DEG; i++) {
      d_vel_d_delta[i] =
          R_tmp[i - 1] * SO3::hat(v) * Jr_kdelta[i] * coeff[i + 1] +
          R_tmp[i] * dcoeff[i + 1];
      d_vel_d_delta[i] *= Jr_delta_inv[i];

      v = exp_k_delta[i] * v + delta_vec[i] * dcoeff[i + 1];
    }

    if (J) {
      J->start_idx = s;
      for (int i = 0; i < N; i++) J->d_val_d_knot[i].setZero();
      for (int i = 0; i < DEG; i++) {
        J->d_val_d_knot[i] -= d_vel_d_delta[i];
        J->d_val_d_knot[i + 1] += d_vel_d_delta[i];
      }
    }

    return rot_vel;
  }

  /// @brief Evaluate rotational acceleration (second time derivative) of SO(3)
  /// B-spline in the body frame
  ///
  /// @param[in] time_ns time for evaluating velocity of the spline in
  /// nanoseconds
  /// @param[out] vel_body if not nullptr, return the rotational velocity in the
  /// body frame (3x1 vector) (side computation)
  /// @return rotational acceleration (3x1 vector)
  VecD accelerationBody(int64_t time_ns, VecD* vel_body = nullptr) const {
    int64_t st_ns = (time_ns - start_t_ns);

    BASALT_ASSERT_STREAM(st_ns >= 0, "st_ns " << st_ns << " time_ns " << time_ns
                                              << " start_t_ns " << start_t_ns);

    int64_t s = st_ns / dt_ns;
    double u = double(st_ns % dt_ns) / double(dt_ns);

    BASALT_ASSERT_STREAM(s >= 0, "s " << s);
    BASALT_ASSERT_STREAM(size_t(s + N) <= knots.size(), "s " << s << " N " << N
                                                             << " knots.size() "
                                                             << knots.size());

    VecN p;
    baseCoeffsWithTime<0>(p, u);
    VecN coeff = blending_matrix_ * p;

    baseCoeffsWithTime<1>(p, u);
    VecN dcoeff = pow_inv_dt[1] * blending_matrix_ * p;

    baseCoeffsWithTime<2>(p, u);
    VecN ddcoeff = pow_inv_dt[2] * blending_matrix_ * p;

    SO3 r_accum;

    VecD rot_vel;
    rot_vel.setZero();

    VecD rot_accel;
    rot_accel.setZero();

    for (int i = DEG - 1; i >= 0; i--) {
      const SO3& p0 = knots[s + i];
      const SO3& p1 = knots[s + i + 1];

      SO3 r01 = p0.inverse() * p1;
      VecD delta = r01.log();

      VecD r_accum_delta = r_accum * delta;

      VecD rot_vel_current = r_accum_delta * dcoeff[i + 1];
      rot_accel +=
          r_accum_delta * ddcoeff[i + 1] + rot_vel_current.cross(rot_vel);
      rot_vel += rot_vel_current;
      r_accum *= SO3::exp(-delta * coeff[i + 1]);
    }

    if (vel_body) *vel_body = rot_vel;
    return rot_accel;
  }

  /// @brief Evaluate rotational acceleration (second time derivative) of SO(3)
  /// B-spline in the body frame
  ///
  /// @param[in] time_ns time for evaluating velocity of the spline in
  /// nanoseconds
  /// @param[out] J_accel if not nullptr, return the Jacobian of the rotational
  /// acceleration in body frame with respect to knots
  /// @param[out] vel_body if not nullptr, return the rotational velocity in the
  /// body frame (3x1 vector) (side computation)
  /// @param[out] J_vel if not nullptr, return the Jacobian of the rotational
  /// velocity in the body frame (side computation)
  /// @return rotational acceleration (3x1 vector)
  VecD accelerationBody(int64_t time_ns, JacobianStruct* J_accel,
                        VecD* vel_body = nullptr,
                        JacobianStruct* J_vel = nullptr) const {
    BASALT_ASSERT(J_accel);

    int64_t st_ns = (time_ns - start_t_ns);

    BASALT_ASSERT_STREAM(st_ns >= 0, "st_ns " << st_ns << " time_ns " << time_ns
                                              << " start_t_ns " << start_t_ns);

    int64_t s = st_ns / dt_ns;
    double u = double(st_ns % dt_ns) / double(dt_ns);

    BASALT_ASSERT_STREAM(s >= 0, "s " << s);
    BASALT_ASSERT_STREAM(size_t(s + N) <= knots.size(), "s " << s << " N " << N
                                                             << " knots.size() "
                                                             << knots.size());

    VecN p;
    baseCoeffsWithTime<0>(p, u);
    VecN coeff = blending_matrix_ * p;

    baseCoeffsWithTime<1>(p, u);
    VecN dcoeff = pow_inv_dt[1] * blending_matrix_ * p;

    baseCoeffsWithTime<2>(p, u);
    VecN ddcoeff = pow_inv_dt[2] * blending_matrix_ * p;

    VecD rot_accel;
    rot_accel.setZero();

    VecD rot_vel;
    rot_vel.setZero();

    VecD delta_vec[DEG];

    MatD R_tmp[DEG];
    SO3 accum;
    SO3 exp_k_delta[DEG];
    VecD rot_vel_current[DEG];

    MatD Jr_delta_inv[DEG], Jr_kdelta[DEG];

    for (int i = DEG - 1; i >= 0; i--) {
      const SO3& p0 = knots[s + i];
      const SO3& p1 = knots[s + i + 1];

      SO3 r01 = p0.inverse() * p1;
      delta_vec[i] = r01.log();

      Sophus::rightJacobianInvSO3(delta_vec[i], Jr_delta_inv[i]);
      Jr_delta_inv[i] *= p1.inverse().matrix();

      VecD k_delta = coeff[i + 1] * delta_vec[i];
      Sophus::rightJacobianSO3(-k_delta, Jr_kdelta[i]);

      R_tmp[i] = accum.matrix();
      VecD accum_delta = R_tmp[i] * delta_vec[i];

      rot_vel_current[i] = accum_delta * dcoeff[i + 1];
      rot_accel +=
          accum_delta * ddcoeff[i + 1] + rot_vel_current[i].cross(rot_vel);
      rot_vel += rot_vel_current[i];

      exp_k_delta[i] = Sophus::SO3d::exp(-k_delta);
      accum *= exp_k_delta[i];
    }

    MatD d_acccel_d_delta[DEG];
    MatD J_v_mat[DEG][DEG];
    MatD vc_mat[DEG];
    VecD vc = rot_vel_current[0] - rot_vel;
    vc_mat[0] = SO3::hat(vc);

    d_acccel_d_delta[0] = R_tmp[0] * ddcoeff[1];
    VecD dv = delta_vec[0] * ddcoeff[1];
    for (int i = 0; i < DEG; i++) {
      if (i > 0) {
        d_acccel_d_delta[i] =
            R_tmp[i - 1] * SO3::hat(dv) * Jr_kdelta[i] * coeff[i + 1] +
            R_tmp[i] * ddcoeff[i + 1];

        dv = exp_k_delta[i] * dv + delta_vec[i] * ddcoeff[i + 1];

        vc += rot_vel_current[i - 1] + rot_vel_current[i];
        vc_mat[i] = SO3::hat(vc);
      }

      J_v_mat[i][i] = R_tmp[i] * dcoeff[i + 1];
      VecD vv = delta_vec[i] * dcoeff[i + 1];
      for (int j = i + 1; j < DEG; j++) {
        J_v_mat[i][j] =
            R_tmp[j - 1] * Sophus::SO3d::hat(vv) * Jr_kdelta[j] * coeff[j + 1];
        vv = exp_k_delta[j] * vv;
      }
    }

    MatD d_vel_d_delta[DEG];

    for (int j = 0; j < DEG; j++) {
      d_vel_d_delta[j].setZero();
      for (int i = 0; i <= j; i++) {
        d_acccel_d_delta[j] += vc_mat[i] * J_v_mat[i][j];
        d_vel_d_delta[j] += J_v_mat[i][j];
      }
      d_acccel_d_delta[j] *= Jr_delta_inv[j];
      d_vel_d_delta[j] *= Jr_delta_inv[j];
    }

    if (J_vel) {
      J_vel->start_idx = s;
      for (int i = 0; i < N; i++) J_vel->d_val_d_knot[i].setZero();
      for (int i = 0; i < DEG; i++) {
        J_vel->d_val_d_knot[i] -= d_vel_d_delta[i];
        J_vel->d_val_d_knot[i + 1] += d_vel_d_delta[i];
      }
    }

    if (J_accel) {
      J_accel->start_idx = s;
      for (int i = 0; i < N; i++) J_accel->d_val_d_knot[i].setZero();
      for (int i = 0; i < DEG; i++) {
        J_accel->d_val_d_knot[i] -= d_acccel_d_delta[i];
        J_accel->d_val_d_knot[i + 1] += d_acccel_d_delta[i];
      }
    }

    if (vel_body) *vel_body = rot_vel;
    return rot_accel;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 protected:
  /// @brief Vector of derivatives of time polynomial.
  ///
  /// Computes a derivative of \f$ \begin{bmatrix}1 & t & t^2 & \dots &
  /// t^{N-1}\end{bmatrix} \f$ with repect to time. For example, the first
  /// derivative would be \f$ \begin{bmatrix}0 & 1 & 2 t & \dots & (N-1)
  /// t^{N-2}\end{bmatrix} \f$.
  /// @param Derivative derivative to evaluate
  /// @param[out] res_const vector to store the result
  /// @param[in] t
  template <int Derivative, class Derived>
  static void baseCoeffsWithTime(const Eigen::MatrixBase<Derived>& res_const,
                                 _Scalar t) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, N);
    Eigen::MatrixBase<Derived>& res =
        const_cast<Eigen::MatrixBase<Derived>&>(res_const);

    res.setZero();

    if (Derivative < N) {
      res[Derivative] = base_coefficients_(Derivative, Derivative);

      _Scalar _t = t;
      for (int j = Derivative + 1; j < N; j++) {
        res[j] = base_coefficients_(Derivative, j) * _t;
        _t = _t * t;
      }
    }
  }

  static const MatN
      blending_matrix_;  ///< Blending matrix. See \ref computeBlendingMatrix.

  static const MatN base_coefficients_;  ///< Base coefficients matrix.
  ///< See \ref computeBaseCoefficients.

  Eigen::deque<SO3> knots;            ///< Knots
  int64_t dt_ns;                      ///< Knot interval in nanoseconds
  int64_t start_t_ns;                 ///< Start time in nanoseconds
  std::array<_Scalar, 3> pow_inv_dt;  ///< Array with inverse powers of dt
};                                    // namespace basalt

template <int _N, typename _Scalar>
const typename So3Spline<_N, _Scalar>::MatN
    So3Spline<_N, _Scalar>::base_coefficients_ =
        computeBaseCoefficients<_N, _Scalar>();

template <int _N, typename _Scalar>
const typename So3Spline<_N, _Scalar>::MatN
    So3Spline<_N, _Scalar>::blending_matrix_ =
        computeBlendingMatrix<_N, _Scalar, true>();

}  // namespace basalt
