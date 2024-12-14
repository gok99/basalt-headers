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
@brief Implementation of double sphere camera model
*/

#pragma once

#include <basalt/camera/camera_static_assert.hpp>

#include <basalt/utils/sophus_utils.hpp>

namespace basalt {

using std::sqrt;

/// @brief Double Sphere camera model
///
/// \image html ds.png
/// This model has N=6 parameters \f$ \mathbf{i} = \left[f_x, f_y, c_x, c_y,
/// \xi, \alpha \right]^T \f$ with \f$ \xi \in [-1,1], \alpha \in [0,1] \f$. See
/// \ref project and \ref unproject functions for more details.
template <typename Scalar_ = double>
class TripleSphereCameraSymforce {
 public:
  using Scalar = Scalar_;
  static constexpr int N = 7;  ///< Number of intrinsic parameters.

  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

  using VecN = Eigen::Matrix<Scalar, N, 1>;
  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  using Mat24 = Eigen::Matrix<Scalar, 2, 4>;
  using Mat2N = Eigen::Matrix<Scalar, 2, N>;

  using Mat42 = Eigen::Matrix<Scalar, 4, 2>;
  using Mat4N = Eigen::Matrix<Scalar, 4, N>;

  /// @brief Default constructor with zero intrinsics
  TripleSphereCameraSymforce() : width_(0), height_(0), fov_deg_(220) { param_.setZero(); }

  /// @brief Construct camera model with given vector of intrinsics
  ///
  /// @param[in] p vector of intrinsic parameters [fx, fy, cx, cy, xi, alpha]
  explicit TripleSphereCameraSymforce(const VecN& p, int fov = 220, int width = 0, int height = 0)
    : param_(p), width_(width), height_(height), fov_deg_(fov) {

    updateR2max();

    // std::cout << "fov_deg_ = " << fov_deg_ << ", r2_max_ = " << r2_max_ << "alpha limit " << Scalar(1) / (Scalar(2) * alpha - Scalar(1)) << std::endl;
  }

  /// @brief Cast to different scalar type
  template <class Scalar2>
  TripleSphereCameraSymforce<Scalar2> cast() const {
    return TripleSphereCameraSymforce<Scalar2>(param_.template cast<Scalar2>(), fov_deg_);
  }

  /// @brief Camera model name
  ///
  /// @return "ts"
  static std::string getName() { return "ts"; }

  /**
   * Project a 3D point to a 2D point using a sphere projection model.
   *
   * Args:
   *     point3d (sf.V3): 3D point
   *     param (sf.V6): projection parameters
   *
   * Returns:
   *     sf.V2: 2D point
   *     res0_D_point3d: (2x3) jacobian of res0 (2) wrt arg point3d (3)
   *     res0_D_param: (2x6) jacobian of res0 (2) wrt arg param (6)
   */
  template <class DerivedPoint3D, class DerivedPoint2D,
          class DerivedJ3D = std::nullptr_t,
          class DerivedJparam = std::nullptr_t>
  inline bool project(const Eigen::MatrixBase<DerivedPoint3D>& point3d,
                      Eigen::MatrixBase<DerivedPoint2D>& proj,
                      DerivedJ3D res0_D_point3d = nullptr,
                      DerivedJparam res0_D_param = nullptr) const {
    checkProjectionDerivedTypes<DerivedPoint3D, DerivedPoint2D, DerivedJ3D,
            DerivedJparam, N>();
    const Scalar epsilon = Scalar(1e-7);
    // Total ops: 140

    // Input arrays

    // Intermediate terms (44)
    const Scalar _tmp0 = std::pow(point3d(0, 0), Scalar(2)) + std::pow(point3d(1, 0), Scalar(2));
    const Scalar _tmp1 = std::sqrt(Scalar(_tmp0 + std::pow(point3d(2, 0), Scalar(2))));
    const Scalar _tmp2 = _tmp1 * param_(4, 0) + point3d(2, 0);
    const Scalar _tmp3 = std::sqrt(Scalar(_tmp0 + std::pow(_tmp2, Scalar(2))));
    const Scalar _tmp4 = _tmp2 + _tmp3 * param_(5, 0);
    const Scalar _tmp5 = std::sqrt(Scalar(_tmp0 + std::pow(_tmp4, Scalar(2))));
    const Scalar _tmp6 = epsilon - param_(6, 0) + 1;
    const Scalar _tmp7 = Scalar(1.0) / (_tmp6);
    const Scalar _tmp8 = _tmp7 * param_(6, 0);
    const Scalar _tmp9 = _tmp4 + _tmp5 * _tmp8;
    const Scalar _tmp10 = _tmp9 + epsilon;
    const Scalar _tmp11 = Scalar(1.0) / (_tmp10);
    const Scalar _tmp12 = _tmp9 * param_(2, 0) + param_(0, 0) * point3d(0, 0);
    const Scalar _tmp13 = _tmp9 * param_(3, 0) + param_(1, 0) * point3d(1, 0);
    const Scalar _tmp14 = param_(4, 0) + param_(5, 0);
    const Scalar _tmp15 = 2 * point3d(0, 0);
    const Scalar _tmp16 = param_(4, 0) / _tmp1;
    const Scalar _tmp17 = _tmp16 * _tmp2;
    const Scalar _tmp18 = param_(5, 0) / _tmp3;
    const Scalar _tmp19 = (Scalar(1) / Scalar(2)) * _tmp18;
    const Scalar _tmp20 = _tmp16 * point3d(0, 0) + _tmp19 * (_tmp15 * _tmp17 + _tmp15);
    const Scalar _tmp21 = 2 * _tmp4;
    const Scalar _tmp22 = _tmp8 / _tmp5;
    const Scalar _tmp23 = (Scalar(1) / Scalar(2)) * _tmp22;
    const Scalar _tmp24 = _tmp20 + _tmp23 * (_tmp15 + _tmp20 * _tmp21);
    const Scalar _tmp25 = std::pow(_tmp10, Scalar(-2));
    const Scalar _tmp26 = _tmp12 * _tmp25;
    const Scalar _tmp27 = _tmp11 * param_(3, 0);
    const Scalar _tmp28 = _tmp13 * _tmp25;
    const Scalar _tmp29 = 2 * point3d(1, 0);
    const Scalar _tmp30 = _tmp16 * point3d(1, 0) + _tmp19 * (_tmp17 * _tmp29 + _tmp29);
    const Scalar _tmp31 = _tmp23 * (_tmp21 * _tmp30 + _tmp29) + _tmp30;
    const Scalar _tmp32 = _tmp11 * param_(2, 0);
    const Scalar _tmp33 = _tmp16 * point3d(2, 0) + 1;
    const Scalar _tmp34 = _tmp18 * _tmp2;
    const Scalar _tmp35 = _tmp33 * _tmp34 + _tmp33;
    const Scalar _tmp36 = _tmp22 * _tmp4;
    const Scalar _tmp37 = _tmp35 * _tmp36 + _tmp35;
    const Scalar _tmp38 = _tmp11 * _tmp9;
    const Scalar _tmp39 = _tmp1 * _tmp34 + _tmp1;
    const Scalar _tmp40 = _tmp36 * _tmp39 + _tmp39;
    const Scalar _tmp41 = _tmp3 * _tmp36 + _tmp3;
    const Scalar _tmp42 = _tmp11 * _tmp41;
    const Scalar _tmp43 = _tmp5 * _tmp7 + _tmp5 * param_(6, 0) / std::pow(_tmp6, Scalar(2));

    proj(0, 0) = _tmp11 * _tmp12;
    proj(1, 0) = _tmp11 * _tmp13;

    bool res = std::max<Scalar>(0, (((_tmp1 * (_tmp14 + _tmp8) /
        std::sqrt(Scalar(std::pow(_tmp14, Scalar(2)) + 2 * _tmp14 * _tmp8 + 1)) +
        point3d(2, 0)) > 0) -
      ((_tmp1 * (_tmp14 + _tmp8) /
        std::sqrt(Scalar(std::pow(_tmp14, Scalar(2)) + 2 * _tmp14 * _tmp8 + 1)) +
        point3d(2, 0)) < 0)));

    if (!res) {
      return false;
    }

    if constexpr (!std::is_same_v<DerivedJ3D, std::nullptr_t>) {
      BASALT_ASSERT(res0_D_point3d);

      // Eigen::Matrix<Scalar, 2, 3>& _res0_D_point3d = (*res0_D_point3d);

      (*res0_D_point3d)(0, 0) = _tmp11 * (_tmp24 * param_(2, 0) + param_(0, 0)) - _tmp24 * _tmp26;
      (*res0_D_point3d)(1, 0) = _tmp24 * _tmp27 - _tmp24 * _tmp28;
      (*res0_D_point3d)(0, 1) = -_tmp26 * _tmp31 + _tmp31 * _tmp32;
      (*res0_D_point3d)(1, 1) = _tmp11 * (_tmp31 * param_(3, 0) + param_(1, 0)) - _tmp28 * _tmp31;
      (*res0_D_point3d)(0, 2) = -_tmp26 * _tmp37 + _tmp32 * _tmp37;
      (*res0_D_point3d)(1, 2) = _tmp27 * _tmp37 - _tmp28 * _tmp37;

      if (!(*res0_D_point3d).array().isFinite().all())
        return false;
    } else {
      UNUSED(res0_D_point3d);
    }

    if constexpr (!std::is_same_v<DerivedJparam, std::nullptr_t>) {
      BASALT_ASSERT(res0_D_param);

      // Eigen::Matrix<Scalar, 2, 6>& _res0_D_param = (*res0_D_param);

      (*res0_D_param)(0, 0) = _tmp11 * point3d(0, 0);
      (*res0_D_param)(1, 0) = 0;
      (*res0_D_param)(0, 1) = 0;
      (*res0_D_param)(1, 1) = _tmp11 * point3d(1, 0);
      (*res0_D_param)(0, 2) = _tmp38;
      (*res0_D_param)(1, 2) = 0;
      (*res0_D_param)(0, 3) = 0;
      (*res0_D_param)(1, 3) = _tmp38;
      (*res0_D_param)(0, 4) = -_tmp26 * _tmp40 + _tmp32 * _tmp40;
      (*res0_D_param)(1, 4) = _tmp27 * _tmp40 - _tmp28 * _tmp40;
      (*res0_D_param)(0, 5) = -_tmp26 * _tmp41 + _tmp42 * param_(2, 0);
      (*res0_D_param)(1, 5) = -_tmp28 * _tmp41 + _tmp42 * param_(3, 0);
      (*res0_D_param)(0, 6) = -_tmp26 * _tmp43 + _tmp32 * _tmp43;
      (*res0_D_param)(1, 6) = _tmp27 * _tmp43 - _tmp28 * _tmp43;

      if (!(*res0_D_param).array().isFinite().all())
        return false;
    } else {
      UNUSED(res0_D_param);
    }

    return true;
  }  // NOLINT(readability/fn_size)

  template <class DerivedPoint2D, class DerivedPoint3D,
          class DerivedJ2D = std::nullptr_t,
          class DerivedJparam = std::nullptr_t>
  bool unproject(
          const Eigen::MatrixBase<DerivedPoint2D>& proj,
          Eigen::MatrixBase<DerivedPoint3D>& p3d,
          DerivedJ2D res_D_proj = nullptr,
          DerivedJparam res_D_param = nullptr) const {
    checkUnprojectionDerivedTypes<DerivedPoint2D, DerivedPoint3D, DerivedJ2D,
            DerivedJparam, N>();
    const Scalar epsilon = Scalar(1e-7);

    // Total ops: 362

    // Input arrays

    // Intermediate terms (127)
    const Scalar _tmp0 = std::pow(param_(0, 0), Scalar(-2));
    const Scalar _tmp1 = -param_(2, 0) + proj(0, 0);
    const Scalar _tmp2 = std::pow(_tmp1, Scalar(2));
    const Scalar _tmp3 = std::pow(param_(1, 0), Scalar(-2));
    const Scalar _tmp4 = -param_(3, 0) + proj(1, 0);
    const Scalar _tmp5 = std::pow(_tmp4, Scalar(2));
    const Scalar _tmp6 = _tmp0 * _tmp2 + _tmp3 * _tmp5;
    const Scalar _tmp7 = std::min<Scalar>(
            0, (((param_(6, 0) + Scalar(-0.5)) > 0) - ((param_(6, 0) + Scalar(-0.5)) < 0)));
    const Scalar _tmp8 = 2 * _tmp7;
    const Scalar _tmp9 = epsilon * (_tmp8 + 1) + param_(6, 0);
    const Scalar _tmp10 = _tmp7 + _tmp9;
    const Scalar _tmp11 = std::pow(_tmp10, Scalar(-2));
    const Scalar _tmp12 = (Scalar(1) / Scalar(2)) * _tmp8 - _tmp9 + 1;
    const Scalar _tmp13 = std::pow(_tmp12, Scalar(2));
    const Scalar _tmp14 = -_tmp11 * _tmp13 + 1;
    const Scalar _tmp15 = std::sqrt(Scalar(_tmp14 * _tmp6 + 1));
    const Scalar _tmp16 = _tmp15 + param_(4, 0);
    const Scalar _tmp17 = Scalar(1.0) / (param_(0, 0));
    const Scalar _tmp18 = _tmp6 + 1;
    const Scalar _tmp19 = Scalar(1.0) / (_tmp18);
    const Scalar _tmp20 = _tmp16 * _tmp19;
    const Scalar _tmp21 = Scalar(1.0) / (_tmp10);
    const Scalar _tmp22 = -_tmp12 * _tmp21 + _tmp20;
    const Scalar _tmp23 = std::pow(param_(5, 0), Scalar(2));
    const Scalar _tmp24 = std::pow(_tmp22, Scalar(2));
    const Scalar _tmp25 = std::sqrt(Scalar(_tmp23 * _tmp24 - _tmp23 + 1));
    const Scalar _tmp26 = _tmp22 * param_(5, 0) + _tmp25;
    const Scalar _tmp27 = _tmp22 * _tmp26 - param_(5, 0);
    const Scalar _tmp28 = std::pow(param_(4, 0), Scalar(2));
    const Scalar _tmp29 = std::pow(_tmp27, Scalar(2));
    const Scalar _tmp30 = std::sqrt(Scalar(_tmp28 * _tmp29 - _tmp28 + 1));
    const Scalar _tmp31 = _tmp27 * param_(4, 0) + _tmp30;
    const Scalar _tmp32 = _tmp19 * _tmp26;
    const Scalar _tmp33 = _tmp31 * _tmp32;
    const Scalar _tmp34 = _tmp17 * _tmp33;
    const Scalar _tmp35 = _tmp1 * _tmp34;
    const Scalar _tmp36 = Scalar(1.0) / (param_(1, 0));
    const Scalar _tmp37 = _tmp33 * _tmp36;
    const Scalar _tmp38 = _tmp16 * _tmp37;
    const Scalar _tmp39 = _tmp2 / [&]() {
        const Scalar base = param_(0, 0);
        return base * base * base;
    }();
    const Scalar _tmp40 = Scalar(1.0) / (_tmp15);
    const Scalar _tmp41 = _tmp14 * _tmp40;
    const Scalar _tmp42 = _tmp33 * _tmp41;
    const Scalar _tmp43 = _tmp39 * _tmp42;
    const Scalar _tmp44 = _tmp0 * _tmp1;
    const Scalar _tmp45 = 2 * _tmp16 / std::pow(_tmp18, Scalar(2));
    const Scalar _tmp46 = _tmp44 * _tmp45;
    const Scalar _tmp47 = _tmp19 * _tmp40;
    const Scalar _tmp48 = _tmp14 * _tmp47;
    const Scalar _tmp49 = _tmp44 * _tmp48;
    const Scalar _tmp50 = -_tmp46 + _tmp49;
    const Scalar _tmp51 = Scalar(1.0) / (_tmp25);
    const Scalar _tmp52 = _tmp22 * _tmp23 * _tmp51;
    const Scalar _tmp53 = _tmp50 * _tmp52 + _tmp50 * param_(5, 0);
    const Scalar _tmp54 = _tmp20 * _tmp31;
    const Scalar _tmp55 = _tmp1 * _tmp17;
    const Scalar _tmp56 = _tmp54 * _tmp55;
    const Scalar _tmp57 = _tmp39 * _tmp45;
    const Scalar _tmp58 = _tmp26 * _tmp31;
    const Scalar _tmp59 = _tmp57 * _tmp58;
    const Scalar _tmp60 = _tmp16 * _tmp34;
    const Scalar _tmp61 = _tmp22 * _tmp53 + _tmp26 * _tmp50;
    const Scalar _tmp62 = Scalar(1.0) / (_tmp30);
    const Scalar _tmp63 = _tmp27 * _tmp28;
    const Scalar _tmp64 = _tmp62 * _tmp63;
    const Scalar _tmp65 = _tmp61 * _tmp64 + _tmp61 * param_(4, 0);
    const Scalar _tmp66 = _tmp16 * _tmp32;
    const Scalar _tmp67 = _tmp55 * _tmp66;
    const Scalar _tmp68 = _tmp36 * _tmp4;
    const Scalar _tmp69 = _tmp54 * _tmp68;
    const Scalar _tmp70 = _tmp37 * _tmp4;
    const Scalar _tmp71 = _tmp41 * _tmp70;
    const Scalar _tmp72 = _tmp44 * _tmp71;
    const Scalar _tmp73 = _tmp46 * _tmp58 * _tmp68;
    const Scalar _tmp74 = _tmp66 * _tmp68;
    const Scalar _tmp75 = _tmp3 * _tmp4;
    const Scalar _tmp76 = _tmp35 * _tmp41;
    const Scalar _tmp77 = _tmp75 * _tmp76;
    const Scalar _tmp78 = _tmp45 * _tmp75;
    const Scalar _tmp79 = _tmp55 * _tmp58 * _tmp78;
    const Scalar _tmp80 = _tmp48 * _tmp75;
    const Scalar _tmp81 = -_tmp78 + _tmp80;
    const Scalar _tmp82 = _tmp52 * _tmp81 + _tmp81 * param_(5, 0);
    const Scalar _tmp83 = _tmp22 * _tmp82 + _tmp26 * _tmp81;
    const Scalar _tmp84 = _tmp64 * _tmp83 + _tmp83 * param_(4, 0);
    const Scalar _tmp85 = _tmp5 / [&]() {
        const Scalar base = param_(1, 0);
        return base * base * base;
    }();
    const Scalar _tmp86 = _tmp42 * _tmp85;
    const Scalar _tmp87 = _tmp45 * _tmp85;
    const Scalar _tmp88 = _tmp58 * _tmp87;
    const Scalar _tmp89 = [&]() {
        const Scalar base = _tmp1;
        return base * base * base;
    }() / std::pow(param_(0, 0), Scalar(4));
    const Scalar _tmp90 = -_tmp39 * _tmp48 + _tmp57;
    const Scalar _tmp91 = _tmp52 * _tmp90 + _tmp90 * param_(5, 0);
    const Scalar _tmp92 = _tmp54 * _tmp91;
    const Scalar _tmp93 = _tmp45 * _tmp58;
    const Scalar _tmp94 = _tmp16 * _tmp33;
    const Scalar _tmp95 = _tmp22 * _tmp91 + _tmp26 * _tmp90;
    const Scalar _tmp96 = _tmp64 * _tmp95 + _tmp95 * param_(4, 0);
    const Scalar _tmp97 = -_tmp48 * _tmp85 + _tmp87;
    const Scalar _tmp98 = _tmp52 * _tmp97 + _tmp97 * param_(5, 0);
    const Scalar _tmp99 = _tmp22 * _tmp98 + _tmp26 * _tmp97;
    const Scalar _tmp100 = _tmp64 * _tmp99 + _tmp99 * param_(4, 0);
    const Scalar _tmp101 = [&]() {
        const Scalar base = _tmp4;
        return base * base * base;
    }() / std::pow(param_(1, 0), Scalar(4));
    const Scalar _tmp102 = _tmp46 - _tmp49;
    const Scalar _tmp103 = _tmp102 * _tmp52 + _tmp102 * param_(5, 0);
    const Scalar _tmp104 = _tmp103 * _tmp54;
    const Scalar _tmp105 = _tmp102 * _tmp26 + _tmp103 * _tmp22;
    const Scalar _tmp106 = _tmp105 * _tmp64 + _tmp105 * param_(4, 0);
    const Scalar _tmp107 = _tmp106 * _tmp66;
    const Scalar _tmp108 = _tmp78 - _tmp80;
    const Scalar _tmp109 = _tmp108 * _tmp52 + _tmp108 * param_(5, 0);
    const Scalar _tmp110 = _tmp108 * _tmp26 + _tmp109 * _tmp22;
    const Scalar _tmp111 = _tmp110 * _tmp64 + _tmp110 * param_(4, 0);
    const Scalar _tmp112 = _tmp19 * _tmp52 + _tmp19 * param_(5, 0);
    const Scalar _tmp113 = _tmp112 * _tmp22 + _tmp32;
    const Scalar _tmp114 = 2 * param_(4, 0);
    const Scalar _tmp115 =
            _tmp113 * param_(4, 0) + _tmp27 +
            (Scalar(1) / Scalar(2)) * _tmp62 * (2 * _tmp113 * _tmp63 + _tmp114 * _tmp29 - _tmp114);
    const Scalar _tmp116 = 2 * param_(5, 0);
    const Scalar _tmp117 = _tmp22 + (Scalar(1) / Scalar(2)) * _tmp51 * (_tmp116 * _tmp24 - _tmp116);
    const Scalar _tmp118 = _tmp117 * _tmp22 - 1;
    const Scalar _tmp119 = _tmp118 * _tmp64 + _tmp118 * param_(4, 0);
    const Scalar _tmp120 = _tmp11 * _tmp12;
    const Scalar _tmp121 = (Scalar(1) / Scalar(2)) * _tmp6 * (2 * _tmp120 + 2 * _tmp13 / [&]() {
        const Scalar base = _tmp10;
        return base * base * base;
    }());
    const Scalar _tmp122 = _tmp121 * _tmp40;
    const Scalar _tmp123 = _tmp120 + _tmp121 * _tmp47 + _tmp21;
    const Scalar _tmp124 = _tmp123 * _tmp52 + _tmp123 * param_(5, 0);
    const Scalar _tmp125 = _tmp123 * _tmp26 + _tmp124 * _tmp22;
    const Scalar _tmp126 = _tmp125 * _tmp64 + _tmp125 * param_(4, 0);

    // Output terms (3)
    p3d(0, 0) = _tmp16 * _tmp35;
    p3d(1, 0) = _tmp38 * _tmp4;
    p3d(2, 0) = _tmp27 * _tmp31 - param_(4, 0);

    if constexpr (!std::is_same_v<DerivedJ2D, std::nullptr_t>) {
      BASALT_ASSERT(res_D_proj);

      (*res_D_proj)(0, 0) = _tmp43 + _tmp53 * _tmp56 - _tmp59 + _tmp60 + _tmp65 * _tmp67;
      (*res_D_proj)(1, 0) = _tmp53 * _tmp69 + _tmp65 * _tmp74 + _tmp72 - _tmp73;
      (*res_D_proj)(2, 0) = _tmp27 * _tmp65 + _tmp31 * _tmp61;
      (*res_D_proj)(0, 1) = _tmp56 * _tmp82 + _tmp67 * _tmp84 + _tmp77 - _tmp79;
      (*res_D_proj)(1, 1) = _tmp38 + _tmp69 * _tmp82 + _tmp74 * _tmp84 + _tmp86 - _tmp88;
      (*res_D_proj)(2, 1) = _tmp27 * _tmp84 + _tmp31 * _tmp83;

      if (!(*res_D_proj).array().isFinite().all())
        return false;
    } else {
      UNUSED(res_D_proj);
    }

    if constexpr (!std::is_same_v<DerivedJparam, std::nullptr_t>) {
      BASALT_ASSERT(res_D_param);

      (*res_D_param)(0, 0) =
              -_tmp42 * _tmp89 - _tmp44 * _tmp94 + _tmp55 * _tmp92 + _tmp67 * _tmp96 + _tmp89 * _tmp93;
      (*res_D_param)(1, 0) = -_tmp39 * _tmp71 + _tmp59 * _tmp68 + _tmp68 * _tmp92 + _tmp74 * _tmp96;
      (*res_D_param)(2, 0) = _tmp27 * _tmp96 + _tmp31 * _tmp95;
      (*res_D_param)(0, 1) = _tmp100 * _tmp67 + _tmp55 * _tmp88 + _tmp56 * _tmp98 - _tmp76 * _tmp85;
      (*res_D_param)(1, 1) =
              _tmp100 * _tmp74 - _tmp101 * _tmp42 + _tmp101 * _tmp93 + _tmp69 * _tmp98 - _tmp75 * _tmp94;
      (*res_D_param)(2, 1) = _tmp100 * _tmp27 + _tmp31 * _tmp99;
      (*res_D_param)(0, 2) = _tmp104 * _tmp55 + _tmp107 * _tmp55 - _tmp43 + _tmp59 - _tmp60;
      (*res_D_param)(1, 2) = _tmp104 * _tmp68 + _tmp107 * _tmp68 - _tmp72 + _tmp73;
      (*res_D_param)(2, 2) = _tmp105 * _tmp31 + _tmp106 * _tmp27;
      (*res_D_param)(0, 3) = _tmp109 * _tmp56 + _tmp111 * _tmp67 - _tmp77 + _tmp79;
      (*res_D_param)(1, 3) = _tmp109 * _tmp69 + _tmp111 * _tmp74 - _tmp38 - _tmp86 + _tmp88;
      (*res_D_param)(2, 3) = _tmp110 * _tmp31 + _tmp111 * _tmp27;
      (*res_D_param)(0, 4) = _tmp112 * _tmp56 + _tmp115 * _tmp67 + _tmp35;
      (*res_D_param)(1, 4) = _tmp112 * _tmp69 + _tmp115 * _tmp74 + _tmp70;
      (*res_D_param)(2, 4) = _tmp113 * _tmp31 + _tmp115 * _tmp27 - 1;
      (*res_D_param)(0, 5) = _tmp117 * _tmp56 + _tmp119 * _tmp67;
      (*res_D_param)(1, 5) = _tmp117 * _tmp69 + _tmp119 * _tmp74;
      (*res_D_param)(2, 5) = _tmp118 * _tmp31 + _tmp119 * _tmp27;
      (*res_D_param)(0, 6) = _tmp122 * _tmp35 + _tmp124 * _tmp56 + _tmp126 * _tmp67;
      (*res_D_param)(1, 6) = _tmp122 * _tmp70 + _tmp124 * _tmp69 + _tmp126 * _tmp74;
      (*res_D_param)(2, 6) = _tmp125 * _tmp31 + _tmp126 * _tmp27;

      if (!(*res_D_param).array().isFinite().all())
        return false;
    } else {
      UNUSED(res_D_param);
    }

    return true;
  }  // NOLINT(readability/fn_size)

  inline bool inBound(const Vec2& proj) const{

    const Scalar& fx = param_[0];
    const Scalar& fy = param_[1];
    const Scalar& cx = param_[2];
    const Scalar& cy = param_[3];

    // const Scalar& alpha = param_[5];

    const Scalar mx = (proj[0] - cx) / fx;
    const Scalar my = (proj[1] - cy) / fy;

    const Scalar r2 = mx * mx + my * my;

    // fov check
    if (r2 > r2_max_)
      return false;

    // image size check
    if (width_ > 0 && height_ > 0) {
      if (proj[0] < 0 || proj[1] < 0)
        return false;
      if (proj[0] > width_ && proj[1] > height_)
        return false;
    }

    // if (alpha > Scalar(0.5)) {
    //   // hm: the bigger the apparent alpha > 0.5, the smaller the acceptable region of r^2
    //   if (r2 >= Scalar(1) / (Scalar(2) * alpha - Scalar(1))) return false;
    // }

    return true;
  }

  inline void makeInBound(Vec2& proj) const{

    const Scalar& fx = param_[0];
    const Scalar& fy = param_[1];
    const Scalar& cx = param_[2];
    const Scalar& cy = param_[3];

    const Scalar mx = (proj[0] - cx) / fx;
    const Scalar my = (proj[1] - cy) / fy;

    const Scalar r2 = mx * mx + my * my;

    if (r2 > r2_max_) {
      Scalar ratio = sqrt(r2_max_ / r2) * Scalar(1.0 - 1e-4);

      proj[0] = mx * ratio * fx + cx;
      proj[1] = my * ratio * fy + cy;
    }

  }

  /// @brief Set parameters from initialization
  ///
  /// Initializes the camera model to  \f$ \left[f_x, f_y, c_x, c_y, xi, lambda, alpha
  /// \right]^T \f$
  ///
  /// @param[in] init vector [fx, fy, cx, cy]
  inline void setFromInit(const Vec4& init, const VecX* ks) {

    if (!ks)
    {
      // roughly fisheye
      param_[4] = -0.2; // xi
      param_[5] = 0.5;  // lambda
      param_[6] = 0.55; // alpha
    }else {
      param_[4] = (*ks)(0);
      param_[5] = (*ks)(1);
      param_[6] = (*ks)(2);
    }

    // hm: scale the focal length accordingly
    param_[0] = (Scalar(1) + param_[4]) * init[0];
    param_[1] = (Scalar(1) + param_[4]) * init[1];
    param_[2] = init[2];
    param_[3] = init[3];


    
    updateR2max();
  }

  inline void scaleParam(double scale) {
    param_[0] *= scale;
    param_[1] *= scale;
    param_[2] *= scale;
    param_[3] *= scale;
    updateR2max();
  }

  /// @brief Increment intrinsic parameters by inc and clamp the values to the
  /// valid range
  ///
  /// @param[in] inc increment vector
  void operator+=(const VecN& inc) {
    param_ += inc;
    param_[4] = std::clamp(param_[4], Scalar(-1), Scalar(1)); // xi
    param_[5] = std::clamp(param_[5], Scalar(-1), Scalar(1));  // lambda
    param_[6] = std::clamp(param_[6], Sophus::Constants<Scalar>::epsilonSqrt(), Scalar(1) - Sophus::Constants<Scalar>::epsilonSqrt()); // alpha
    updateR2max();
  }

  /// @brief Returns a const reference to the intrinsic parameters vector
  ///
  /// The order is following: \f$ \left[f_x, f_y, c_x, c_y, \xi, \alpha
  /// \right]^T \f$
  /// @return const reference to the intrinsic parameters vector
  const VecN& getParam() const { return param_; }

  /// @brief Projections used for unit-tests
  static Eigen::aligned_vector<TripleSphereCameraSymforce> getTestProjections() {
    Eigen::aligned_vector<TripleSphereCameraSymforce> res;

    VecN vec1;
    vec1 << 0.5 * 805, 0.5 * 800, 505, 509, 0.5 * -0.150694, 0.5 * -0.150694, 0.5 * 1.48785;
    res.emplace_back(vec1);

    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param_;
  int width_, height_; // for bound checking only
  int fov_deg_;
  Scalar r2_max_;

  void updateR2max() {
    // calculate r2_max

    int width_backup = width_;
    int height_backup = height_;

    width_= 0;
    height_ = 0;

    // sweep through radial
    int d = 0;
    r2_max_ = Scalar(1e10); // put to max first
    Scalar r2_max = Scalar(0);
    std::cout << "fov_deg_ = " << fov_deg_ << ", d = " << d << std::endl;
    for (; d <= fov_deg_ / 2; d++) {
      // construct 3d point given the degree
      Eigen::Matrix<Scalar, 3, 1> p3d;
      p3d[0] = std::sin(Scalar(d) / Scalar(180 / M_PI));
      p3d[1] = Scalar(0);
      p3d[2] = std::cos(Scalar(d) / Scalar(180 / M_PI));

      Vec2 p2d;

      bool success = project(p3d, p2d);

      std::cout << "Current p2d: " << p2d << std::endl;
      std::cout << "Current d: " << d << ", success: " << success << std::endl;

      if (!success)
        break;

      // update

      const Scalar& fx = param_[0];
      const Scalar& fy = param_[1];
      const Scalar& cx = param_[2];
      const Scalar& cy = param_[3];

      const Scalar mx = (p2d[0] - cx) / fx;
      const Scalar my = (p2d[1] - cy) / fy;

      const Scalar r2 = mx * mx + my * my;

      assert(r2 >= r2_max); // no wrapping around should happen

      r2_max = r2;
      // std::cout << r2_max << " ";
    }
    // std::cout << "r2_max" << std::endl;
    assert(d > 0);
    assert(r2_max > 0);

    r2_max_ = r2_max;
    fov_deg_ = std::min(fov_deg_, (d-1)*2);
    width_ = width_backup;
    height_ = height_backup;


    const Scalar& alpha = param_[5];
    if (alpha > Scalar(0.5)) {
      r2_max_ = std::min(r2_max_, Scalar(1) / (Scalar(2) * alpha - Scalar(1)) - std::numeric_limits<Scalar>::epsilon());
    }
  }
};

}  // namespace basalt
