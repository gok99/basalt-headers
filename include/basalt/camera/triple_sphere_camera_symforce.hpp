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
    const Scalar _tmp0 = -param_(2, 0) + proj(0, 0);
    const Scalar _tmp1 = Scalar(1.0) / (param_(0, 0));
    const Scalar _tmp2 = std::min<Scalar>(
            0, (((param_(6, 0) + Scalar(-0.5)) > 0) - ((param_(6, 0) + Scalar(-0.5)) < 0)));
    const Scalar _tmp3 = 2 * _tmp2;
    const Scalar _tmp4 = epsilon * (_tmp3 + 1) + param_(6, 0);
    const Scalar _tmp5 = _tmp2 + _tmp4;
    const Scalar _tmp6 = Scalar(1.0) / (_tmp5);
    const Scalar _tmp7 = (Scalar(1) / Scalar(2)) * _tmp3 - _tmp4 + 1;
    const Scalar _tmp8 = _tmp6 * _tmp7;
    const Scalar _tmp9 = std::pow(param_(0, 0), Scalar(-2));
    const Scalar _tmp10 = std::pow(_tmp0, Scalar(2));
    const Scalar _tmp11 = std::pow(param_(1, 0), Scalar(-2));
    const Scalar _tmp12 = -param_(3, 0) + proj(1, 0);
    const Scalar _tmp13 = std::pow(_tmp12, Scalar(2));
    const Scalar _tmp14 = _tmp10 * _tmp9 + _tmp11 * _tmp13;
    const Scalar _tmp15 = _tmp14 + 1;
    const Scalar _tmp16 = Scalar(1.0) / (_tmp15);
    const Scalar _tmp17 = std::pow(_tmp5, Scalar(-2));
    const Scalar _tmp18 = std::pow(_tmp7, Scalar(2));
    const Scalar _tmp19 = -_tmp17 * _tmp18 + 1;
    const Scalar _tmp20 = std::sqrt(Scalar(_tmp14 * _tmp19 + 1));
    const Scalar _tmp21 = _tmp20 + _tmp8;
    const Scalar _tmp22 = _tmp16 * _tmp21;
    const Scalar _tmp23 = _tmp22 - _tmp8;
    const Scalar _tmp24 = std::pow(param_(5, 0), Scalar(2));
    const Scalar _tmp25 = std::pow(_tmp23, Scalar(2));
    const Scalar _tmp26 = std::sqrt(Scalar(_tmp24 * _tmp25 - _tmp24 + 1));
    const Scalar _tmp27 = _tmp23 * param_(5, 0) + _tmp26;
    const Scalar _tmp28 = std::pow(param_(4, 0), Scalar(2));
    const Scalar _tmp29 = _tmp23 * _tmp27 - param_(5, 0);
    const Scalar _tmp30 = std::pow(_tmp29, Scalar(2));
    const Scalar _tmp31 = std::sqrt(Scalar(_tmp28 * _tmp30 - _tmp28 + 1));
    const Scalar _tmp32 = _tmp29 * param_(4, 0) + _tmp31;
    const Scalar _tmp33 = _tmp22 * _tmp32;
    const Scalar _tmp34 = _tmp27 * _tmp33;
    const Scalar _tmp35 = _tmp1 * _tmp34;
    const Scalar _tmp36 = Scalar(1.0) / (param_(1, 0));
    const Scalar _tmp37 = _tmp34 * _tmp36;
    const Scalar _tmp38 = _tmp10 / [&]() {
        const Scalar base = param_(0, 0);
        return base * base * base;
    }();
    const Scalar _tmp39 = Scalar(1.0) / (_tmp20);
    const Scalar _tmp40 = _tmp16 * _tmp19 * _tmp39;
    const Scalar _tmp41 = _tmp38 * _tmp40;
    const Scalar _tmp42 = _tmp27 * _tmp32;
    const Scalar _tmp43 = _tmp41 * _tmp42;
    const Scalar _tmp44 = _tmp0 * _tmp9;
    const Scalar _tmp45 = 2 * _tmp21 / std::pow(_tmp15, Scalar(2));
    const Scalar _tmp46 = _tmp44 * _tmp45;
    const Scalar _tmp47 = _tmp40 * _tmp44;
    const Scalar _tmp48 = -_tmp46 + _tmp47;
    const Scalar _tmp49 = Scalar(1.0) / (_tmp26);
    const Scalar _tmp50 = _tmp23 * _tmp24 * _tmp49;
    const Scalar _tmp51 = _tmp48 * _tmp50 + _tmp48 * param_(5, 0);
    const Scalar _tmp52 = _tmp23 * _tmp51 + _tmp27 * _tmp48;
    const Scalar _tmp53 = Scalar(1.0) / (_tmp31);
    const Scalar _tmp54 = _tmp28 * _tmp29 * _tmp53;
    const Scalar _tmp55 = _tmp52 * _tmp54 + _tmp52 * param_(4, 0);
    const Scalar _tmp56 = _tmp0 * _tmp1;
    const Scalar _tmp57 = _tmp22 * _tmp27;
    const Scalar _tmp58 = _tmp56 * _tmp57;
    const Scalar _tmp59 = _tmp33 * _tmp51;
    const Scalar _tmp60 = _tmp38 * _tmp45;
    const Scalar _tmp61 = _tmp42 * _tmp60;
    const Scalar _tmp62 = _tmp12 * _tmp36;
    const Scalar _tmp63 = _tmp42 * _tmp62;
    const Scalar _tmp64 = _tmp47 * _tmp63;
    const Scalar _tmp65 = _tmp57 * _tmp62;
    const Scalar _tmp66 = _tmp46 * _tmp63;
    const Scalar _tmp67 = _tmp11 * _tmp12;
    const Scalar _tmp68 = _tmp40 * _tmp67;
    const Scalar _tmp69 = _tmp42 * _tmp56;
    const Scalar _tmp70 = _tmp68 * _tmp69;
    const Scalar _tmp71 = _tmp45 * _tmp67;
    const Scalar _tmp72 = _tmp68 - _tmp71;
    const Scalar _tmp73 = _tmp50 * _tmp72 + _tmp72 * param_(5, 0);
    const Scalar _tmp74 = _tmp23 * _tmp73 + _tmp27 * _tmp72;
    const Scalar _tmp75 = _tmp54 * _tmp74 + _tmp74 * param_(4, 0);
    const Scalar _tmp76 = _tmp33 * _tmp56;
    const Scalar _tmp77 = _tmp69 * _tmp71;
    const Scalar _tmp78 = _tmp13 / [&]() {
        const Scalar base = param_(1, 0);
        return base * base * base;
    }();
    const Scalar _tmp79 = _tmp40 * _tmp78;
    const Scalar _tmp80 = _tmp42 * _tmp79;
    const Scalar _tmp81 = _tmp33 * _tmp62;
    const Scalar _tmp82 = _tmp45 * _tmp78;
    const Scalar _tmp83 = _tmp42 * _tmp82;
    const Scalar _tmp84 = -_tmp41 + _tmp60;
    const Scalar _tmp85 = _tmp50 * _tmp84 + _tmp84 * param_(5, 0);
    const Scalar _tmp86 = [&]() {
        const Scalar base = _tmp0;
        return base * base * base;
    }() / std::pow(param_(0, 0), Scalar(4));
    const Scalar _tmp87 = _tmp40 * _tmp42;
    const Scalar _tmp88 = _tmp23 * _tmp85 + _tmp27 * _tmp84;
    const Scalar _tmp89 = _tmp54 * _tmp88 + _tmp88 * param_(4, 0);
    const Scalar _tmp90 = _tmp42 * _tmp45;
    const Scalar _tmp91 = -_tmp79 + _tmp82;
    const Scalar _tmp92 = _tmp50 * _tmp91 + _tmp91 * param_(5, 0);
    const Scalar _tmp93 = _tmp23 * _tmp92 + _tmp27 * _tmp91;
    const Scalar _tmp94 = _tmp54 * _tmp93 + _tmp93 * param_(4, 0);
    const Scalar _tmp95 = [&]() {
        const Scalar base = _tmp12;
        return base * base * base;
    }() / std::pow(param_(1, 0), Scalar(4));
    const Scalar _tmp96 = _tmp46 - _tmp47;
    const Scalar _tmp97 = _tmp50 * _tmp96 + _tmp96 * param_(5, 0);
    const Scalar _tmp98 = _tmp23 * _tmp97 + _tmp27 * _tmp96;
    const Scalar _tmp99 = _tmp54 * _tmp98 + _tmp98 * param_(4, 0);
    const Scalar _tmp100 = -_tmp68 + _tmp71;
    const Scalar _tmp101 = _tmp100 * _tmp50 + _tmp100 * param_(5, 0);
    const Scalar _tmp102 = _tmp100 * _tmp27 + _tmp101 * _tmp23;
    const Scalar _tmp103 = _tmp102 * _tmp54 + _tmp102 * param_(4, 0);
    const Scalar _tmp104 = 2 * param_(4, 0);
    const Scalar _tmp105 = _tmp29 + (Scalar(1) / Scalar(2)) * _tmp53 * (_tmp104 * _tmp30 - _tmp104);
    const Scalar _tmp106 = 2 * param_(5, 0);
    const Scalar _tmp107 = _tmp23 + (Scalar(1) / Scalar(2)) * _tmp49 * (_tmp106 * _tmp25 - _tmp106);
    const Scalar _tmp108 = _tmp107 * _tmp23 - 1;
    const Scalar _tmp109 = _tmp108 * _tmp54 + _tmp108 * param_(4, 0);
    const Scalar _tmp110 = _tmp17 * _tmp7;
    const Scalar _tmp111 = _tmp16 * (-_tmp110 +
                                     (Scalar(1) / Scalar(2)) * _tmp14 * _tmp39 *
                                     (2 * _tmp110 + 2 * _tmp18 /
                                                    [&]() {
                                                        const Scalar base = _tmp5;
                                                        return base * base * base;
                                                    }()) -
                                     _tmp6);
    const Scalar _tmp112 = _tmp111 * _tmp42;
    const Scalar _tmp113 = _tmp110 + _tmp111 + _tmp6;
    const Scalar _tmp114 = _tmp113 * _tmp50 + _tmp113 * param_(5, 0);
    const Scalar _tmp115 = _tmp113 * _tmp27 + _tmp114 * _tmp23;
    const Scalar _tmp116 = _tmp115 * _tmp54 + _tmp115 * param_(4, 0);

    // Output terms (3)
    p3d(0, 0) = _tmp0 * _tmp35;
    p3d(1, 0) = _tmp12 * _tmp37;
    p3d(2, 0) = _tmp29 * _tmp32 - param_(4, 0);

    if constexpr (!std::is_same_v<DerivedJ2D, std::nullptr_t>) {
      BASALT_ASSERT(res_D_proj);

      (*res_D_proj)(0, 0) = _tmp35 + _tmp43 + _tmp55 * _tmp58 + _tmp56 * _tmp59 - _tmp61;
      (*res_D_proj)(1, 0) = _tmp55 * _tmp65 + _tmp59 * _tmp62 + _tmp64 - _tmp66;
      (*res_D_proj)(2, 0) = _tmp29 * _tmp55 + _tmp32 * _tmp52;
      (*res_D_proj)(0, 1) = _tmp58 * _tmp75 + _tmp70 + _tmp73 * _tmp76 - _tmp77;
      (*res_D_proj)(1, 1) = _tmp37 + _tmp65 * _tmp75 + _tmp73 * _tmp81 + _tmp80 - _tmp83;
      (*res_D_proj)(2, 1) = _tmp29 * _tmp75 + _tmp32 * _tmp74;

      if (!(*res_D_proj).array().isFinite().all())
        return false;
    } else {
      UNUSED(res_D_proj);
    }

    if constexpr (!std::is_same_v<DerivedJparam, std::nullptr_t>) {
      BASALT_ASSERT(res_D_param);

      (*res_D_param)(0, 0) =
              -_tmp34 * _tmp44 + _tmp58 * _tmp89 + _tmp76 * _tmp85 - _tmp86 * _tmp87 + _tmp86 * _tmp90;
      (*res_D_param)(1, 0) = -_tmp43 * _tmp62 + _tmp61 * _tmp62 + _tmp65 * _tmp89 + _tmp81 * _tmp85;
      (*res_D_param)(2, 0) = _tmp29 * _tmp89 + _tmp32 * _tmp88;
      (*res_D_param)(0, 1) = -_tmp56 * _tmp80 + _tmp56 * _tmp83 + _tmp58 * _tmp94 + _tmp76 * _tmp92;
      (*res_D_param)(1, 1) =
              -_tmp34 * _tmp67 + _tmp65 * _tmp94 + _tmp81 * _tmp92 - _tmp87 * _tmp95 + _tmp90 * _tmp95;
      (*res_D_param)(2, 1) = _tmp29 * _tmp94 + _tmp32 * _tmp93;
      (*res_D_param)(0, 2) = -_tmp35 - _tmp43 + _tmp58 * _tmp99 + _tmp61 + _tmp76 * _tmp97;
      (*res_D_param)(1, 2) = -_tmp64 + _tmp65 * _tmp99 + _tmp66 + _tmp81 * _tmp97;
      (*res_D_param)(2, 2) = _tmp29 * _tmp99 + _tmp32 * _tmp98;
      (*res_D_param)(0, 3) = _tmp101 * _tmp76 + _tmp103 * _tmp58 - _tmp70 + _tmp77;
      (*res_D_param)(1, 3) = _tmp101 * _tmp81 + _tmp103 * _tmp65 - _tmp37 - _tmp80 + _tmp83;
      (*res_D_param)(2, 3) = _tmp102 * _tmp32 + _tmp103 * _tmp29;
      (*res_D_param)(0, 4) = _tmp105 * _tmp58;
      (*res_D_param)(1, 4) = _tmp105 * _tmp65;
      (*res_D_param)(2, 4) = _tmp105 * _tmp29 - 1;
      (*res_D_param)(0, 5) = _tmp107 * _tmp76 + _tmp109 * _tmp58;
      (*res_D_param)(1, 5) = _tmp107 * _tmp81 + _tmp109 * _tmp65;
      (*res_D_param)(2, 5) = _tmp108 * _tmp32 + _tmp109 * _tmp29;
      (*res_D_param)(0, 6) = _tmp112 * _tmp56 + _tmp114 * _tmp76 + _tmp116 * _tmp58;
      (*res_D_param)(1, 6) = _tmp112 * _tmp62 + _tmp114 * _tmp81 + _tmp116 * _tmp65;
      (*res_D_param)(2, 6) = _tmp115 * _tmp32 + _tmp116 * _tmp29;

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
