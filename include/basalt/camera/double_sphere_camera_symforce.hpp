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
class DoubleSphereCameraSymforce {
 public:
  using Scalar = Scalar_;
  static constexpr int N = 6;  ///< Number of intrinsic parameters.

  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

  using VecN = Eigen::Matrix<Scalar, N, 1>;
  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  using Mat24 = Eigen::Matrix<Scalar, 2, 4>;
  using Mat2N = Eigen::Matrix<Scalar, 2, N>;

  using Mat42 = Eigen::Matrix<Scalar, 4, 2>;
  using Mat4N = Eigen::Matrix<Scalar, 4, N>;

  /// @brief Default constructor with zero intrinsics
  DoubleSphereCameraSymforce() : width_(0), height_(0), fov_deg_(220) { param_.setZero(); }

  /// @brief Construct camera model with given vector of intrinsics
  ///
  /// @param[in] p vector of intrinsic parameters [fx, fy, cx, cy, xi, alpha]
  explicit DoubleSphereCameraSymforce(const VecN& p, int fov = 220, int width = 0, int height = 0)
    : param_(p), width_(width), height_(height), fov_deg_(fov) {

    updateR2max();

    // std::cout << "fov_deg_ = " << fov_deg_ << ", r2_max_ = " << r2_max_ << "alpha limit " << Scalar(1) / (Scalar(2) * alpha - Scalar(1)) << std::endl;
  }

  /// @brief Cast to different scalar type
  template <class Scalar2>
  DoubleSphereCameraSymforce<Scalar2> cast() const {
    return DoubleSphereCameraSymforce<Scalar2>(param_.template cast<Scalar2>(), fov_deg_);
  }

  /// @brief Camera model name
  ///
  /// @return "ds"
  static std::string getName() { return "ds-sym"; }

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
    // Total ops: 104

    // Input arrays
    const Scalar epsilon = Scalar(1e-7);

    // Intermediate terms (30)
    const Scalar _tmp0 = 1 - param_(5, 0);
    const Scalar _tmp1 = std::pow(point3d(0, 0), Scalar(2)) + std::pow(point3d(1, 0), Scalar(2));
    const Scalar _tmp2 = std::sqrt(Scalar(_tmp1 + std::pow(point3d(2, 0), Scalar(2))));
    const Scalar _tmp3 = _tmp2 * param_(4, 0) + point3d(2, 0);
    const Scalar _tmp4 = std::sqrt(Scalar(_tmp1 + std::pow(_tmp3, Scalar(2))));
    const Scalar _tmp5 = _tmp0 * _tmp3 + _tmp4 * param_(5, 0) + epsilon;
    const Scalar _tmp6 = Scalar(1.0) / (_tmp5);
    const Scalar _tmp7 = _tmp6 * param_(0, 0);
    const Scalar _tmp8 = _tmp6 * param_(1, 0);
    const Scalar _tmp9 = std::min<Scalar>(
            0, (((param_(5, 0) + Scalar(-0.5)) > 0) - ((param_(5, 0) + Scalar(-0.5)) < 0)));
    const Scalar _tmp10 = 2 * _tmp9;
    const Scalar _tmp11 = epsilon * (_tmp10 + 1) + param_(5, 0);
    const Scalar _tmp12 = ((Scalar(1) / Scalar(2)) * _tmp10 - _tmp11 + 1) / (_tmp11 + _tmp9);
    const Scalar _tmp13 = 2 * point3d(0, 0);
    const Scalar _tmp14 = param_(4, 0) / _tmp2;
    const Scalar _tmp15 = _tmp14 * _tmp3;
    const Scalar _tmp16 = param_(5, 0) / _tmp4;
    const Scalar _tmp17 = (Scalar(1) / Scalar(2)) * _tmp16;
    const Scalar _tmp18 = _tmp0 * _tmp14;
    const Scalar _tmp19 = std::pow(_tmp5, Scalar(-2));
    const Scalar _tmp20 = _tmp19 * (_tmp17 * (_tmp13 * _tmp15 + _tmp13) + _tmp18 * point3d(0, 0));
    const Scalar _tmp21 = param_(0, 0) * point3d(0, 0);
    const Scalar _tmp22 = param_(1, 0) * point3d(1, 0);
    const Scalar _tmp23 = 2 * point3d(1, 0);
    const Scalar _tmp24 = _tmp19 * (_tmp17 * (_tmp15 * _tmp23 + _tmp23) + _tmp18 * point3d(1, 0));
    const Scalar _tmp25 = _tmp14 * point3d(2, 0) + 1;
    const Scalar _tmp26 = _tmp16 * _tmp3;
    const Scalar _tmp27 = _tmp19 * (_tmp0 * _tmp25 + _tmp25 * _tmp26);
    const Scalar _tmp28 = _tmp19 * (_tmp0 * _tmp2 + _tmp2 * _tmp26);
    const Scalar _tmp29 = _tmp19 * (-_tmp3 + _tmp4);

    // Output terms (4)
    proj(0, 0) = _tmp7 * point3d(0, 0) + param_(2, 0);
    proj(1, 0) = _tmp8 * point3d(1, 0) + param_(3, 0);

    bool res = std::max<Scalar>(
            0, (((_tmp2 * (_tmp12 + param_(4, 0)) /
                  (2 * _tmp12 * param_(4, 0) + std::pow(param_(4, 0), Scalar(2)) + 1) -
                  epsilon + point3d(2, 0)) > 0) -
                ((_tmp2 * (_tmp12 + param_(4, 0)) /
                  (2 * _tmp12 * param_(4, 0) + std::pow(param_(4, 0), Scalar(2)) + 1) -
                  epsilon + point3d(2, 0)) < 0))) == Scalar(1);

    if (!res) {
      return false;
    }

    if constexpr (!std::is_same_v<DerivedJ3D, std::nullptr_t>) {
      BASALT_ASSERT(res0_D_point3d);

      // Eigen::Matrix<Scalar, 2, 3>& _res0_D_point3d = (*res0_D_point3d);

      (*res0_D_point3d)(0, 0) = -_tmp20 * _tmp21 + _tmp7;
      (*res0_D_point3d)(1, 0) = -_tmp20 * _tmp22;
      (*res0_D_point3d)(0, 1) = -_tmp21 * _tmp24;
      (*res0_D_point3d)(1, 1) = -_tmp22 * _tmp24 + _tmp8;
      (*res0_D_point3d)(0, 2) = -_tmp21 * _tmp27;
      (*res0_D_point3d)(1, 2) = -_tmp22 * _tmp27;

      if (!(*res0_D_point3d).array().isFinite().all())
        return false;
    } else {
      UNUSED(res0_D_point3d);
    }

    if constexpr (!std::is_same_v<DerivedJparam, std::nullptr_t>) {
      BASALT_ASSERT(res0_D_param);

      // Eigen::Matrix<Scalar, 2, 6>& _res0_D_param = (*res0_D_param);

      (*res0_D_param)(0, 0) = _tmp6 * point3d(0, 0);
      (*res0_D_param)(1, 0) = 0;
      (*res0_D_param)(0, 1) = 0;
      (*res0_D_param)(1, 1) = _tmp6 * point3d(1, 0);
      (*res0_D_param)(0, 2) = 1;
      (*res0_D_param)(1, 2) = 0;
      (*res0_D_param)(0, 3) = 0;
      (*res0_D_param)(1, 3) = 1;
      (*res0_D_param)(0, 4) = -_tmp21 * _tmp28;
      (*res0_D_param)(1, 4) = -_tmp22 * _tmp28;
      (*res0_D_param)(0, 5) = -_tmp21 * _tmp29;
      (*res0_D_param)(1, 5) = -_tmp22 * _tmp29;

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
    // Total ops: 241

    // Input arrays

    // Intermediate terms (102)
    const Scalar _tmp0 = -param_(2, 0) + proj(0, 0);
    const Scalar _tmp1 = epsilon + param_(0, 0);
    const Scalar _tmp2 = Scalar(1.0) / (_tmp1);
    const Scalar _tmp3 = _tmp0 * _tmp2;
    const Scalar _tmp4 = -param_(3, 0) + proj(1, 0);
    const Scalar _tmp5 = std::pow(_tmp4, Scalar(2));
    const Scalar _tmp6 = epsilon + param_(1, 0);
    const Scalar _tmp7 = std::pow(_tmp6, Scalar(-2));
    const Scalar _tmp8 = std::pow(_tmp0, Scalar(2));
    const Scalar _tmp9 = std::pow(_tmp1, Scalar(-2));
    const Scalar _tmp10 = _tmp5 * _tmp7 + _tmp8 * _tmp9;
    const Scalar _tmp11 = 1 - std::pow(param_(4, 0), Scalar(2));
    const Scalar _tmp12 = std::pow(param_(5, 0), Scalar(2));
    const Scalar _tmp13 = -_tmp10 * _tmp12 + 1;
    const Scalar _tmp14 = std::pow(_tmp13, Scalar(2));
    const Scalar _tmp15 = 2 * param_(5, 0);
    const Scalar _tmp16 = _tmp15 - 1;
    const Scalar _tmp17 = std::sqrt(Scalar(-_tmp10 * _tmp16 + 1));
    const Scalar _tmp18 = _tmp17 * param_(5, 0) + epsilon - param_(5, 0) + 1;
    const Scalar _tmp19 = std::pow(_tmp18, Scalar(-2));
    const Scalar _tmp20 = _tmp14 * _tmp19;
    const Scalar _tmp21 = std::sqrt(Scalar(_tmp10 * _tmp11 + _tmp20));
    const Scalar _tmp22 = _tmp21 + _tmp3 * param_(4, 0);
    const Scalar _tmp23 = _tmp10 + _tmp20 + epsilon;
    const Scalar _tmp24 = Scalar(1.0) / (_tmp23);
    const Scalar _tmp25 = _tmp22 * _tmp24;
    const Scalar _tmp26 = Scalar(1.0) / (_tmp6);
    const Scalar _tmp27 = _tmp25 * _tmp26;
    const Scalar _tmp28 = Scalar(1.0) / (_tmp18);
    const Scalar _tmp29 = _tmp13 * _tmp28;
    const Scalar _tmp30 = _tmp0 * _tmp9;
    const Scalar _tmp31 = 2 * _tmp30;
    const Scalar _tmp32 = Scalar(1.0) / (_tmp17);
    const Scalar _tmp33 = _tmp16 * _tmp32;
    const Scalar _tmp34 = _tmp14 / [&]() {
        const Scalar base = _tmp18;
        return base * base * base;
    }();
    const Scalar _tmp35 = _tmp15 * _tmp34;
    const Scalar _tmp36 = _tmp33 * _tmp35;
    const Scalar _tmp37 = _tmp30 * _tmp36;
    const Scalar _tmp38 = _tmp13 * _tmp19;
    const Scalar _tmp39 = 4 * _tmp38;
    const Scalar _tmp40 = _tmp12 * _tmp39;
    const Scalar _tmp41 = _tmp30 * _tmp40;
    const Scalar _tmp42 = _tmp37 - _tmp41;
    const Scalar _tmp43 = _tmp31 + _tmp42;
    const Scalar _tmp44 = _tmp22 / std::pow(_tmp23, Scalar(2));
    const Scalar _tmp45 = _tmp3 * _tmp44;
    const Scalar _tmp46 = _tmp2 * _tmp25;
    const Scalar _tmp47 = _tmp2 * param_(4, 0);
    const Scalar _tmp48 = _tmp11 * _tmp31;
    const Scalar _tmp49 = Scalar(1.0) / (_tmp21);
    const Scalar _tmp50 = (Scalar(1) / Scalar(2)) * _tmp49;
    const Scalar _tmp51 = _tmp24 * (_tmp47 + _tmp50 * (_tmp42 + _tmp48));
    const Scalar _tmp52 = _tmp26 * _tmp4;
    const Scalar _tmp53 = _tmp44 * _tmp52;
    const Scalar _tmp54 = _tmp25 * _tmp28;
    const Scalar _tmp55 = _tmp12 * _tmp54;
    const Scalar _tmp56 = _tmp31 * _tmp55;
    const Scalar _tmp57 = _tmp25 * _tmp30;
    const Scalar _tmp58 = _tmp33 * _tmp38 * param_(5, 0);
    const Scalar _tmp59 = _tmp57 * _tmp58;
    const Scalar _tmp60 = _tmp29 * _tmp44;
    const Scalar _tmp61 = _tmp4 * _tmp7;
    const Scalar _tmp62 = 2 * _tmp61;
    const Scalar _tmp63 = _tmp11 * _tmp62;
    const Scalar _tmp64 = _tmp36 * _tmp61;
    const Scalar _tmp65 = _tmp40 * _tmp61;
    const Scalar _tmp66 = _tmp64 - _tmp65;
    const Scalar _tmp67 = _tmp63 + _tmp66;
    const Scalar _tmp68 = _tmp24 * _tmp3;
    const Scalar _tmp69 = _tmp50 * _tmp68;
    const Scalar _tmp70 = _tmp62 + _tmp66;
    const Scalar _tmp71 = _tmp24 * _tmp52;
    const Scalar _tmp72 = _tmp50 * _tmp71;
    const Scalar _tmp73 = _tmp55 * _tmp62;
    const Scalar _tmp74 = _tmp25 * _tmp61;
    const Scalar _tmp75 = _tmp58 * _tmp74;
    const Scalar _tmp76 = _tmp24 * _tmp29;
    const Scalar _tmp77 = _tmp50 * _tmp76;
    const Scalar _tmp78 = _tmp8 / [&]() {
        const Scalar base = _tmp1;
        return base * base * base;
    }();
    const Scalar _tmp79 = 2 * _tmp78;
    const Scalar _tmp80 = _tmp33 * _tmp78;
    const Scalar _tmp81 = -_tmp35 * _tmp80 + _tmp40 * _tmp78;
    const Scalar _tmp82 = -_tmp79 + _tmp81;
    const Scalar _tmp83 = -_tmp30 * param_(4, 0) + _tmp50 * (-_tmp11 * _tmp79 + _tmp81);
    const Scalar _tmp84 = _tmp24 * _tmp83;
    const Scalar _tmp85 = _tmp25 * _tmp38;
    const Scalar _tmp86 = _tmp85 * param_(5, 0);
    const Scalar _tmp87 = _tmp5 / [&]() {
        const Scalar base = _tmp6;
        return base * base * base;
    }();
    const Scalar _tmp88 = 2 * _tmp87;
    const Scalar _tmp89 = -_tmp36 * _tmp87 + _tmp40 * _tmp87;
    const Scalar _tmp90 = -_tmp11 * _tmp88 + _tmp89;
    const Scalar _tmp91 = -_tmp88 + _tmp89;
    const Scalar _tmp92 = -_tmp37 + _tmp41;
    const Scalar _tmp93 = -_tmp47 + _tmp50 * (-_tmp48 + _tmp92);
    const Scalar _tmp94 = -_tmp31 + _tmp92;
    const Scalar _tmp95 = -_tmp64 + _tmp65;
    const Scalar _tmp96 = -_tmp63 + _tmp95;
    const Scalar _tmp97 = -_tmp62 + _tmp95;
    const Scalar _tmp98 = -_tmp10 * _tmp49 * param_(4, 0) + _tmp3;
    const Scalar _tmp99 = _tmp10 * param_(5, 0);
    const Scalar _tmp100 = _tmp17 - _tmp32 * _tmp99 - 1;
    const Scalar _tmp101 = -2 * _tmp100 * _tmp34 - _tmp39 * _tmp99;

    // Output terms (3)
    p3d(0, 0) = _tmp25 * _tmp3;
    p3d(1, 0) = _tmp27 * _tmp4;
    p3d(2, 0) = _tmp25 * _tmp29 - param_(4, 0);

    if constexpr (!std::is_same_v<DerivedJ2D, std::nullptr_t>) {
      BASALT_ASSERT(res_D_proj);
      (*res_D_proj)(0, 0) = _tmp3 * _tmp51 - _tmp43 * _tmp45 + _tmp46;
      (*res_D_proj)(1, 0) = -_tmp43 * _tmp53 + _tmp51 * _tmp52;
      (*res_D_proj)(2, 0) = _tmp29 * _tmp51 - _tmp43 * _tmp60 - _tmp56 + _tmp59;
      (*res_D_proj)(0, 1) = -_tmp45 * _tmp70 + _tmp67 * _tmp69;
      (*res_D_proj)(1, 1) = _tmp27 - _tmp53 * _tmp70 + _tmp67 * _tmp72;
      (*res_D_proj)(2, 1) = -_tmp60 * _tmp70 + _tmp67 * _tmp77 - _tmp73 + _tmp75;
      if (!(*res_D_proj).array().isFinite().all())
        return false;
    } else {
      UNUSED(res_D_proj);
    }

    if constexpr (!std::is_same_v<DerivedJparam, std::nullptr_t>) {
      BASALT_ASSERT(res_D_param);

      (*res_D_param)(0, 0) = _tmp3 * _tmp84 - _tmp45 * _tmp82 - _tmp57;
      (*res_D_param)(1, 0) = _tmp52 * _tmp84 - _tmp53 * _tmp82;
      (*res_D_param)(2, 0) = _tmp55 * _tmp79 - _tmp60 * _tmp82 + _tmp76 * _tmp83 - _tmp80 * _tmp86;
      (*res_D_param)(0, 1) = -_tmp45 * _tmp91 + _tmp69 * _tmp90;
      (*res_D_param)(1, 1) = -_tmp53 * _tmp91 + _tmp72 * _tmp90 - _tmp74;
      (*res_D_param)(2, 1) =
              -_tmp33 * _tmp86 * _tmp87 + _tmp55 * _tmp88 - _tmp60 * _tmp91 + _tmp77 * _tmp90;
      (*res_D_param)(0, 2) = -_tmp45 * _tmp94 - _tmp46 + _tmp68 * _tmp93;
      (*res_D_param)(1, 2) = -_tmp53 * _tmp94 + _tmp71 * _tmp93;
      (*res_D_param)(2, 2) = _tmp56 - _tmp59 - _tmp60 * _tmp94 + _tmp76 * _tmp93;
      (*res_D_param)(0, 3) = -_tmp45 * _tmp97 + _tmp69 * _tmp96;
      (*res_D_param)(1, 3) = -_tmp27 - _tmp53 * _tmp97 + _tmp72 * _tmp96;
      (*res_D_param)(2, 3) = -_tmp60 * _tmp97 + _tmp73 - _tmp75 + _tmp77 * _tmp96;
      (*res_D_param)(0, 4) = _tmp68 * _tmp98;
      (*res_D_param)(1, 4) = _tmp71 * _tmp98;
      (*res_D_param)(2, 4) = _tmp76 * _tmp98 - 1;
      (*res_D_param)(0, 5) = -_tmp101 * _tmp45 + _tmp101 * _tmp69;
      (*res_D_param)(1, 5) = -_tmp101 * _tmp53 + _tmp101 * _tmp72;
      (*res_D_param)(2, 5) =
              -_tmp10 * _tmp15 * _tmp54 - _tmp100 * _tmp85 - _tmp101 * _tmp60 + _tmp101 * _tmp77;

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
  /// Initializes the camera model to  \f$ \left[f_x, f_y, c_x, c_y, xi, alpha
  /// \right]^T \f$
  ///
  /// @param[in] init vector [fx, fy, cx, cy]
  inline void setFromInit(const Vec4& init, const VecX* ks) {

    if (!ks)
    {
      // roughly fisheye
      param_[4] = -0.2; // xi
      param_[5] = 0.55; // alpha
    }else {
      param_[4] = (*ks)(0);
      param_[5] = (*ks)(1);
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
    param_[5] = std::clamp(param_[5], Sophus::Constants<Scalar>::epsilonSqrt(), Scalar(1) - Sophus::Constants<Scalar>::epsilonSqrt()); // alpha
    updateR2max();
  }

  /// @brief Returns a const reference to the intrinsic parameters vector
  ///
  /// The order is following: \f$ \left[f_x, f_y, c_x, c_y, \xi, \alpha
  /// \right]^T \f$
  /// @return const reference to the intrinsic parameters vector
  const VecN& getParam() const { return param_; }

  /// @brief Projections used for unit-tests
  static Eigen::aligned_vector<DoubleSphereCameraSymforce> getTestProjections() {
    Eigen::aligned_vector<DoubleSphereCameraSymforce> res;

    VecN vec1;
    vec1 << 0.5 * 805, 0.5 * 800, 505, 509, 0.5 * -0.150694, 0.5 * 1.48785;
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
