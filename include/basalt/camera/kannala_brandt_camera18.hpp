/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt-headers.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
Copyright (c) 2022, Otto Seiskari. Spectacular AI Ltd.
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
@brief Implementation of Brown-Conrady camera model
*/

#pragma once

#include <basalt/camera/camera_static_assert.hpp>
#include <basalt/utils/sophus_utils.hpp>
#include <array>
#include <basalt/utils/automatic_differentiation.hpp>

namespace basalt {

/// @brief Full non-radial Kannala-Brandt camera model with 18 distortion coefficients
///
/// This model has N=12 parameters \f$ \mathbf{i} = \left[f_x, f_y, c_x, c_y, 
/// k_1, k_2, k_3, k_4, l_1, l_2, l_3, m_1, m_2, m_3,
/// i_1, i_2, i_3, i_4, j_1, j_2, j_3, j_4 \right]^T \f$ with. See \ref
/// project and \ref unproject functions for more details.
template <typename Scalar_ = double>
class KannalaBrandtCamera18 {
 public:
  using Scalar = Scalar_;
  static constexpr int N = 22;  ///< Number of intrinsic parameters.

  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

  using VecN = Eigen::Matrix<Scalar, N, 1>;

  using Mat24 = Eigen::Matrix<Scalar, 2, 4>;
  using Mat2N = Eigen::Matrix<Scalar, 2, N>;

  using Mat42 = Eigen::Matrix<Scalar, 4, 2>;
  using Mat4N = Eigen::Matrix<Scalar, 4, N>;

  /// @brief Default constructor with zero intrinsics
  KannalaBrandtCamera18() { param_.setZero(); }

  /// @brief Construct camera model with given vector of intrinsics
  ///
  /// @param[in] p vector of intrinsic parameters
  explicit KannalaBrandtCamera18(const VecN& p) { param_ = p; }

  /// @brief Cast to different scalar type
  template <class Scalar2>
  KannalaBrandtCamera18<Scalar2> cast() const {
    return KannalaBrandtCamera18<Scalar2>(param_.template cast<Scalar2>());
  }

  /// @brief Camera model name
  ///
  /// @return "brown-conrady"
  static std::string getName() { return "kb18"; }

  /// @brief Project the point and optionally compute Jacobians
  ///
  /// @param[in] p3d point to project
  /// @param[out] proj result of projection
  /// @param[out] d_proj_d_p3d if not nullptr computed Jacobian of projection
  /// with respect to p3d
  /// @param[out] d_proj_d_param point if not nullptr computed Jacobian of
  /// projection with respect to intrinsic parameters
  /// @return if projection is valid
  template <class DerivedPoint3D, class DerivedPoint2D,
            class DerivedJ3D = std::nullptr_t,
            class DerivedJparam = std::nullptr_t>
  inline bool project(const Eigen::MatrixBase<DerivedPoint3D>& p3d,
                      Eigen::MatrixBase<DerivedPoint2D>& proj,
                      DerivedJ3D d_proj_d_p3d = nullptr,
                      DerivedJparam d_proj_d_param = nullptr) const {
    checkProjectionDerivedTypes<DerivedPoint3D, DerivedPoint2D, DerivedJ3D,
                                DerivedJparam, N>();

    const typename EvalOrReference<DerivedPoint3D>::Type p3d_eval(p3d);

    const Scalar& x = p3d_eval[0];
    const Scalar& y = p3d_eval[1];
    const Scalar& z = p3d_eval[2];

    if (!project(param_, x, y, z, proj[0], proj[1])) return false;
    const bool is_valid = z >= Sophus::Constants<Scalar>::epsilonSqrt();

    if constexpr (!std::is_same_v<DerivedJ3D, std::nullptr_t>) {
      BASALT_ASSERT(d_proj_d_p3d);
      autodiff::Jet<Scalar, 3> xJet, yJet, zJet, projXJet, projYJet;
      xJet.a = x;
      yJet.a = y;
      zJet.a = z;
      xJet.v << 1, 0, 0;
      yJet.v << 0, 1, 0;
      zJet.v << 0, 0, 1;
      if (!project(param_, xJet, yJet, zJet, projXJet, projYJet)) return false;
      (*d_proj_d_p3d).setZero();
      (*d_proj_d_p3d).template block<1, 3>(0, 0) = projXJet.v;
      (*d_proj_d_p3d).template block<1, 3>(1, 0) = projYJet.v;
    } else {
      UNUSED(d_proj_d_p3d);
    }

    if constexpr (!std::is_same_v<DerivedJparam, std::nullptr_t>) {
      BASALT_ASSERT(d_proj_d_param);
      autodiff::Jet<Scalar, N> projXJet, projYJet;
      if (!project(parameterJets(), x, y, z, projXJet, projYJet)) return false;
      (*d_proj_d_param).row(0) = projXJet.v;
      (*d_proj_d_param).row(1) = projYJet.v;
    } else {
      UNUSED(d_proj_d_param);
    }

    return is_valid;
  }

  /// @brief Unproject the point and optionally compute Jacobians
  ///
  /// @param[in] proj point to unproject
  /// @param[out] p3d result of unprojection
  /// @param[out] d_p3d_d_proj if not nullptr computed Jacobian of
  /// unprojection with respect to proj
  /// @param[out] d_p3d_d_param point if not nullptr computed Jacobian of
  /// unprojection with respect to intrinsic parameters
  /// @return if unprojection is valid
  template <class DerivedPoint2D, class DerivedPoint3D,
            class DerivedJ2D = std::nullptr_t,
            class DerivedJparam = std::nullptr_t>
  inline bool unproject(const Eigen::MatrixBase<DerivedPoint2D>& proj,
                        Eigen::MatrixBase<DerivedPoint3D>& p3d,
                        DerivedJ2D d_p3d_d_proj = nullptr,
                        DerivedJparam d_p3d_d_param = nullptr) const {
    checkUnprojectionDerivedTypes<DerivedPoint2D, DerivedPoint3D, DerivedJ2D,
                                  DerivedJparam, N>();

    const typename EvalOrReference<DerivedPoint2D>::Type proj_eval(proj);

    if (!unproject(param_, proj_eval[0], proj_eval[1], p3d[0], p3d[1], p3d[2])) return false;

    if constexpr (!std::is_same_v<DerivedJ2D, std::nullptr_t>) {
      BASALT_ASSERT(d_p3d_d_proj);

      autodiff::Jet<Scalar, 3> projXJet, projYJet, xJet, yJet, zJet;
      projXJet.a = proj_eval[0];
      projYJet.a = proj_eval[1];
      projXJet.v << 1, 0;
      projYJet.v << 0, 1;
      if (!unproject(param_, projXJet, projYJet, xJet, yJet, zJet)) return false;
      (*d_p3d_d_proj).row(0) = xJet.v;
      (*d_p3d_d_proj).row(1) = yJet.v;
      (*d_p3d_d_proj).row(2) = zJet.v;
    } else {
      UNUSED(d_p3d_d_proj);
    }

    if constexpr (!std::is_same_v<DerivedJparam, std::nullptr_t>) {
      BASALT_ASSERT(d_p3d_d_param);

      autodiff::Jet<Scalar, N> xJet, yJet, zJet;
      if (!unproject(parameterJets(), proj_eval[0], proj_eval[1], xJet, yJet, zJet)) return false;
      (*d_p3d_d_param).row(0) = xJet.v;
      (*d_p3d_d_param).row(1) = yJet.v;
      (*d_p3d_d_param).row(2) = zJet.v;
    } else {
      UNUSED(d_p3d_d_param);
    }

    return true;
  }

  /// @brief Set parameters from initialization
  ///
  /// Initializes the camera model to  \f$ \left[f_x, f_y, c_x, c_y, \right]^T
  /// \f$
  ///
  /// @param[in] init vector [fx, fy, cx, cy]
  inline void setFromInit(const Vec4& init) {
    param_[0] = init[0];
    param_[1] = init[1];
    param_[2] = init[2];
    param_[3] = init[3];
    for (size_t j = 4; j < N; ++j)
      param_[j] = Scalar(0.01); // do not set to all zeros (saddle point)
  }

  /// @brief Increment intrinsic parameters by inc
  ///
  /// @param[in] inc increment vector
  void operator+=(const VecN& inc) { param_ += inc; }

  /// @brief Returns a const reference to the intrinsic parameters vector
  ///
  /// The order is following: \f$ \left[f_x, f_y, c_x, c_y, \right]^T \f$
  /// @return const reference to the intrinsic parameters vector
  const VecN& getParam() const { return param_; }

  // TODO: test projections

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:

  std::array<autodiff::Jet<Scalar, N>, N> parameterJets() const {
    std::array<autodiff::Jet<Scalar, N>, N> jets;
    for (int i = 0; i < N; ++i) {
      jets[i].a = param_[i];
      jets[i].v.setZero();
      jets[i].v[i] = 1;
    }
    return jets;
  }

  template <class OutPointT, class ParamT, class PointT, class GradientT>
  static inline OutPointT distortGeneric(
    const ParamT &m1,
    const ParamT &m2,
    const ParamT &m3,
    const ParamT &j1,
    const ParamT &j2,
    const ParamT &j3,
    const ParamT &j4,
    const PointT &theta,
    const PointT &rx,
    const PointT &ry,
    GradientT *gradient)
  {
    const auto &t = theta;
    const auto t2 = t*t;

    const auto cosP = rx;
    const auto sinP = ry;
    const auto cos2P = Scalar(1) - Scalar(2) * sinP * sinP;
    const auto sin2P = Scalar(2) * cosP * sinP;

    const auto phiRadial = t*(m1 + t2*(m2 + t2*m3));
    const auto phiDep = j1 * cosP + j2 * sinP + j3 * cos2P + j4 * sin2P;

    if (gradient != nullptr) {
      (*gradient)[0] = phiDep * (m1 + Scalar(3)*t2*(m2 + Scalar(5.0/3)*t2*m3));

      constexpr auto dcosPDrx = Scalar(1);
      constexpr auto dsinPDrx = Scalar(0);
      constexpr auto dcos2PDrx = Scalar(0);
      const auto dsin2PDrx = Scalar(2) * sinP;

      constexpr auto dcosPDry = Scalar(0);
      constexpr auto dsinPDry = Scalar(1);
      const auto dcos2PDry = -Scalar(4) * sinP;
      const auto dsin2PDry = Scalar(2) * cosP;

      const auto dphiDepDrx = j1 * dcosPDrx + j2 * dsinPDrx + j3 * dcos2PDrx + j4 * dsin2PDrx;
      const auto dphiDepDry = j1 * dcosPDry + j2 * dsinPDry + j3 * dcos2PDry + j4 * dsin2PDry;

      (*gradient)[1] = phiRadial * dphiDepDrx;
      (*gradient)[2] = phiRadial * dphiDepDry;
    }

    return phiRadial * phiDep;
  }

  template <class OutPointT, class ParamArrayT, class PointT, class GradientT>
  static inline OutPointT distortTangential(
    const ParamArrayT &p,
    const PointT &theta,
    const PointT &rx,
    const PointT &ry,
    GradientT *gradient
  ) {
    const auto &m1 = p[11];
    const auto &m2 = p[12];
    const auto &m3 = p[13];
    // const auto &i1 = p[14];
    // const auto &i2 = p[15];
    // const auto &i3 = p[16];
    // const auto &i4 = p[17];
    const auto &j1 = p[18];
    const auto &j2 = p[19];
    const auto &j3 = p[20];
    const auto &j4 = p[21];

    return distortGeneric<OutPointT>(m1, m2, m3, j1, j2, j3, j4, theta, rx, ry, gradient);
  }

  template <class OutPointT, class ParamArrayT, class PointT, class GradientT>
  static inline OutPointT distortRadial(
    const ParamArrayT &p,
    const PointT &theta,
    const PointT &rx,
    const PointT &ry,
    GradientT *gradient
  ) {
    const auto &k0 = p[4];
    const auto &k1 = p[5];
    const auto &k2 = p[6];
    const auto &k3 = p[7];

    const auto &l1 = p[8];
    const auto &l2 = p[9];
    const auto &l3 = p[10];

    // const auto &m1 = p[11];
    // const auto &m2 = p[12];
    // const auto &m3 = p[13];

    const auto &i1 = p[14];
    const auto &i2 = p[15];
    const auto &i3 = p[16];
    const auto &i4 = p[17];

    const auto &t = theta;
    const auto t2 = t*t;

    const auto result = t*(Scalar(1) + t2*(k0 + t2*(k1 + t2*(k2 + t2*k3))))
      + distortGeneric<OutPointT>(l1, l2, l3, i1, i2, i3, i4, theta, rx, ry, gradient);

    if constexpr (!std::is_same_v<GradientT, std::nullptr_t>) {
      if (gradient != nullptr)
        (*gradient)[0] = (*gradient)[0] + Scalar(1) + Scalar(3)*t2*(k0 + Scalar(5.0/3)*t2*(k1 + Scalar(7.0/5)*t2*(k2 + Scalar(9.0/7)*t2*k3)));
    }

    return result;
  }

  template <class ParamArrayT, class PointT, class OutPointT>
  static inline bool undistort(
    const ParamArrayT &p,
    PointT &x1,
    PointT &y1,
    OutPointT &x0,
    OutPointT &y0
  ) {
    constexpr int MAX_ITER = 100;
    constexpr double EPS = 1e-5;

    OutPointT x = x1;
    OutPointT y = y1;
    OutPointT xDistorted, yDistorted;
    int iter = 0;
    std::array<std::array<OutPointT, 2>, 2> J, Jinv;
    double deltaNorm2 = 0;

    do {
      if (!distort(p, x, y, xDistorted, yDistorted, &J)) return false;

      const auto errX = x1 - xDistorted;
      const auto errY = y1 - yDistorted;

      // 2x2 matrix inverse (slow and unstable)
      const auto Jdet = J[0][0]*J[1][1] - J[0][1]*J[1][0];

      if (std::abs(autodiff::asScalar<Scalar>(Jdet)) < EPS) return false;

      const auto invDet = Scalar(1.0) / Jdet;
      Jinv[0][0] = invDet * J[1][1];
      Jinv[1][1] = invDet * J[0][0];
      Jinv[0][1] = -invDet * J[1][0];
      Jinv[1][0] = -invDet * J[0][1];

      const auto deltaX = Jinv[0][0] * errX + Jinv[0][1] * errY;
      const auto deltaY = Jinv[1][0] * errX + Jinv[1][1] * errY;

      x = x + deltaX;
      y = y + deltaY;

      deltaNorm2 = std::pow(autodiff::asScalar<Scalar>(errX), 2) + std::pow(autodiff::asScalar<Scalar>(errY), 2);

    } while (deltaNorm2 > EPS*EPS && ++iter < MAX_ITER);

    x0 = x;
    y0 = y;
    return true;
  }

  template <class ParamArrayT, class PointT, class OutPointT, class JacobianT>
  static inline bool distort(
    const ParamArrayT &p,
    const PointT &x0,
    const PointT &y0,
    OutPointT &xd,
    OutPointT &yd,
    JacobianT *jacobian)
  {
    const PointT xy0norm2 = x0*x0 + y0*y0;
    if (autodiff::asScalar<Scalar>(xy0norm2) > Scalar(1)) return false;

    const PointT z0 = autodiff::sqrt(Scalar(1) - xy0norm2);

    const PointT theta = autodiff::acos(z0);

    // Could also set a stricter value. May help in some cases
    constexpr Scalar maxValidThetaRad = M_PI / 2;

    if (autodiff::asScalar<Scalar>(theta) > maxValidThetaRad || autodiff::asScalar<Scalar>(z0) <= Scalar(0))
      return false;

    const PointT r0 = autodiff::sqrt(xy0norm2);

    const bool computeJacobian = jacobian != nullptr;

    constexpr Scalar DIR_EPS = 1e-6;
    if (autodiff::asScalar<Scalar>(r0) < DIR_EPS) {
      xd = autodiff::maybeJet<OutPointT>(x0);
      yd = autodiff::maybeJet<OutPointT>(y0);

      if constexpr (!std::is_same_v<JacobianT, std::nullptr_t>) {
        if (computeJacobian) {
          (*jacobian)[0][0] = autodiff::maybeJet<OutPointT>(Scalar(1));
          (*jacobian)[0][1] = autodiff::maybeJet<OutPointT>(Scalar(0));
          (*jacobian)[1][0] = autodiff::maybeJet<OutPointT>(Scalar(0));
          (*jacobian)[1][1] = autodiff::maybeJet<OutPointT>(Scalar(1));
        }
      }

      return true;
    }

    const PointT invxy0norm = Scalar(1) / r0;

    // radial direction
    const PointT rx = x0 * invxy0norm;
    const PointT ry = y0 * invxy0norm;

    // tangential drection
    const PointT tx = -ry;
    const PointT ty = rx;

    std::array<OutPointT, 3> gradientRadial, gradientTangential;
    auto *gradientRadialPtr = &gradientRadial, *gradientTangentialPtr = &gradientTangential;

    if (!computeJacobian) {
      gradientRadialPtr = nullptr;
      gradientTangentialPtr = nullptr;
    }

    const OutPointT r = distortRadial<OutPointT>(p, theta, rx, ry, gradientRadialPtr);
    const OutPointT t = distortTangential<OutPointT>(p, theta, rx, ry, gradientTangentialPtr);

    if constexpr (!std::is_same_v<JacobianT, std::nullptr_t>) {
      if (computeJacobian) {
        const PointT dxy0norm2Dx0 = Scalar(2)*x0;
        const PointT dxy0norm2Dy0 = Scalar(2)*y0;

        const PointT outerDsqrt = Scalar(0.5) / z0;
        const PointT dz0Dx0 = -outerDsqrt * dxy0norm2Dx0;
        const PointT dz0Dy0 = -outerDsqrt * dxy0norm2Dy0;

        const PointT outerDacos = -invxy0norm;

        const PointT dthetaDx0 = outerDacos * dz0Dx0;
        const PointT dthetaDy0 = outerDacos * dz0Dy0;

        const PointT drxDx0 = invxy0norm * (Scalar(1) - rx*rx);
        const PointT drxDy0 = invxy0norm * (-rx*ry);
        const PointT dryDx0 = drxDy0;
        const PointT dryDy0 = invxy0norm * (Scalar(1) - ry*ry);

        const PointT dtxDx0 = -dryDx0;
        const PointT dtxDy0 = -dryDy0;
        const PointT dtyDx0 = drxDx0;
        const PointT dtyDy0 = drxDy0;

        const PointT drDx0 = gradientRadial[0] * dthetaDx0
            + gradientRadial[1] * drxDx0
            + gradientRadial[2] * dryDx0;

        const PointT drDy0 = gradientRadial[0] * dthetaDy0
            + gradientRadial[1] * drxDy0
            + gradientRadial[2] * dryDy0;

        const PointT dtDx0 = gradientTangential[0] * dthetaDx0
            + gradientTangential[1] * drxDx0
            + gradientTangential[2] * dryDx0;

        const PointT dtDy0 = gradientTangential[0] * dthetaDy0
            + gradientTangential[1] * drxDy0
            + gradientTangential[2] * dryDy0;

        (*jacobian)[0][0] = drDx0 * rx + dtDx0 * tx + r * drxDx0 + t * dtxDx0;
        (*jacobian)[0][1] = drDy0 * rx + dtDy0 * tx + r * drxDy0 + t * dtxDy0;
        (*jacobian)[1][0] = drDx0 * ry + dtDx0 * ty + r * dryDx0 + t * dtyDx0;
        (*jacobian)[1][1] = drDy0 * ry + dtDy0 * ty + r * dryDy0 + t * dtyDy0;
      }
    } else {
      UNUSED(jacobian);
    }

    xd = r * rx + t * tx;
    yd = r * ry + t * ty;
    return true;
  }

  template <class ParamArrayT, class PointT, class ProjT>
  static inline bool project(
    const ParamArrayT &p,
    const PointT &x,
    const PointT &y,
    const PointT &z,
    ProjT &projX,
    ProjT &projY
  ) {
    const PointT norm2 = x*x + y*y + z*z;
    if (autodiff::asScalar<Scalar>(norm2) <= Scalar(0)) return false;
    // Note: the model could support z < 0 too but this implementation does not
    if (autodiff::asScalar<Scalar>(z) <= Scalar(0)) return false;
    const PointT invDist = Scalar(1) / autodiff::sqrt(norm2);
    const PointT x0 = x * invDist;
    const PointT y0 = y * invDist;

    ProjT xd, yd;
    if (!distort<ParamArrayT, PointT, ProjT, std::nullptr_t>(p, x0, y0, xd, yd, nullptr)) return false;

    const auto &fx = p[0];
    const auto &fy = p[1];
    const auto &cx = p[2];
    const auto &cy = p[3];
    
    projX = fx * xd + cx;
    projY = fy * yd + cy;
    return true;
  }

  template <class ParamArrayT, class ProjT, class PointT>
  static inline bool unproject(
    const ParamArrayT &p,
    const ProjT &projX,
    const ProjT &projY,
    PointT &x0,
    PointT &y0,
    PointT &z0
  ) {
    const auto &fx = p[0];
    const auto &fy = p[1];
    const auto &cx = p[2];
    const auto &cy = p[3];

    auto mx = (projX - cx) / fx;
    auto my = (projY - cy) / fy;
    if (!undistort(p, mx, my, x0, y0)) return false;

    const PointT xy0norm2 = x0*x0 + y0*y0;
    if (autodiff::asScalar<Scalar>(xy0norm2) > Scalar(1)) return false;
    z0 = autodiff::sqrt(Scalar(1) - xy0norm2);
    return true;
  }

  VecN param_;
};

}  // namespace basalt
