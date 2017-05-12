#include <iostream>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry> 
#include <Eigen/SVD>

#include "viewer_wrapper.h"
#include "helpers.h"

using namespace std;
using namespace Eigen;

MatrixX3d sampleFromVertices(const MatrixX3d &V, int n) {
  MatrixX3d result(n, 3);

  for (int i = 0; i < n; i++)
    result.row(i) = V.row(rand() % V.rows());

  return result;
}

RowVector3d nearestNeighbor(const RowVector3d &p, const MatrixX3d &S) {
  int closest = 0;
  double closest_sq_dist = (S.row(0) - p).squaredNorm();

  for (int i = 0; i < S.rows(); i++) {
    double tmp_sq_dist = (S.row(i) - p).squaredNorm();

    if (tmp_sq_dist < closest_sq_dist) {
      closest_sq_dist = tmp_sq_dist;
      closest = i;
    }
  }

  return S.row(closest);
}

// Finds correspondences of T's vertices from S.
MatrixX3d findCorrespondences(const MatrixX3d &S, const MatrixX3d &T) {
  int n = T.rows();

  MatrixX3d result(n, 3);

  for (int i = 0; i < n; i++)
    result.row(i) = nearestNeighbor(T.row(i), S);

  return result;
}

void minimizePointToPoint(const MatrixX3d &S, const MatrixX3d &T,
                          MatrixX3d &T_aligned, Matrix3d &R, Vector3d &t) {
  R.setIdentity();
  t.setZero();

  int n = T.rows();

  T_aligned = T;

  for (int iteration = 0; iteration < 100; iteration++) {
    MatrixX3d P = T_aligned;
    MatrixX3d Q = findCorrespondences(S, P);

    Vector3d p_bar = P.colwise().mean().transpose();
    Vector3d q_bar = Q.colwise().mean().transpose();

    Vector3d t_candidate = (q_bar - p_bar);

    // Apply translation to P.
    P.rowwise() += t_candidate.transpose();

    Matrix3d A = Matrix3d::Zero();

    for (int i = 0; i < n; i++) {
      const Vector3d &p_i = P.row(i).transpose();
      const Vector3d &q_i = Q.row(i).transpose();

      A += p_i * q_i.transpose();
    }

    JacobiSVD<Matrix3d> SVD(A, ComputeFullU | ComputeFullV);
    const Matrix3d &U = SVD.matrixU();
    const Matrix3d &V = SVD.matrixV();

    Matrix3d R_candidate = V * U.transpose();

    // Apply rotation to P.
    P = (R_candidate * P.transpose()).transpose();

    // Point to point squared norm.
    double energy = (P - Q).rowwise().squaredNorm().sum();

    cout << "Iteration: " << iteration << " Energy: " << energy << endl;

    // Update cumulative translation and rotation matrices.
    R = R_candidate * R;
    t = R_candidate * (t + t_candidate);

    // Update aligned points.
    T_aligned = P;
  }

  ViewerWrapper viewer;
  viewer.addPoints(T_aligned);
  viewer.addPoints(T);
  viewer.addPoints(S);
  viewer.renderPoints();
  viewer.reset();
}

int main (int argc, char* argv[]) {
  MatrixX3d V_bunny;
  MatrixX3i F_bunny;
  Helpers::readOBJ("../obj/bunny.obj", V_bunny, F_bunny);

  int m = 1000;
  int n = 500;

  // Source.
  MatrixX3d S = sampleFromVertices(V_bunny, m);

  // Target.
  MatrixX3d T = sampleFromVertices(V_bunny, n);

  // Alter the points.
  Matrix3d R = AngleAxisd(0.25 * M_PI, Vector3d::UnitX()).toRotationMatrix();
  Vector3d t = 0.15 * Vector3d::Random(3, 1);

  // Apply rotation then translation.
  T = (R * T.transpose()).transpose();
  T.rowwise() += t.transpose();

  MatrixX3d T_aligned;
  Matrix3d R_solved;
  Vector3d t_solved;
  minimizePointToPoint(S, T, T_aligned, R_solved, t_solved);

  MatrixX3d T_tmp = T;
  T_tmp = (R_solved * T_tmp.transpose()).transpose();
  T_tmp.rowwise() += t_solved.transpose();

  ViewerWrapper viewer;
  viewer.addPoints(T_tmp);
  viewer.addPoints(T);
  viewer.addPoints(S);
  viewer.renderPoints();
  viewer.reset();
}
