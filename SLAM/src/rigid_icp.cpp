#include <iostream>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <igl/per_vertex_normals.h>

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

void sampleFromVerticesWithNormals(const MatrixX3d &V, const MatrixX3i &F, int n,
                                   MatrixX3d &V_sample, MatrixX3d &N_sample) {
  MatrixX3d N;
  igl::per_vertex_normals(V, F, N);

  V_sample.resize(n, 3);
  N_sample.resize(n, 3);

  for (int i = 0; i < n; i++) {
    int index = rand() % V.rows();

    V_sample.row(i) = V.row(index);
    N_sample.row(i) = N.row(index);
  }
}

void applyRotationTranslation(MatrixX3d &V, const Vector3d &R, const Vector3d &t) {
  Matrix3d M = Helpers::toRotationMatrix(R);

  for (int i = 0; i < V.rows(); i++)
    V.row(i) = M * V.row(i).transpose() + t;
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

void nearestNeighborWithNormals(const RowVector3d &p, const MatrixX3d &V,
                                const MatrixX3d &N,
                                RowVector3d &point, RowVector3d &normal) {
  int closest = 0;
  double closest_sq_dist = (V.row(0) - p).squaredNorm();

  for (int i = 0; i < V.rows(); i++) {
    double tmp_sq_dist = (V.row(i) - p).squaredNorm();

    if (tmp_sq_dist < closest_sq_dist) {
      closest_sq_dist = tmp_sq_dist;
      closest = i;
    }
  }

  point = V.row(closest);
  normal = N.row(closest);
}

// Finds correspondences of T's vertices from S.
MatrixX3d findCorrespondences(const MatrixX3d &S, const MatrixX3d &T) {
  int n = T.rows();

  MatrixX3d result(n, 3);

  for (int i = 0; i < n; i++)
    result.row(i) = nearestNeighbor(T.row(i), S);

  return result;
}

void findCorrespondencesWithNormals(const MatrixX3d &V_S, const MatrixX3d &N_S,
                                    const MatrixX3d &V_T,
                                    MatrixX3d &V, MatrixX3d &N) {
  int n = V_T.rows();

  V.resize(n, 3);
  N.resize(n, 3);

  for (int i = 0; i < n; i++) {
    RowVector3d point;
    RowVector3d normal;
    nearestNeighborWithNormals(V_T.row(i), V_S, N_S, point, normal);

    V.row(i) = point;
    N.row(i) = normal;
  }
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
}

void pointToPointDemo() {
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
  Matrix3d R = AngleAxisd(0.1 * M_PI, Vector3d::UnitX()).toRotationMatrix();
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

void minimizePointToPlane(const MatrixX3d &V_S, const MatrixX3d &N_S,
                          const MatrixX3d &V_T) {
  int n = V_T.rows();

  MatrixX3d V = V_T;

  Vector3d R(0.0, 0.0, 0.0);
  Vector3d t(0.0, 0.0, 0.0);

  for (int iteration = 0; iteration < 100; iteration++) {
    MatrixX3d D;
    MatrixX3d N;
    findCorrespondencesWithNormals(V_S, N_S, V, D, N);

    MatrixXd A(n, 6);
    VectorXd b(n);

    A.setZero();
    b.setZero();

    double energy = 0.0;

    for (int i = 0; i < n; i++) {
      double tmp = (D.row(i) - V.row(i)).dot(N.row(i));
      energy += tmp * tmp;

      A(i, 0) = V(i, 1) * N(i, 2) - V(i, 2) * N(i, 1);
      A(i, 1) = V(i, 2) * N(i, 0) - V(i, 0) * N(i, 2);
      A(i, 2) = V(i, 0) * N(i, 1) - V(i, 1) * N(i, 0);
      A(i, 3) = N(i, 0);
      A(i, 4) = N(i, 1);
      A(i, 5) = N(i, 2);

      for (int j = 0; j < 3; j++)
        b(i) += (D(i, j) - V(i, j)) * N(i, j);
    }

    ColPivHouseholderQR<MatrixXd> qr(A);
    VectorXd x = qr.solve(b);

    cout << "Iteration: " << iteration << " Energy: " << energy << endl;

    R = Vector3d(x(0), x(1), x(2));
    t = Vector3d(x(3), x(4), x(5));

    applyRotationTranslation(V, R, t);
  }

  ViewerWrapper viewer;
  viewer.addPoints(V);
  viewer.addPoints(V_T);
  viewer.addPoints(V_S);
  viewer.renderPoints();
  viewer.reset();
}

void pointToPlaneDemo() {
  MatrixX3d V_bunny;
  MatrixX3i F_bunny;
  Helpers::readOBJ("../obj/bunny.obj", V_bunny, F_bunny);

  int n = 1000;
  int m = 500;

  MatrixX3d V_S;
  MatrixX3d N_S;
  sampleFromVerticesWithNormals(V_bunny, F_bunny, n, V_S, N_S);

  MatrixX3d V_T = sampleFromVertices(V_bunny, m);

  Matrix3d R = AngleAxisd(0.15 * M_PI, Vector3d::UnitX()).toRotationMatrix();
  Vector3d t = 0.15 * Vector3d::Random(3, 1);

  V_T = (R * V_T.transpose()).transpose();
  V_T.rowwise() += t.transpose();

  minimizePointToPlane(V_S, N_S, V_T);
}

int main(int argc, char* argv[]) {
  pointToPointDemo();
  pointToPlaneDemo();
}
