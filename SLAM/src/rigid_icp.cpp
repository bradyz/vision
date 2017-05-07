#include <iostream>

#include <Eigen/Core>

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

int main (int argc, char* argv[]) {
  ViewerWrapper viewer;

  MatrixX3d V_bunny;
  MatrixX3i F_bunny;

  Helpers::readOBJ("../obj/bunny.obj", V_bunny, F_bunny);

  int n = 1000;

  // Source.
  MatrixX3d V_s = sampleFromVertices(V_bunny, n);
  VectorXd C_s(n); 
  for (int i = 0; i < n; i++)
    C_s(i) = 100.0;

  // Target.
  MatrixX3d V_t = sampleFromVertices(V_bunny, n);
  VectorXd C_t(n); 
  for (int i = 0; i < n; i++)
    C_t(i) = 200.0;

  viewer.addPoints(V_s, C_s);
  viewer.addPoints(V_t, C_t);
  viewer.renderPoints();
}
