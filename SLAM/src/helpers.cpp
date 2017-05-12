#include "Helpers.h"

#include <igl/readOBJ.h>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;

namespace Helpers {

void readOBJ(const string &filename, MatrixX3d &V, MatrixX3i &F) {
  igl::readOBJ(filename, V, F);
}

void coordinateAxis(MatrixX3d &V, MatrixX2i &E, MatrixX3d &C) {
  V.resize(4, 3);
  E.resize(3, 2);
  C.resize(3, 3);

  V.row(0) = RowVector3d(0.0, 0.0, 0.0);
  V.row(1) = RowVector3d(1.0, 0.0, 0.0);
  V.row(2) = RowVector3d(0.0, 1.0, 0.0);
  V.row(3) = RowVector3d(0.0, 0.0, 1.0);

  E.row(0) = RowVector2i(0, 1);
  E.row(1) = RowVector2i(0, 2);
  E.row(2) = RowVector2i(0, 3);

  C.row(0) = RowVector3d(1.0, 0.0, 0.0);
  C.row(1) = RowVector3d(0.0, 1.0, 0.0);
  C.row(2) = RowVector3d(0.0, 0.0, 1.0);
}

}
