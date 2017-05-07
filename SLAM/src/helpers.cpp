#include "Helpers.h"

#include <igl/readOBJ.h>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;

namespace Helpers {

void readOBJ(const string &filename, MatrixX3d &V, MatrixX3i &F) {
  igl::readOBJ(filename, V, F);
}

}
