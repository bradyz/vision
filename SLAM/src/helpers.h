#ifndef HELPERS_H
#define HELPERS_H

#include <string>

#include <Eigen/Core>

namespace Helpers {

void readOBJ(const std::string &filename,
             Eigen::MatrixX3d &V, Eigen::MatrixX3i &F);

};

#endif
