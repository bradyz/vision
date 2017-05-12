#ifndef VIEWER_WRAPPER_H
#define VIEWER_WRAPPER_H

#include <vector>

#include <Eigen/Core>

class ViewerWrapper {

public:
  void addMesh(const Eigen::MatrixX3d &V, const Eigen::MatrixX3i &F);
  void addMesh(const Eigen::MatrixX3d &V,
               const Eigen::MatrixX3i &F,
               const Eigen::VectorXd &C);
  void renderMesh();

  void addPoints(const Eigen::MatrixX3d &V);
  void addPoints(const Eigen::MatrixX3d &V, const Eigen::VectorXd &C);
  void renderPoints();

  void reset();

private:
  std::vector<Eigen::MatrixX3d> vertices_;
  std::vector<Eigen::MatrixX3i> faces_;

  std::vector<Eigen::MatrixX3d> points_;

  std::vector<Eigen::VectorXd> colors_;

};

#endif
