#define IGL_VIEWER_VIEWER_QUIET

#include "viewer_wrapper.h"
#include "timer.h"
#include "helpers.h"

#include <Eigen/core>

#include <igl/jet.h>
#include <igl/viewer/Viewer.h>

using namespace std;
using namespace Eigen;

void ViewerWrapper::addMesh(const MatrixX3d &V, const MatrixX3i &F) {
  VectorXd C(V.rows());

  for (int i = 0; i < V.rows(); i++)
    C(i) = colors_.size();

  addMesh(V, F, C);
}

void ViewerWrapper::addMesh(const MatrixX3d &V, const MatrixX3i &F,
                            const VectorXd &C) {
  vertices_.push_back(V);
  faces_.push_back(F);
  colors_.push_back(C);
}

void ViewerWrapper::renderMesh() {
  // Find out the total number of vertex and face rows needed.
  int total_v_rows = 0;
  int total_f_rows = 0;

  for (int i = 0; i < vertices_.size(); i++) {
    total_v_rows += vertices_[i].rows();
    total_f_rows += faces_[i].rows();
  }

  // Reallocate to the proper amount of space.
  MatrixX3d V(total_v_rows, 3);
  MatrixX3i F(total_f_rows, 3);
  VectorXd C(total_v_rows);

  // The current block of vertices_ and faces_.
  int total_v = 0;
  int total_f = 0;

  // Add each mesh to be rendered.
  for (int i = 0; i < vertices_.size(); i++) {
    int nb_v = vertices_[i].rows();
    int nb_f = faces_[i].rows();

    V.block(total_v, 0, nb_v, 3) = vertices_[i];
    F.block(total_f, 0, nb_f, 3) = faces_[i].array() + total_v;
    C.block(total_v, 0, nb_v, 1) = colors_[i];

    // Get the next offset.
    total_v += nb_v;
    total_f += nb_f;
  }

  MatrixX3d C_jet;
  igl::jet(C, true, C_jet);

  // Update the viewer.
  igl::viewer::Viewer viewer;
  viewer.data.set_mesh(V, F);
  viewer.data.set_colors(C_jet);
  viewer.core.align_camera_center(V, F);

  viewer.launch();
}

void ViewerWrapper::addPoints(const MatrixX3d &V) {
  VectorXd C(V.rows());

  for (int i = 0; i < V.rows(); i++)
    C(i) = colors_.size();

  addPoints(V, C);
}

void ViewerWrapper::addPoints(const MatrixX3d &V, const VectorXd &C) {
  points_.push_back(V);
  colors_.push_back(C);
}

void ViewerWrapper::renderPoints() {
  int total_p_rows = 0;

  for (int i = 0; i < points_.size(); i++)
    total_p_rows += points_[i].rows();

  // Reallocate to the proper amount of space.
  MatrixX3d V(total_p_rows, 3);
  VectorXd C(total_p_rows);

  // The current block of points.
  int total_p = 0;

  // Add each mesh to be rendered.
  for (int i = 0; i < points_.size(); i++) {
    int nb_p = points_[i].rows();

    V.block(total_p, 0, nb_p, 3) = points_[i];
    C.block(total_p, 0, nb_p, 1) = colors_[i];

    // Get the next offset.
    total_p += nb_p;
  }

  // Turn the scalars into colors.
  MatrixX3d C_jet;
  igl::jet(C, true, C_jet);

  MatrixX3d V_axis;
  MatrixX2i E_axis;
  MatrixX3d C_axis;
  Helpers::coordinateAxis(V_axis, E_axis, C_axis);

  // Update the viewer.
  igl::viewer::Viewer viewer;
  viewer.data.clear();
  viewer.data.set_points(V, C_jet);
  viewer.data.set_edges(V_axis, E_axis, C_axis);
  viewer.core.align_camera_center(V);

  viewer.launch();
}

void ViewerWrapper::reset() {
  vertices_.clear();
  faces_.clear();
  points_.clear();
  colors_.clear();
}
