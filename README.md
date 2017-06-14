## Computer Vision

### Rigid ICP

This algorithm aligns two points clouds.

1. Point to point loss - SVD.
2. Point to plane loss - least squares.

<img src="SLAM/screenshot.png" width="50%">

### Normal Estimation

To compute an estimate of a normal at each point for a point cloud -

1. Take the nearest k neighbors.
2. Compute the covariance matrix.

The eigenvector with the smallest eigenvalue will be a good estimate for the normal.

<img src="normal_estimation/screenshots/bunny_normals.png" width="50%">

### Style Transfer

This is an optimization problem trying to find the optimal image that minimizes a loss function consisting of content loss and style loss.

The content loss is a L2 distance between activation maps.

The style loss is a L2 distance between Gram Matrices (all combinations of dot products) of activation maps.

<img src="visualizing_cnn/candy_cat.png" width="25%"><img src="visualizing_cnn/abstract_cat.png" width="25%"><img src="visualizing_cnn/cubism_cat.png" width="25%"><img src="visualizing_cnn/starry_night_cat.png" width="25%">

### Deep Dream

This is an optimization problem trying to find the optimal image that maximizes a random node in a deep network.

This is equivalent to doing gradient ascent on a loss function that consists just of the node's output.

<img src="visualizing_cnn/deepdream.png" width="50%">

### Face Detector

Given an image x, output a bounding box that encloses a face.

This is done through some naive kernels and using a linear SVM (through gradient descent) that was trained on a batch of 20 positive/negative samples.

<img src="face_detector/face_detector.png" width="50%">
