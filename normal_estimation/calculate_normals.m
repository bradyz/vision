function [normals] = calculate_normals(vertices)
% Input:
%   vertices, an m x 3 matrix where there are m data points.
%
% Output:
%   normals, an m x 3 matrix, the corresponding normal estimates.
CLUSTER_SIZE = 10;

knn = knnsearch(vertices, vertices, 'K', CLUSTER_SIZE);

for i = 1:size(vertices, 1);
    % Define the cluster of the K nearest neighbors.
    for j = 1:CLUSTER_SIZE
        points(j, :) = vertices(knn(i, j), :);
    end

    % Note that a covariance matrix is SPD -
    % This means the SVD and Eigendecomposition will yield the same result.
    [U, S_unused, V_unused] = svd(covariance(points));

    % The normal is the smallest principle component.
    normals(i, :) = U(:, 3)';
end

end
