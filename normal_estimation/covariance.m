function [A] = covariance(points)
% Input:
%   points, an m x n matrix where there are m data points.
%
% Output:
%   M, an n x n matrix.

[m, n] = size(points);

% Centroid.
x_bar = sum(points) ./ m;
x_bar = x_bar';

A = zeros(n, n);
for i = 1:m
    x = points(i, :)';
    A = A + (x - x_bar) * (x - x_bar)';
end
A = A ./ m;

end
