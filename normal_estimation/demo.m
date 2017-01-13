NORMAL_SCALE = 1 / 300;

[vertices, faces] = read_obj('data/bunny.obj');
normals = calculate_normals(vertices);

clf;

for i = 1:size(vertices, 1);
    % All used for plotting.
    x = vertices(i, 1);
    y = vertices(i, 2);
    z = vertices(i, 3);
    u = normals(i, 1);
    v = normals(i, 2);
    w = normals(i, 3);

    quiver3(x, y, z, u, v, w, NORMAL_SCALE);
    hold on;
end
