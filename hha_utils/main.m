clc;
addpath('./utils/nyu-hooks');
% matrix_filename = 'camera_rotations_NYU.txt';
depth_image_root = '../datasets/data/depth'       % dir where depth images are in.
rawdepth_image_root = '../datasets/data/rawdepth'       % dir where raw depth images are in.
hha_image_root = '../datasets/data/hha'

C = getCameraParam('color');

for i=1:1449
    i
    matrix = C;    %camera_matrix(1+(i-1)*3:i*3,:);        % matrix of this image, 3*3
    D = imread(fullfile(depth_image_root, '/', ['img_', mat2str(5000+i),'.png']));

    % here, RD is the same as D, because there is some problem about NYU Depth V2 raw-depth-images
    RD = imread(fullfile(rawdepth_image_root, '/', ['img_', mat2str(5000+i),'.png']));
    hha = saveHHA(['img_', mat2str(5000+i)], matrix, hha_image_root, D, RD);
    % hha = saveHHA(['complete_img_', mat2str(5000+i)], matrix, hha_image_root, D, D);
end 