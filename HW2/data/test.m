img1 = imread('incline_L.png');
img2 = imread('incline_R.png');

[H1, W1] = deal(size(img1, 1), size(img1, 2));
[H2, W2] = deal(size(img2, 1), size(img2, 2));

corner = [1 1 1; 1 H2 1; W2 1 1; W2 H2 1];
display(corner)

H2to1 = [-1.81474345e-03  9.37095331e-05 -9.98722122e-01;
  2.16467672e-04 -2.42530304e-03  5.03720097e-02;
 9.65527113e-07  1.53082933e-08 -2.74833167e-03];

% perform warping on the corners
warp_c = H2to1 * corner';

% divided by the last line of dummy
dummy = repmat(warp_c(3,:), 3, 1); 
warp_c = warp_c ./ dummy;

% generate matrix M for scaling and translating
translate_width = min(min(warp_c(1,:)), 1);
width = max(max(warp_c(1,:)), W1) - translate_width;

translate_height = min(min(warp_c(2,:)), 1);
height = max(max(warp_c(2,:)), H1) - translate_height;

W_out = 2000;
scalar = W_out / width;
out_size = [ceil(scalar * height) W_out];

M = [scalar 0 -translate_width; 0 scalar -translate_height; 0 0 1];