function [pos, quat, R_w2c, t_w2c] = estimate_pose(sensor, varargin)
%ESTIMATE_POSE 6DOF pose estimator based on apriltags
%   sensor - struct stored in provided dataset, fields include
%          - is_ready: logical, indicates whether sensor data is valid
%          - rpy, omg, acc: imu readings
%          - img: uint8, 240x376 grayscale image
%          - id: 1xn ids of detected tags
%          - p0, p1, p2, p3, p4: 2xn pixel position of center and
%                                four corners of detected tags

%   pos - 3x1 position of the quadrotor in world frame
%   q   - 4x1 quaternion of the quadrotor [w, x, y, z] 


calib_matrix = [311.0520 0 201.8724; 0 311.3885 113.6210; 0 0 1];

cam2robot = [0.7071   -0.7071         0   -0.04;
           -0.7071   -0.7071         0       0;
             0         0        -1.0000   -0.03;
             0         0         0       1]; 

% generate a map of world coordinates of the tag corners

tags = [  0, 12, 24, 36, 48, 60, 72, 84,  96;
          1, 13, 25, 37, 49, 61, 73, 85,  97;
          2, 14, 26, 38, 50, 62, 74, 86,  98;
          3, 15, 27, 39, 51, 63, 75, 87,  99;
          4, 16, 28, 40, 52, 64, 76, 88, 100;
          5, 17, 29, 41, 53, 65, 77, 89, 101;
          6, 18, 30, 42, 54, 66, 78, 90, 102;
          7, 19, 31, 43, 55, 67, 79, 91, 103;
          8, 20, 32, 44, 56, 68, 80, 92, 104;
          9, 21, 33, 45, 57, 69, 81, 93, 105;
         10, 22, 34, 46, 58, 70, 82, 94, 106;
         11, 23, 35, 47, 59, 71, 83, 95, 107  ];

w = 0.152; 
Xw_array = (2*w)*ones(size(tags));
Xw_array(1,:) = 0; 
Xw_array = cumsum(Xw_array);

Yw_array = (2*w)*ones(size(tags));
Yw_array(:,1) = 0; 
Yw_array(:,4) = w+0.178; 
Yw_array(:,7) = w+0.178;
Yw_array = cumsum(Yw_array,2);

ids = sensor.id + 1;

% calculate world&image coordinates for tags captured
p_w = [ Xw_array(ids)+w/2     Xw_array(ids)+w         Xw_array(ids)+w     Xw_array(ids)       Xw_array(ids);
        Yw_array(ids)+w/2     Yw_array(ids)           Yw_array(ids)+w     Yw_array(ids)+w     Yw_array(ids)];
 
p_i = [sensor.p0 sensor.p1 sensor.p2 sensor.p3 sensor.p4];

if isempty(ids) % return empty for empty packets
    pos =[];
    quat =[];
    R_w2c = [];
    t_w2c = [];
    return
end

u = (p_i(1,:))';
v = (p_i(2,:))';
x_w = p_w(1,:)';
y_w = p_w(2,:)';

%% compute homography matrix
num = size(x_w,1); 
A = zeros(num*2,9);

A(1:num,1:3) = [x_w y_w ones(num,1)];
A(1:num,7:9) = -A(1:num,1:3).*(u*[1 1 1]);
A(num+1:2*num,4:6) =  A(1:num,1:3);
A(num+1:2*num,7:9) = -A(1:num,1:3).*(v*[1 1 1]);

[~,~,V] = svd(A,'econ'); 
h = V(:,end);
H = [h(1:3)';h(4:6)';h(7:9)'];
H = H./H(3,3); 


%% pose estimation

r1 = calib_matrix\H(:,1) / norm(calib_matrix\H(:,1));
r2 = calib_matrix\H(:,2) / norm(calib_matrix\H(:,2)); 
t_w2c = calib_matrix\H(:,3) / norm(calib_matrix\H(:,2)); 
R_w2c = [r1 r2 cross(r1,r2)];


T_wr = [(R_w2c)' -(R_w2c)'*t_w2c; 0 0 0 1]* cam2robot; 
pos = T_wr(1:3,end);
quat = Rot2Quat(T_wr(1:3,1:3));


end

function q = Rot2Quat(R)
    trace = R(1,1) + R(2,2) + R(3,3);
   
if trace>0
    s = (1/2)/sqrt(trace+1);
    w = (1/4)/s;
    x = (R(3, 2) - R(2, 3))*s;
    y = (R(1, 3) - R(3, 1))*s;
    z = (R(2, 1) - R(1, 2))*s;
elseif R(1, 1) > R(2, 2) && R(1, 1) > R(3, 3)
    s = 2*sqrt(1 + R(1, 1) - R(2, 2) - R(3, 3));
    w = (R(3, 2) - R(2, 3))/s;
    x = s/4;
    y = (R(1, 2) + R(2, 1))/s;
    z = (R(1, 3) + R(3, 1))/s;
elseif R(2, 2) > R(3, 3)
    s = 2*sqrt(1 + R(2, 2) - R(1, 1) - R(3, 3));
    w = (R(1, 3) - R(3, 1) ) / s;
    x = (R(1, 2) + R(2, 1) ) / s;
    y =  s/4;
    z = (R(2, 3) + R(3, 2) ) / s;
else
    s = 2 * sqrt( 1.0 + R(3, 3) - R(1, 1) - R(2, 2) );
    w = (R(2, 1) - R(1, 2) ) / s;
    x = (R(1, 3) + R(3, 1) ) / s;
    y = (R(2, 3) + R(3, 2) ) / s;
    z =  s/4;
end
  
q = [w; x; y; z];
end



