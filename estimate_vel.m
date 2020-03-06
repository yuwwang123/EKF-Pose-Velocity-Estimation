function [vel, omg] = estimate_vel(sensor, varargin)

K =  [311.0520 0 201.8724; 0 311.3885 113.6210; 0 0 1];

persistent t_prev dt_prev last_image point_tracker corners_prev count 

[pos,q, R_w2c,t_w2c] = estimate_pose(sensor);
if isempty(sensor.id)||isempty(t_w2c) || isempty(R_w2c) 
    vel = [];
    omg = [];
    return
end

% Initialization code
if isempty(t_prev)
    last_image = sensor.img;
    point_tracker = vision.PointTracker('MaxBidirectionalError',0.005);
    corners_prev = detectFASTFeatures(last_image);
    corners_prev = corners_prev.selectStrongest(150);
    corners_prev = corners_prev.Location;
    initialize(point_tracker,corners_prev,last_image);
    vel = [0;0;0];
    omg = [0;0;0];
    t_prev = sensor.t;
    dt_prev = sensor.t - 0;
    count = 1;
    return
end

[corners,valid_id] = step(point_tracker,sensor.img);

corners_valid = corners(valid_id,:);
corners_prev_valid = corners_prev(valid_id,:);


% compute p_dot
n = size(corners_valid, 1);
p_prev = K \ [corners_prev_valid'; ones(1,n)];
p_current = K \ [corners_valid'; ones(1,n)];

% depths
Z = ((R_w2c(:,3)'*t_w2c)./(R_w2c(:,3)'*p_current))';

% low-pass filter on dt
alpha = 0.2; 
dt = alpha*(sensor.t-t_prev)+(1-alpha)*dt_prev; 

p_dot = (p_current(1:2,:) - p_prev(1:2,:))/dt;


% RANSAC 

max_num = 0;
best_id = [];

if n<6
   vel = [];
   omg = [];
   return
end
for i = 1:150
    rand_id = randperm(n,6);
    sample_pdot = p_dot(:,rand_id);
    sample_pdot = sample_pdot(:);  % stack into 6x1 vector
    x_sample = p_current(1,rand_id);
    y_sample = p_current(2,rand_id);
    Z_sample = transpose(Z(rand_id));
    
    F = [f1(x_sample(1), y_sample(1), Z_sample(1));
        f2(x_sample(1), y_sample(1), Z_sample(1));
        f1(x_sample(2), y_sample(2), Z_sample(2));
        f2(x_sample(2), y_sample(2), Z_sample(2));
        f1(x_sample(3), y_sample(3), Z_sample(3));
        f2(x_sample(3), y_sample(3), Z_sample(3))
        f1(x_sample(4), y_sample(4), Z_sample(4));
        f2(x_sample(4), y_sample(4), Z_sample(4));
        f1(x_sample(5), y_sample(5), Z_sample(5));
        f2(x_sample(5), y_sample(5), Z_sample(5));
        f1(x_sample(6), y_sample(6), Z_sample(6));
        f2(x_sample(6), y_sample(6), Z_sample(6))];
    
    Twist_c = F \ sample_pdot;    % as the model
    
    pdot_estimate = zeros(size(p_dot,1),size(p_dot,2));
    for j= 1:size(p_dot,2)
        x = p_current(1,j);
        y = p_current(2,j);
        z = transpose(Z(j));
        pdot_estimate(:,j) = [f1(x, y, z); f2(x, y, z)] * Twist_c;       
    end
     
    error = pdot_estimate - p_dot;
    error_array = sqrt(error(1,:).^2 + error(2,:).^2); 
    inliers_id = find(error_array < 0.05);
    if length(inliers_id)> max_num
        max_num = length(inliers_id);
        best_id = inliers_id;
    end
    if max_num == size(p_dot,2)
        break;
    end
end

% solve for Twist using the best model
% first construct the large F matrix, then find the least square solution
% by inverting F
x_best = p_current(1,best_id);
y_best = p_current(2,best_id);
Z_best = transpose(Z(best_id));

F_best = zeros(2*length(best_id),6);
for i=1:length(best_id)
    F_best(2*i-1,:) = f1(x_best(i), y_best(i), Z_best(i));
    F_best(2*i,:) = f2(x_best(i), y_best(i), Z_best(i));
end

best_pdot = p_dot(:,best_id);
best_pdot = best_pdot(:);

Twist_c = F_best \ best_pdot;    % least square solution

T_cr = [ 0.7071   -0.7071   -0.0000    0.0283;
   -0.7071   -0.7071   -0.0000   -0.0283;
         0    0.0000   -1.0000    -0.03;
         0     0         0        1   ];

omg = R_w2c'*Twist_c(4:6);
vel = R_w2c'*(Twist_c(1:3) - cross(T_cr(1:3,4),Twist_c(4:6)));     
     
t_prev = sensor.t;
dt_prev = dt;
last_image = sensor.img;
count = count + 1;

    point_tracker = vision.PointTracker('MaxBidirectionalError',0.005);
    corners_prev = detectFASTFeatures(last_image);
    corners_prev = corners_prev.selectStrongest(150);
    corners_prev = corners_prev.Location;
    initialize(point_tracker,corners_prev,last_image);
end

function out = f1(x,y,Z)
out = [ -1./Z, 0, x./Z,  x.*y, -(1+x.^2),  y ];
end

function out = f2(x,y,Z)
out = [ 0, -1./Z, y./Z, (1+y.^2), -x.*y,  -x ];
end