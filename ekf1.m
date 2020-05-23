function [X, Z] = ekf1(sensor, vic, varargin)
% EKF1 Extended Kalman Filter with Vicon velocity as inputs
%
% INPUTS:
%   sensor - struct stored in provided dataset, fields include
%          - is_ready: logical, indicates whether sensor data is valid
%          - t: sensor timestamp
%          - rpy, omg, acc: imu readings
%          - img: uint8, 240x376 grayscale image
%          - id: 1xn ids of detected tags
%          - p0, p1, p2, p3, p4: 2xn pixel position of center and
%                                four corners of detected tags

%   vic    - struct for storing vicon linear velocity in world frame and
%            angular velocity in body frame, fields include
%          - t: vicon timestamp
%          - vel = [vx; vy; vz; wx; wy; wz]

%
% OUTPUTS:
% X - nx1 state of the quadrotor, n should be greater or equal to 6
%     the state should be in the following order
%     [x; y; z; qw; qx; qy; qz; other states you use]
%     we will only take the first 7 rows of X


persistent Myu Sigma t_prev

if isempty(Myu)
   if isempty(sensor.id)
       Myu = zeros(9,1);
       Sigma = zeros(9);
       t_prev = sensor.t;
       Z = zeros(6,1);
       X = zeros(7,1);
       return
   end
   
   [p_m, quat, ~, ~] = estimate_pose(sensor);

    [yaw,roll,pitch] = quat2angle(quat', 'ZXY');
    q_m = [roll;pitch;yaw]; 
    Z = [p_m; q_m];
    
   Myu = [Z; 0; 0; 0];
   quat = angle2quat(Myu(6), Myu(4), Myu(5), 'ZXY');
   X = [Myu(1:3);quat'];
   return
end

dt = vic.t - t_prev;

Q = eye(9); 
R = eye(6);
   
C = zeros(6,9);
C(1:6,1:6) = eye(6);
W = eye(6);

v_m = vic.vel(1:3);
w_m = vic.vel(4:6);

bg = Myu(7:9);
phi = Myu(4);
theta = Myu(5);
G = compute_G(phi, theta);
x_dot = [v_m; G\(w_m - bg); 0;0;0 ];

%% prediction step
F = ekf1_A(Myu,w_m,zeros(9,1))*dt + eye(9);
V = ekf1_U(Myu)*dt;
Myu = Myu + dt*x_dot;
Sigma = F*Sigma*F' + V*Q*V';


Z = zeros(6,1);  %initialize z before measurement is taken
%% Update step (only if a tag is captured)
if  ~isempty(sensor.id)&& sensor.is_ready
    [p_m, quat, ~, ~] = estimate_pose(sensor);
    [yaw,roll,pitch] = quat2angle(quat', 'ZXY');
    q_m = [roll;pitch;yaw]; 
    Z = [p_m; q_m];
    
    K = Sigma*C'/(C*Sigma*C' + W*R*W'); 
    Myu = Myu + K*(Z - C*Myu);
    Sigma = Sigma - K*C*Sigma;  
end

quat = angle2quat(Myu(6), Myu(4), Myu(5), 'ZXY');
X = [Myu(1:3);quat'];
t_prev = vic.t;

end



%% helper
function G = compute_G(phi, theta)
G =  [cos(theta)    0    -cos(phi)*sin(theta);
          0         1                 sin(phi);
      sin(theta)    0    cos(phi)*cos(theta) ];
end


function A = ekf1_A(Myu,omg_m,n)
A = reshape([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,...
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,...
    0.0,1.0./cos(Myu(4,:)).^2.*sin(Myu(4,:)).*(cos(Myu(5,:)).*omg_m(3,:)+n(4,:).*sin(Myu(5,:))+sin(Myu(5,:)).*Myu(7,:)-n(6,:).*cos(Myu(5,:))-sin(Myu(5,:)).*omg_m(1,:)-cos(Myu(5,:)).*Myu(9,:)),...
    0.0,0.0,0.0,0.0,0.0,0.0,cos(Myu(5,:)).*omg_m(3,:)+n(4,:).*sin(Myu(5,:))+sin(Myu(5,:)).*Myu(7,:)-n(6,:).*cos(Myu(5,:))-sin(Myu(5,:)).*omg_m(1,:)-cos(Myu(5,:)).*Myu(9,:),0.0,(1.0/cos(Myu(4,:))).*(n(4,:).*cos(Myu(5,:))+n(6,:).*sin(Myu(5,:))-cos(Myu(5,:)).*omg_m(1,:)-sin(Myu(5,:)).*omg_m(3,:)+cos(Myu(5,:)).*Myu(7,:)+sin(Myu(5,:)).*Myu(9,:)),...
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-cos(Myu(5,:)),0.0,sin(Myu(5,:)).*(1.0/cos(Myu(4,:))),0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-sin(Myu(5,:)),0.0,-cos(Myu(5,:)).*(1.0/cos(Myu(4,:))),0.0,0.0,0.0],[9, 9]);
end


function U = ekf1_U(Myu)
U = reshape([-1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0,0.0,0.0,0.0,0.0,...
    0.0,0.0,0.0,0.0,0.0,-1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-cos(Myu(5,:)),...
    0.0,sin(Myu(5,:)).*(1.0./cos(Myu(4,:))),0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0,0.0,...
    0.0,0.0,0.0,0.0,0.0,0.0,-sin(Myu(5,:)),0.0,-(1.0./cos(Myu(4,:))).*cos(Myu(5,:)),...
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,...
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0],[9, 9]);
end

