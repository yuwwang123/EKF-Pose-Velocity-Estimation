function [X, Z] = ekf2(sensor, varargin)
% EKF2 Extended Kalman Filter with IMU as inputs
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

% OUTPUTS:
% X - nx1 state of the quadrotor, n should be greater or equal to 9
%     the state should be in the following order
%     [x; y; z; vx; vy; vz; qw; qx; qy; qz; other states you use]
%     we will only take the first 10 rows of X

persistent Myu Sigma t_prev

if isempty(Myu)
   if isempty(sensor.id)|| ~sensor.is_ready
       Myu = zeros(15,1);
       Sigma = zeros(15);
       t_prev = sensor.t;
       Z = zeros(9,1);
       X = zeros(10,1);
       return
   end
   
   [p_m, quat, ~, ~] = estimate_pose(sensor);

    [yaw,roll,pitch] = quat2euler(quat', 'ZXY');
    q_m = [roll;pitch;yaw]; 
    Z = [p_m; q_m; 0;0;0];
    
   Myu = [Z; zeros(6,1)];
   
   quat = angle2quat(Myu(6), Myu(4), Myu(5), 'ZXY');
   X = [Myu(1:3);Myu(7:9);quat'];
   return
end

dt = sensor.t - t_prev;

Q = eye(15); 
R = eye(9);   
C = zeros(9,15);
C(1:9,1:9) = eye(9);
W = eye(9);

w_m = sensor.omg;
acc_m = sensor.acc;
g = [0;0;9.81];

bg = Myu(10:12);
ba = Myu(13:15);
phi = Myu(4);
theta = Myu(5);
sai = Myu(6);
G = compute_G(phi, theta);
Rot = euler2Rot(phi, theta, sai);

%% prediction step
f = [ Myu(7:9);G\(w_m - bg); -g + Rot*(acc_m - ba); zeros(6,1)];

F = ekf2_A(Myu,w_m,acc_m,zeros(15,1))*dt + eye(15);
V = ekf2_U(Myu)*dt;
Myu = Myu + dt*f;
Sigma = F*Sigma*F' + V*Q*V';

%% Update step
Z = zeros(9,1);  %initialize z before measurement is taken
if  ~isempty(sensor.id)&& sensor.is_ready
    [p_m, quat, ~, ~] = estimate_pose(sensor);

    [yaw,roll,pitch] = quat2euler(quat');
    q_m = [roll;pitch;yaw]; 
    
    [vel_m, ~] = estimate_vel(sensor);
    
    if isempty(vel_m)
        vel_m = Myu(7:9);
    end
    
    Z = [p_m; q_m; vel_m];
    
    K = Sigma*C'/(C*Sigma*C' + W*R*W'); 
    Myu = Myu + K*(Z - C*Myu);
    Sigma = Sigma - K*C*Sigma;  
end

quat = angle2quat(Myu(6), Myu(4), Myu(5), 'ZXY');
X = [Myu(1:3);Myu(7:9);quat'];
t_prev = sensor.t;

end




%% helper
function G = compute_G(phi, theta)
G =  [cos(theta)    0    -cos(phi)*sin(theta);
          0         1                 sin(phi);
      sin(theta)    0    cos(phi)*cos(theta) ];
end

function R = euler2Rot(phi, theta, sai)
R = [cos(sai)*cos(theta)-sin(phi)*sin(sai)*sin(theta),       -cos(phi)*sin(sai),       cos(sai)*sin(theta) + cos(theta)*sin(phi)*sin(sai) ; 
          cos(theta)*sin(phi)+ cos(sai)*sin(phi)*sin(theta) ,    cos(phi)*cos(sai),    sin(sai)*sin(theta) - cos(sai)*cos(theta)*sin(phi) ;
       -cos(phi)*sin(theta) ,                                     sin(phi) ,                                             cos(phi)*cos(theta)];
end


function  [yaw,roll,pitch] = quat2euler(quat)  

quat_n = quat./(norm(quat,2)* ones(1,4));

r11 = -2.*(quat_n(:,2).*quat_n(:,3) - quat_n(:,1).*quat_n(:,4));
r12 = quat_n(:,1).^2 - quat_n(:,2).^2 + quat_n(:,3).^2 - quat_n(:,4).^2;
r21 = 2.*(quat_n(:,3).*quat_n(:,4) + quat_n(:,1).*quat_n(:,2));
r31 =  -2.*(quat_n(:,2).*quat_n(:,4) - quat_n(:,1).*quat_n(:,3));
r32 = quat_n(:,1).^2 - quat_n(:,2).^2 - quat_n(:,3).^2 + quat_n(:,4).^2;

yaw = atan2( r11, r12 );
roll = asin( r21 );
pitch = atan2( r31, r32 );
end

function A = ekf2_A(Myu,w_m,acc_m,n)

term_1 = sin(Myu(5,:)).*cos(Myu(6,:));
term_2 = cos(Myu(5,:)).*sin(Myu(4,:)).* sin(Myu(6,:));
term_3 = sin(Myu(5,:)).* sin(Myu(6,:));
term_4 = cos(Myu(5,:)).*sin(Myu(4,:)).*cos(Myu(6,:));
term_5 = term_3-term_4;
term_6 = cos(Myu(5,:)).* sin(Myu(6,:));
term_7 = sin(Myu(5,:)).*sin(Myu(4,:)).*cos(Myu(6,:));
term_8 = term_6+term_7;
term_9 = sin(Myu(5,:)).*sin(Myu(4,:)).* sin(Myu(6,:));
term_10 = term_1+term_2;
A = reshape([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0./cos(Myu(4,:)).^2.*sin(Myu(4,:)).*((cos(Myu(5,:)).*w_m(3,:))+n(4,:).*sin(Myu(5,:))+(sin(Myu(5,:)).*Myu(10,:))-n(6,:).*cos(Myu(5,:))-sin(Myu(5,:)).*w_m(1,:)-cos(Myu(5,:)).*Myu(12,:)),-sin(Myu(4,:)).* sin(Myu(6,:)).*(-acc_m(2,:)+n(8,:)+Myu(14,:))+sin(Myu(5,:)).*cos(Myu(4,:)).* sin(Myu(6,:)).*(-acc_m(1,:)+n(7,:)+Myu(13,:))-cos(Myu(5,:)).*cos(Myu(4,:)).* sin(Myu(6,:)).*(-acc_m(3,:)+n(9,:)+Myu(15,:)),sin(Myu(4,:)).*cos(Myu(6,:)).*(-acc_m(2,:)+n(8,:)+Myu(14,:))-sin(Myu(5,:)).*cos(Myu(4,:)).*(-acc_m(1,:)+n(7,:)+Myu(13,:)).*cos(Myu(6,:))+cos(Myu(5,:)).*cos(Myu(4,:)).*cos(Myu(6,:)).*(-acc_m(3,:)+n(9,:)+Myu(15,:)),-cos(Myu(4,:)).*(-acc_m(2,:)+n(8,:)+Myu(14,:))-sin(Myu(5,:)).*sin(Myu(4,:)).*(-acc_m(1,:)+n(7,:)+Myu(13,:))+cos(Myu(5,:)).*sin(Myu(4,:)).*(-acc_m(3,:)+n(9,:)+Myu(15,:)),0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,(cos(Myu(5,:)).*w_m(3,:))+n(4,:).*sin(Myu(5,:))+(sin(Myu(5,:)).*Myu(10,:))-n(6,:).*cos(Myu(5,:))-sin(Myu(5,:)).*w_m(1,:)-cos(Myu(5,:)).*Myu(12,:),0.0,(1.0./cos(Myu(4,:))).*(n(4,:).*cos(Myu(5,:))+n(6,:).*sin(Myu(5,:))-cos(Myu(5,:)).*w_m(1,:)-sin(Myu(5,:)).*w_m(3,:)+cos(Myu(5,:)).*Myu(10,:)+sin(Myu(5,:)).*Myu(12,:)),(-acc_m(1,:)+n(7,:)+Myu(13,:)).*term_10-(-acc_m(3,:)+n(9,:)+Myu(15,:)).*((cos(Myu(5,:)).*cos(Myu(6,:)))-sin(Myu(5,:)).*sin(Myu(4,:)).* sin(Myu(6,:))),(-acc_m(1,:)+n(7,:)+Myu(13,:)).*term_5-(-acc_m(3,:)+n(9,:)+Myu(15,:)).*term_8,cos(Myu(5,:)).*cos(Myu(4,:)).*(-acc_m(1,:)+n(7,:)+Myu(13,:))+sin(Myu(5,:)).*cos(Myu(4,:)).*(-acc_m(3,:)+n(9,:)+Myu(15,:)),0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,(-acc_m(3,:)+n(9,:)+Myu(15,:)).*term_5+(-acc_m(1,:)+n(7,:)+Myu(13,:)).*term_8+cos(Myu(4,:)).*cos(Myu(6,:)).*(-acc_m(2,:)+n(8,:)+Myu(14,:)),-(-acc_m(3,:)+n(9,:)+Myu(15,:)).*term_10-(-acc_m(1,:)+n(7,:)+Myu(13,:)).*((cos(Myu(5,:)).*cos(Myu(6,:)))-term_9)+cos(Myu(4,:)).* sin(Myu(6,:)).*(-acc_m(2,:)+n(8,:)+Myu(14,:)),0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-cos(Myu(5,:)),0.0,sin(Myu(5,:)).*(1.0./cos(Myu(4,:))),0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-sin(Myu(5,:)),0.0,-cos(Myu(5,:)).*(1.0./cos(Myu(4,:))),0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-(cos(Myu(5,:)).*cos(Myu(6,:)))+term_9,-term_6-term_7,sin(Myu(5,:)).*cos(Myu(4,:)),0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,cos(Myu(4,:)).* sin(Myu(6,:)),-cos(Myu(4,:)).*cos(Myu(6,:)),-sin(Myu(4,:)),0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-term_1-term_2,-term_3+term_4,-cos(Myu(5,:)).*cos(Myu(4,:)),0.0,0.0,0.0,0.0,0.0,0.0],[15, 15]);
end

function U = ekf2_U(Myu)
U = reshape([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,- cos(Myu(5,:)),0.0,sin(Myu(5,:)).*(1.0./cos(Myu(4,:))),0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-sin(Myu(5,:)),0.0,-(1.0./cos(Myu(4,:))).* cos(Myu(5,:)),0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,- cos(Myu(5,:)).*cos(Myu(6,:))+sin(Myu(5,:)).*sin(Myu(6,:)).*sin(Myu(4,:)),- cos(Myu(5,:)).*sin(Myu(6,:))-sin(Myu(5,:)).*cos(Myu(6,:)).*sin(Myu(4,:)),sin(Myu(5,:)).*cos(Myu(4,:)),0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,cos(Myu(4,:)).*sin(Myu(6,:)),-cos(Myu(4,:)).*cos(Myu(6,:)),-sin(Myu(4,:)),0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-sin(Myu(5,:)).*cos(Myu(6,:))- cos(Myu(5,:)).*sin(Myu(6,:)).*sin(Myu(4,:)),-sin(Myu(5,:)).*sin(Myu(6,:))+ cos(Myu(5,:)).*cos(Myu(6,:)).*sin(Myu(4,:)),-cos(Myu(4,:)).* cos(Myu(5,:)),0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0],[15, 15]);
end
