
init_script;
load('data/studentdata9.mat');
n = size(data,2);
X = zeros(10,n);
m = size(data,2);
dtime=zeros(m,1);
for j = 1:m
    dtime(j) = data(j).t;
end
for i = 1:n
        [X(:,i),~] = ekf2(data(i));
end

vtime = time;
vpos = vicon(1:3,:);
vrot = vicon(4:6,:);
vvel = vicon(7:9,:);
vquat = (angle2quat(vrot(3,:), vrot(1,:), vrot(2,:), 'ZXY'))';

figure
title('X');
plot(dtime,X(1,:),'g',vtime,vpos(1,:),'r');
figure
title('Y');
plot(dtime,X(2,:),'g',vtime,vpos(2,:),'r');
figure
title('Z');
plot(dtime,X(3,:),'g',vtime,vpos(3,:),'r');
figure
title('Vx');
plot(dtime,X(4,:),'g',vtime,vvel(1,:),'r');
figure
title('Vy');
plot(dtime,X(5,:),'g',vtime,vvel(2,:),'r');
figure
title('Vz');
plot(dtime,X(6,:),'g',vtime,vvel(3,:),'r');
figure
title('Qw');
plot(dtime,X(7,:),'g',vtime,vquat(1,:),'r');
figure
title('Qx');
plot(dtime,X(8,:),'g',vtime,vquat(2,:),'r');
figure
title('Qy');
plot(dtime,X(9,:),'g',vtime,vquat(3,:),'r');
figure
title('Qz');
plot(dtime,X(10,:),'g',vtime,vquat(4,:),'r');
