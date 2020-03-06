
init_script;
load('data/studentdata9.mat');
%load('data/empty.mat');
n = size(vicon,2);
X = zeros(7,n);
m = size(data,2);
dtime=zeros(m,1);
for j = 1:m
    dtime(j) = data(j).t;
end
for i = 1:n
    value1 = time(i);
    value2 = vicon(7:12,i);
    vic = struct('t',value1,'vel',value2);
    if ismember(time(i),dtime)
        [~,idx]=ismember(time(i),dtime);
        [X(:,i),~] = ekf1(data(idx),vic);
    else
%        [X(:,i),~] = ekf1(empty,vic);
    end
end

vtime = time;
vpos = vicon(1:3,:);
vrot = vicon(4:6,:);
vquat = (angle2quat(vrot(3,:), vrot(1,:), vrot(2,:), 'ZXY'))';

figure
title('X');
plot(vtime,X(1,:),'g',vtime,vpos(1,:),'r');
figure
title('Y');
plot(vtime,X(2,:),'g',vtime,vpos(2,:),'r');
figure
title('Z');
plot(vtime,X(3,:),'g',vtime,vpos(3,:),'r');
figure
title('Qw');
plot(vtime,X(4,:),'g',vtime,vquat(1,:),'r');
figure
title('Qx');
plot(vtime,X(5,:),'g',vtime,vquat(2,:),'r');
figure
title('Qy');
plot(vtime,X(6,:),'g',vtime,vquat(3,:),'r');
figure
title('Qz');
plot(vtime,X(7,:),'g',vtime,vquat(4,:),'r');
