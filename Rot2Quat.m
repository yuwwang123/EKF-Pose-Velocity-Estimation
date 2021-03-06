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

