# EKF-Pose-Velocity-Estimation
Implementation of perception stack of a camera/IMU based Crazyflie micro-quadcoptor:

1. AprilTag based perception: Pose Estimation with Homography
2. Velocity Estimation with Optical Flow; RANSAC is applied to reject outliers in feature matching.
3. Sensor Fusion: Extended Kalman Filter (EKF) is used to fuse camera and IMU data to realize full 3d odometry estimation.
