《模型》 《ORBvoc》 《YAML》 《数据集》 《时间戳》

############################################################

              FWS-SLAM 开源数据集

############################################################
EuRoC
/home/dl/FWS-SLAM/bin/stereo_euroc \
/home/dl/FWS-SLAM/Vocabulary/ORBvoc.txt \
/home/dl/FWS-SLAM/Examples/Stereo/EuRoC.yaml \
/home/dl/SLAM_Dataset/V1_03_difficult \
/home/dl/FWS-SLAM/Examples/Stereo/EuRoC_TimeStamps/V103.txt 

/home/dl/FWS-SLAM/bin/mono_euroc \
/home/dl/FWS-SLAM/Vocabulary/ORBvoc.txt \
/home/dl/FWS-SLAM/Examples/Stereo/EuRoC.yaml \
/home/dl/SLAM_Dataset/MH_01_easy \
/home/dl/FWS-SLAM/Examples/Stereo/EuRoC_TimeStamps/MH01.txt 



############################################################

              FWS-SLAM 扑翼数据集

############################################################
/home/dl/FWS-SLAM/bin/mono_euroc_mine \
/home/dl/FWS-SLAM/Vocabulary/ORBvoc.txt \
/home/dl/SLAM_Dataset/videos/calibration/YAML/2026/left4mm720p.yaml  \
/home/dl/SLAM_Dataset/videos/319/left  \
/home/dl/SLAM_Dataset/videos/319/time.txt  



/home/dl/FWS-SLAM/bin/mono_euroc_mine \
/home/dl/FWS-SLAM/Vocabulary/ORBvoc.txt \
/home/dl/SLAM_Dataset/videos/calibration/YAML/2026/left4mm720p.yaml  \
/home/dl/SLAM_Dataset/videos/313/four/left  \
/home/dl/SLAM_Dataset/videos/313/four/time6.txt 

/home/dl/FWS-SLAM/bin/mono_euroc_mine \
/home/dl/FWS-SLAM/Vocabulary/ORBvoc.txt \
/home/dl/SLAM_Dataset/videos/calibration/YAML/2026/left4mm720p.yaml  \
/home/dl/SLAM_Dataset/videos/313/four/left  \
/home/dl/SLAM_Dataset/videos/313/four/time7.txt 

############################################################

                   EVO

############################################################
evo_ape tum /home/dl/SLAM_Dataset/videos/319/time.tum CameraTrajectory.txt --align -s -v --correct_scale --plot
evo_ape tum /home/dl/SLAM_Dataset/videos/313/four/time7.tum CameraTrajectory.txt --align -s -v --correct_scale --plot

evo_traj tum /home/dl/FWS-SLAM/build/CameraTrajectory.txt --ref /home/dl/SLAM_Dataset/videos/319/time.tum --align --correct_scale --plot --plot_mode xyz

evo_traj tum /home/dl/FWS-SLAM/build/CameraTrajectory.txt /home/dl/FWS-SLAM/evo/ORB-SLAM3/time/1/ORBslam3.txt --ref /home/dl/SLAM_Dataset/videos/319/time.tum --align --correct_scale --plot --plot_mode xyz



############################################################

              FWS-SLAM ROS实时运行

############################################################
roscore

rosrun usb_cam usb_cam_node _video_device:=/dev/video2 _image_width:=2560 \
_image_height:=720 _framerate:=60 _pixel_format:=mjpeg

rosrun FWS-SLAM-ROS bin/Mono_Stereo_Left \
/home/dl/FWS-SLAM/Vocabulary/ORBvoc.txt \
/home/dl/SLAM_Dataset/videos/calibration/YAML/2026/left4mm720p.yaml \
/camera/image_raw:=/usb_cam/image_raw \


############################################################

              FWS-SLAM ROS运行bag

############################################################





























