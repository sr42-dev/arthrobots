#!/usr/bin/bash
find ~/catkin_ws/src/arthrobots_git/quadruped/src/ddpg/ -type f -iname "*.py" -exec chmod +x {} \;
find ~/catkin_ws/src/arthrobots_git/quadruped/src/dreamerv3/ -type f -iname "*.py" -exec chmod +x {} \;
find ~/catkin_ws/src/arthrobots_git/quadruped/src/qlearning/ -type f -iname "*.py" -exec chmod +x {} \;
find ~/catkin_ws/src/arthrobots_git/quadruped/src/hardware/ -type f -iname "*.py" -exec chmod +x {} \;
find ~/catkin_ws/src/arthrobots_git/quadruped/src/dqn/ -type f -iname "*.py" -exec chmod +x {} \;
