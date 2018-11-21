# Instructions for parallel ROS + Gazebo executions

1) Modify the `/usr/share/gazebo/setup.sh` and replace

    `export GAZEBO_MASTER_URI=http://localhost:11345`

  with

    `export GAZEBO_MASTER_URI=${GAZEBO_MASTER_URI:-"http://localhost:11345"}`

2) In the a new shell, set a new port for the rosmaster:

    `export ROS_MASTER_URI=http://localhost:11315`

3) You also need to update a new port for gazebo:

    `export GAZEBO_MASTER_URI=http://localhost:11355`

4) Now, if you launch a new process from this shell, you'll have a complete separate environment from what's already running on your machine.

### Additional resources

1) [RosAnswers topic](https://answers.ros.org/question/193062/how-to-run-multiple-independent-gazebo-instances-on-the-same-machine/?answer=193397#post-id-193397)
2) [RosGazeboParallelSimulations example code](https://github.com/alidemir1/RosGazeboParallelSimulations)
