#!/usr/bin/env python
# coding=utf-8

from robotiq_85_msgs.msg import GripperCmd,GripperStat
from sensor_msgs.msg import JointState
import time
import rospy

class Robotiq(object):
    def __init__(self):
        self._gripper_state_topic = '/gripper/stat'
        self._gripper_joint_states_topic = '/gripper/joint_states'
        self._gripper_control_topic = '/gripper/cmd'

        self.gripper_state_sub = rospy.Subscriber(self._gripper_state_topic,GripperStat,self.update_gripper_state_handle)
        self.gripper_joint_states_sub = rospy.Subscriber(self._gripper_joint_states_topic,JointState,
                                                        self.update_gripper_joint_states_handle)
        self.gripper_control_pub = rospy.Publisher(self._gripper_control_topic, GripperCmd, queue_size=10)

        self._gripper_state = None
        self._gripper_joint_states = None
        self._r = rospy.Rate(30)
        time.sleep(1)

    def update_gripper_state_handle(self,data):
        self._gripper_state = data

    def update_gripper_joint_states_handle(self,data):
        self._gripper_joint_states =data


    def is_moving(self):
        return self._gripper_state.is_moving

    def obj_dectected(self):
        return self._gripper_state.obj_detected

    def position(self):
        return self._gripper_state.position

    def current(self):
        return self._gripper_state.current

    def set_gripper_property(self, pos=0, speed=0, force=0):
        control_value = GripperCmd()
        control_value.position = pos
        control_value.speed = speed
        control_value.force = force
        return control_value

    # set the position/velocity/torque of gripper
    def set_gripper_position(self,value=0):
        target = self.set_gripper_property(pos=value)
        self.gripper_control_pub.publish(target)
        self._r.sleep()
    def set_gripper_velocity(self,value=0):
        target = self.set_gripper_property(speed=value)
        self.gripper_control_pub.publish(target)
        self._r.sleep()
    def set_gripper_torque(self,value=0):
        target = self.set_gripper_property(force=value)
        self.gripper_control_pub.publish(target)
        self._r.sleep()

    def open_gripper(self):
        target = self.set_gripper_property(pos=0.085, speed=0.2, force=10)
        self.gripper_control_pub.publish(target)
        self._r.sleep()

    def close_gripper(self, speed, force):
        target = self.set_gripper_property(pos=0.0, speed=speed, force=force)
        self.gripper_control_pub.publish(target)
        self._r.sleep()
