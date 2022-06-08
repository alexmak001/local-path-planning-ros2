import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile

import sys
import os
os.environ['OPENBLAS_NUM_THREADS'] = str(1)
import numpy as np
import datetime
import json
import time
import configparser
import graph_ltpl

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class MotionPlan(Node):

    def __init__(self):
        # Here we have the class constructor
        # call the class constructor
        super().__init__('motion_plan')
        toppath = "/home/triton-ai/motion_plan_pkg/motion_plan_pkg"
        sys.path.append(toppath)


        # init dummy object list
        self.obj_list_dummy = graph_ltpl.testing_tools.src.objectlist_dummy.ObjectlistDummy(dynamic=False,vel_scale=0.3,s0=250.0)
        self.object_list = self.obj_list_dummy.get_objectlist()
        # self.object_list = [{'X': 75, 'Y': 103, 'theta': 0.0, 'type': 'physical', 'id': 1, 'length': 1.0, 'width': 1.0, 'v': 0.0}]

        # state variables
        self.traj_set = {'straight': None}
        self.legacy_state = []
        self.behavior = 'straight'
        self.odom_flag, self.dummy_flag = True, True 

        # publishable path msg objects
        self.pathMsg = Path()

        # create the publisher object (Output to Race Control)
        self.path_pub = self.create_publisher(Path, 'motion_plan', 10)

        # create timer
        self.timer_period = 0.01
        self.timer = self.create_timer(self.timer_period, self.send_path)

        # create subsciber objects
        # self.odom_sub = self.create_subscription()
        # self.object_sub = self.create_subscription()



        # ----------------------------------------------------------------------------------------------------------------------
        # INITIALIZATION AND OFFLINE PART --------------------------------------------------------------------------------------
        # ----------------------------------------------------------------------------------------------------------------------
        
        track_specifier = "ims"
        # define all relevant paths
        path_dict = {'globtraj_input_path': toppath + "/inputs/traj_ltpl_cl/traj_ltpl_cl_" + track_specifier + ".csv", #replace with path of wherever Team4 stores race line
                    'graph_store_path': toppath + "/inputs/stored_graph.pckl", #new path to store graph of offline graph (?)
                    'ltpl_offline_param_path': toppath + "/params/ltpl_config_offline.ini", #params for generating offline graph (all possible nodes, splines, etc.)
                    'ltpl_online_param_path': toppath + "/params/ltpl_config_online.ini", #params for generating online graph (all possible nodes, splines, etc.)
                    'log_path': toppath + "/logs/graph_ltpl/",  #new path to store local path csv's and messages that are regularly updated and published? and offline graphs
                    'graph_log_id': datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S") #date time for organizing logs
                    }


        # intialize graph_ltpl-class
        self.ltpl_obj = graph_ltpl.Graph_LTPL.Graph_LTPL(path_dict=path_dict,
                                                    visual_mode=True, #disable when actually running on car
                                                    log_to_file=True)
        
        # calculate offline graph
        self.ltpl_obj.graph_init()

        # set start pose based on first point in provided reference-line
        refline = graph_ltpl.imp_global_traj.src.import_globtraj_csv.\
            import_globtraj_csv(import_path=path_dict['globtraj_input_path'])[0]
        self.pos_est = refline[0, :] #position along race line
        self.heading_est = np.arctan2(np.diff(refline[0:2, 1]), np.diff(refline[0:2, 0])) - np.pi / 2 #where is car facing
        self.vel_est = 0.0

        # puts car on graph based on start pose
        self.ltpl_obj.set_startpos(pos_est=self.pos_est,
                            heading_est=self.heading_est)

        # self.traj_set = self.ltpl_obj.calc_vel_profile(pos_est=self.pos_est,
        #                                         vel_est=self.vel_est)[0]


        self.tic = time.time()

        # self.zone_example = {'sample_zone': [[64, 64, 64, 64, 64, 64, 64, 65, 65, 65, 65, 65, 65, 65, 66, 66, 66, 66, 66, 66, 66], #blocked layers
        #                                 [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6], #blocked nodes, pair with blocked layers
        #                                 np.array([[-20.54, 227.56], [23.80, 186.64]]), #left bound region
        #                                 np.array([[-23.80, 224.06], [20.17, 183.60]])]} #right bound region

    def send_path(self):
        """
        Input: 2D numpy array
        Publish: Only X and Y columns of the output
        """
        # Comment this out when not using simulator
        # if not self.odom_flag or not self.dummy_flag:
        #     self.get_logger().info("Behavior: {%s}" % (self.behavior))
        #     return

        self.behavior_planning()
        self.calc_local_path()
        
        
        # for sel_action in action_state:
        for sel_action in self.action_state:  # try to force 'right', else try next in list
            if sel_action in self.traj_set.keys():
                self.behavior = sel_action
                break

        # Print what behavior is choosen
        self.get_logger().info("Action State: {%s}" % (self.action_state))
        self.get_logger().info("Behavior: {%s}" % (self.behavior))

        self.pathMsg.header.frame_id = 'map'

        self.get_logger().info("Trag Set: {%s}" % list(self.traj_set.keys()))

        for row in self.traj_set[self.behavior][0]:
            pose_msg = PoseStamped()
            pose_msg.pose.position.x = row[1]
            pose_msg.pose.position.y = row[2]

            self.pathMsg.poses.append(pose_msg)

        self.path_pub.publish(self.pathMsg)
        self.pathMsg = Path()

    def calc_local_path(self):
        self.ltpl_obj.calc_paths(prev_action_id=self.behavior,
                            object_list=self.object_list)
                            # blocked_zones=zone_example)

        if self.traj_set[self.behavior] is not None:
            self.pos_est, self.vel_est = graph_ltpl.testing_tools.src.vdc_dummy.\
                vdc_dummy(pos_est=self.pos_est,
                            last_s_course=(self.traj_set[self.behavior][0][:, 0]),
                            last_path=(self.traj_set[self.behavior][0][:, 1:3]),
                            last_vel_course=(self.traj_set[self.behavior][0][:, 5]),
                            iter_time=time.time() - self.tic)

        self.traj_set = self.ltpl_obj.calc_vel_profile(pos_est=self.pos_est,
                                                vel_est=self.vel_est)[0]

        self.tic = time.time()

    def sensor_fusion(self, msg):
        self.dummy_flag = True
        self.object_list = msg 

    def odom_fusion(self, msg):
        self.odom_flag = True
        self.odom_msg = msg

    def race_line(self, msg):
        self.race_line = msg


    def behavior_planning(self):
        """
        Team 1 Works Here: behavior decision making

        Basic Logic: Take the closest opponent's distance and pick behavior.
        - check direct distance to the car to make sure there are enough room
        - check orientation to look for overtaking opportunity
        - cases:
            1. no opponents: straight
            2. more than 1 opponents close to the car: follow
            3. if one side of the closest opponent is open: overtake
        """

        """Distance Constant"""
        opponents_distance_offset = 20 # Modify to fit actual value of car length
        opponents_horizontal_angle_offset = 20 # Modify to fit actual value of car width

        """State Variable"""
        find_dist = lambda x1, y1, x2, y2: np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        # Populate with pathMsg
        self_X, self_Y = 0, 0
        car_distance_wall = [0, 0]
        self_leftwall_bound, self_rightwall_bound = 0, 0
        self_velocity = 0

        # Populate with sensorMsg
        opponent_count = 0
        nearest_opponet_index = 0
        orientation_angle = [0]*len(self.object_list)
        # opponent_relative_direction = [0]*len(self.object_list) # ahaed of behind or align
        opponent_linear_distances = [float('inf')]*len(self.object_list)
        # opponent_dimensions = [[0, 0]]*len(self.object_list)
        opponent_velocity = [0]*len(self.object_list)

        opponent_horizontal_distance = [0]*len(self.object_list)
        relative_velocity = [0]*len(self.object_list)
        # acceleration = [0]*len(self.object_list) 

        action_state = [] # order the options in order

        # Utility variables
        temp_min_distance = float('inf')

        for index, obj in enumerate(self.object_list):
            # print(obj)
            opponent_count+=1
            opponent_linear_distances[index] = np.sqrt((obj['X'] - self_X)**2 + (obj['Y'] - self_Y)**2)
            if (obj['Y'] - self_Y) > 10:
                # ahead
                opponent_relative_direction = 0
            elif (obj['Y'] - self_Y) < -10:
                # behind
                opponent_relative_direction = 1
            else:
                # align
                opponent_relative_direction = 0

            orientation_angle[index] = obj['theta']
            opponent_velocity[index] = obj['v'] if 'v' in obj else obj['v_x']
            relative_velocity[index] = opponent_velocity[index] - self_velocity
            opponent_horizontal_distance[index] = self_X-obj['X']

            # find the cloest opponent
            if temp_min_distance > opponent_linear_distances[-1]:
                temp_min_distance = opponent_linear_distances[-1]
                nearest_opponet_index = index

        # no opponent detected
        if opponent_count == 0:
            action_state.append('straight')

        # opponent detected
        else:
            # check distance
            if opponent_linear_distances[nearest_opponet_index] > 20:
                action_state.append('follow')
            else:
                if relative_velocity <= 0:
                    if abs(self_X - self_leftwall_bound) > abs(self_rightwall_bound - self_X) :
                        action_state.append('left')
                    else:
                        action_state.append('right')
                    action_state.append(self.legacy_state[0]) # taking the bast option from the previous decision, ex: if we were overtaking, we probably still want to overtake
                    action_state.append('follow')
                else:
                    action_state.append('follow')

            self.legacy_state = action_state
            action_state.append('straight')
        # print("Action State: ", action_state)
        # print("Traj_Set", self.traj_set)

        self.action_state = action_state
        return


def main(args=None):
    # initialize the ROS communication
    rclpy.init(args=args)
    # declare the node constructor
    motion_plan = MotionPlan()
    # pause the program execution, waits for a request to kill the node (ctrl+c)
    rclpy.spin(motion_plan)
    # Explicity destroy the node
    motion_plan.destroy_node()
    # shutdown the ROS communication
    rclpy.shutdown()

if __name__ == '__main__':
    main()
