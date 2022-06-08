from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='motion_plan_pkg',
            executable='motion_plan',
            output='screen'),
    ])
