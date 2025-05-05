#!/usr/bin/env python3
"""
Sentry Movement Node for iRobot Create3

Build and Run:
    cd ~/ros2_ws
    colcon build --packages-select create3_control
    source install/setup.bash
    ros2 run create3_control sentry_movement --ros-args -p num_cycles:=3
"""
# File: /home/glitchy/ros2_ws/src/create3_control/create3_control/sentry_movement.py
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from irobot_create_msgs.action import Dock, Undock
import math

from math import radians as d2r

# Convert feet to meters
FEET_TO_METER = 0.3048

class SentryMovement(Node):
    def __init__(self):
        super().__init__('sentry_movement')

        self.declare_parameter('num_cycles', 4)
        self.num_cycles = self.get_parameter('num_cycles').value

        self.cmd_vel_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        self.odom_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.pub = self.create_publisher(Twist, '/cmd_vel', self.cmd_vel_qos)
        self.create_subscription(Odometry, '/odom', self.odom_cb, self.odom_qos)

        self.undock_client = ActionClient(self, Undock, 'undock')
        self.dock_client = ActionClient(self, Dock, 'dock')

        self.get_logger().info("Waiting for `/undock` and `/dock` action servers...")
        self.undock_client.wait_for_server()
        self.dock_client.wait_for_server()

        self.state = 'IDLE'
        self.initial_position = None
        self.initial_yaw = None
        self.move_heading = None
        self.current_position = None
        self.current_yaw = None
        self.cycles_completed = 0
        self.current_step = 0

        self.linear_speed = 0.306
        self.angular_speed = 1.0
        self.k_heading = 2.0

        self.move_distances = [
            20.0, 13.5, 7.0, 20.5, 9.0
        ]
        self.turn_angles = [
            (45.0, False),
            (45.0, False),
            (90.0, True),
            (90.0, True)
        ]

        self.undock()
        self.state = 'MOVING'
        self.create_timer(0.1, self.control_loop)

    def undock(self):
        self.get_logger().info('Sending undock command...')
        goal = Undock.Goal()
        send_goal = self.undock_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal)
        handle = send_goal.result()
        if not handle.accepted:
            self.get_logger().error('Undock rejected')
            return False
        res = handle.get_result_async()
        rclpy.spin_until_future_complete(self, res)
        self.get_logger().info('Undocked successfully')
        return True

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.current_position = p
        q = msg.pose.pose.orientation
        num = 2.0 * (q.w * q.z + q.x * q.y)
        den = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        self.current_yaw = math.atan2(num, den)
        if self.state == 'MOVING' and self.initial_position is None:
            self.initial_position = p
            self.move_heading = self.current_yaw

    def control_loop(self):
        if self.state not in ['MOVING', 'TURNING']:
            return
        if self.current_position is None or self.current_yaw is None:
            return
        if self.current_step >= len(self.move_distances):
            self.cycles_completed += 1
            if self.cycles_completed >= self.num_cycles:
                self.get_logger().info('Patrol complete, docking...')
                self.state = 'DOCKING'
                self.start_docking()
                return
            else:
                self.get_logger().info(f'Cycle {self.cycles_completed} complete, restarting path')
                self.current_step = 0
                self.state = 'MOVING'
                self.initial_position = None
                self.initial_yaw = None
                self.move_heading = None
                return

        cmd = Twist()
        feet = self.move_distances[self.current_step]
        meters_to_move = feet * FEET_TO_METER

        if self.state == 'MOVING':
            dx = self.current_position.x - self.initial_position.x
            dy = self.current_position.y - self.initial_position.y
            distance_moved = math.hypot(dx, dy)

            if distance_moved < meters_to_move:
                cmd.linear.x = self.linear_speed
                err = (self.current_yaw - self.move_heading + math.pi) % (2*math.pi) - math.pi
                cmd.angular.z = -self.k_heading * err
            else:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.state = 'TURNING'
                self.initial_yaw = None
                self.get_logger().info(f'Move {self.current_step} complete, starting turn')

        elif self.state == 'TURNING':
            if self.initial_yaw is None:
                self.initial_yaw = self.current_yaw
                self.get_logger().info(f"Starting turn {self.current_step}")
                return

            if self.current_step < len(self.turn_angles):
                angle_deg, ccw = self.turn_angles[self.current_step]
                target_rad = d2r(angle_deg)
                if not ccw:
                    target_rad = -target_rad

                angle_moved = (self.current_yaw - self.initial_yaw + math.pi) % (2*math.pi) - math.pi

                if abs(angle_moved) < abs(target_rad):
                    cmd.angular.z = self.angular_speed if target_rad > 0 else -self.angular_speed
                else:
                    cmd.angular.z = 0.0
                    self.current_step += 1
                    self.state = 'MOVING'
                    self.initial_position = None
                    self.initial_yaw = None
                    self.move_heading = None
                    self.get_logger().info(f'Turn complete, advancing to step {self.current_step}')
            else:
                self.current_step += 1
                self.state = 'MOVING'
                self.initial_position = None
                self.initial_yaw = None
                self.move_heading = None

        self.pub.publish(cmd)

    def start_docking(self):
        goal = Dock.Goal()
        send_goal = self.dock_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal)
        handle = send_goal.result()
        if not handle.accepted:
            self.get_logger().error('Dock rejected')
            rclpy.shutdown()
            return
        res = handle.get_result_async()
        rclpy.spin_until_future_complete(self, res)
        if res.result.is_success:
            self.get_logger().info('Docking succeeded')
        else:
            self.get_logger().error('Docking failed')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = SentryMovement()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
