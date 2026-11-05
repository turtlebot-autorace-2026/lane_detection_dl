import cv2
import time
import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist

THRESHOLD = 1

class IntersectionDetection(Node):
    def __init__(self):
        super().__init__('intersection_detection_node')
        self.videoSubscriber = self.create_subscription(
            CompressedImage,
            '/camera2',
            self.videoSubscriber_callback,
            10
        )
        self.stateSubscriber = self.create_subscription(
            Bool,
            '/intersection_state',
            self.stateSubscriber_callback,
            10
        )
        self.statePublisher = self.create_publisher(
            Bool,
            'intersection_state',
            10
        )
        self.laneModePublisher = self.create_publisher(
            Bool,
            'lane_mode',
            10
        )
        self.laneStatePublisher = self.create_publisher(
            Bool,
            'lane_state',
            10
        )
        self.velPublisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        
        self.state = False
        self.first = True
        self.l_count = 0
        self.r_count = 0
    
    def stateSubscriber_callback(self, msg):
        self.state = msg.data
        if self.state == True:
            if self.first:
                print('\nStart Intersection Detection')
                self.first = False
            msg = Bool()
            msg.data = True
            self.statePublisher.publish(msg)
        else: print('Terminate Intersection Detection\n')
    
    def gaussianBlur(self, src):
        gaussian_src = cv2.GaussianBlur(src, (9,9), sigmaX=0, sigmaY=0)
        return gaussian_src
    
    def detectRL(self, src):
        gray = self.gaussianBlur(src)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=200,
            param1=100,
            param2=50,#60
            minRadius=10,
            maxRadius=100
        )
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x,y,r) in circles[0,:]:
                cv2.circle(src, (x,y), r, (0,255,0), 2)
                cv2.circle(src, (x,y), 3, (0,255,0), -1)
                try:
                    left = (int(x-r/3),int(y+r/3))
                    right = (int(x+r/3),int(y+r/3))
                    cv2.circle(src, left, 3, (255,50,50), -1)
                    cv2.circle(src, right, 3, (255,150,150), -1)
                    if gray[left[1],left[0]] > gray[right[1],right[0]]: return src, -1
                    elif gray[left[1],left[0]] < gray[right[1],right[0]]: return src, 1
                except: self.get_logger().info('\n**out of frame**\n')
        return src, 0
    
    def move_back(self):
        twist = Twist()
        twist.linear.x = -0.2
        duration = 0.8
        end_time = time.time() + duration
        while time.time() < end_time:
            self.velPublisher.publish(twist)
            time.sleep(0.1)
        msg = Bool()
        msg.data = True
        self.laneStatePublisher.publish(msg)
    
    def videoSubscriber_callback(self, msg):
        if not self.state: return
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            src = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            src, result = self.detectRL(src)
        except Exception as e:
            self.get_logger().error(f'Error decoding compressed image: {e}')
            return
        
        if result == 1:
            '''
            if self.l_count < THRESHOLD: self.l_count += 1
            else:
                msg = Bool()
                msg.data = False
                self.laneStatePublisher.publish(msg)
                self.get_logger().info('left sign detected')
                self.move_back()
                msg.data = True
                self.laneModePublisher.publish(msg)
                msg.data = False
                self.statePublisher.publish(msg)
                self.state = False
                cv2.destroyAllWindows()
                self.get_logger().info('Terminate Intersection Detection')
                self.destroy_node()
                rclpy.shutdown()
            '''
            msg = Bool()
            msg.data = False
            self.laneStatePublisher.publish(msg)
            self.get_logger().info('left sign detected')
            self.move_back()
            msg.data = True
            self.laneModePublisher.publish(msg)
            msg.data = False
            self.statePublisher.publish(msg)
            self.state = False
            cv2.destroyAllWindows()
            self.get_logger().info('Terminate Intersection Detection')
            self.destroy_node()
            rclpy.shutdown()
            
        elif result == -1:
            '''
            if self.r_count < THRESHOLD: self.r_count += 1
            else:
                msg = Bool()
                msg.data = False
                self.laneStatePublisher.publish(msg)
                self.get_logger().info('right sign detected')
                self.move_back()
                msg.data = False
                self.laneModePublisher.publish(msg)
                msg.data = False
                self.statePublisher.publish(msg)
                self.state = False
                cv2.destroyAllWindows()
                self.get_logger().info('Terminate Intersection Detection')
                self.destroy_node()
                rclpy.shutdown()
            '''
            msg = Bool()
            msg.data = False
            self.laneStatePublisher.publish(msg)
            self.get_logger().info('left sign detected')
            self.move_back()
            msg.data = True
            self.laneModePublisher.publish(msg)
            msg.data = False
            self.statePublisher.publish(msg)
            self.state = False
            cv2.destroyAllWindows()
            self.get_logger().info('Terminate Intersection Detection')
            self.destroy_node()
            rclpy.shutdown()
        cv2.imshow('src', src)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            self.destroy_node()
            rclpy.shutdown()

def main():
    rclpy.init()
    node = IntersectionDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
