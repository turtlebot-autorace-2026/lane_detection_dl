import os
import cv2
import rclpy
import threading
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO

from lane_detection.lane_estimator import LaneEstimator, LaneConfig, to_bev

class LaneDetection(Node):
    def __init__(self):
        super().__init__('lane_detection_node')
        self.videoSubscriber = self.create_subscription(
            CompressedImage,
            '/camera',
            self.videoSubscriber_callback,
            10
        )
        self.laneModeSubscriber = self.create_subscription(
            Bool,
            '/lane_mode',
            self.laneModeSubscriber_callback,
            10
        )
        self.stateSubscriber = self.create_subscription(
            Bool,
            '/lane_state',
            self.stateSubscriber_callback,
            10
        )
        self.velocityPublisher = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )
        self.intersectionStateSubscriber = self.create_subscription(
            Bool,
            '/intersection_state',
            self.intersectionStateSubscriber_callback,
            10
        )
        self.lane_mode = True  # default = True(yellow)
        self.state = False
        self.is_intersection = False

        # Deep learning lane model (YOLO seg, classes: 0 white, 1 yellow)
        default_model = os.path.join(get_package_share_directory('lane_detection'), 'models', 'lane_yolo26n_seg.pt')
        self.declare_parameter('model_path', default_model)
        self.declare_parameter('conf', 0.4)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('device', '')          # '' = auto (cuda if available), 'cpu', '0'
        self.declare_parameter('show', True)          # cv2 debug window
        self.declare_parameter('process_hz', 15.0)
        self.declare_parameter('publish_debug', True)  # BEV debug view on /lane_detection/debug/compressed
        self.declare_parameter('v_max', 0.12)                 # m/s
        self.declare_parameter('v_max_intersection', 0.08)    # m/s
        self.declare_parameter('k_lat', 1.0)                  # pure pursuit gain
        self.declare_parameter('camera_height', 0.129)        # m, camera height above the road
        self.declare_parameter('roi_top_row', 220)            # ROI: BEV rows below this only (480 = bottom)
        self.declare_parameter('roi_half_width', 250)         # ROI: +-px around the robot center line
        self.declare_parameter('max_ang', 0.8)                # rad/s
        self.declare_parameter('lookahead_row', 415)          # BEV row of the steering target (480 = bottom)
        self.declare_parameter('single_offset', 0.6)          # distance from the line when following one lane [x lane width]
        self.conf = self.get_parameter('conf').value
        self.imgsz = self.get_parameter('imgsz').value
        self.device = self.get_parameter('device').value or None
        self.show = self.get_parameter('show').value
        model_path = self.get_parameter('model_path').value
        self.model = YOLO(model_path, task='segment')
        self.get_logger().info(f'Lane model loaded: {model_path}')

        self.debugPublisher = None
        if self.get_parameter('publish_debug').value:
            self.debugPublisher = self.create_publisher(CompressedImage, '/lane_detection/debug/compressed', 10)

        self.estimator = LaneEstimator(self.lane_config())
        # process only the latest frame (inference may be slower than the camera)
        self.latest_msg = None
        self.timer = self.create_timer(1.0 / self.get_parameter('process_hz').value, self.process)


    def intersectionStateSubscriber_callback(self, msg):
        if not self.is_intersection and msg.data:
            self.is_intersection = msg.data
            self.get_logger().info('===>> Start Intersection Mode')
        if self.is_intersection and not msg.data:
            self.is_intersection = msg.data
            self.get_logger().info('===>> Finish Intersection Mode')
            if self.lane_mode == True:
                threading.Timer(25.0, self.intersection_left_handling).start()
            else:
                threading.Timer(12.0, self.intersection_right_handling).start()
                threading.Timer(28.0, self.intersection_right_handling2).start()

    def intersection_left_handling(self):
        self.lane_mode = False
        cv2.destroyAllWindows()

    def intersection_right_handling(self):
        self.lane_mode = True
        cv2.destroyAllWindows()

    def intersection_right_handling2(self):
        self.lane_mode = False
        cv2.destroyAllWindows()

    def stateSubscriber_callback(self, msg):
        self.state = msg.data
        if self.state == True:
            self.estimator = LaneEstimator(self.lane_config())   # start with a fresh lane track
            print('Start Lane Detection')
        else:
            twist = Twist()
            self.velocityPublisher.publish(twist)
            print('Terminate Lane Detection')
            cv2.destroyAllWindows()

    def laneModeSubscriber_callback(self, msg):
        self.lane_mode = msg.data
        if self.lane_mode: self.get_logger().info('Change lane mode to YELLOW')
        else: self.get_logger().info('Change lane mode to WHITE')
        cv2.destroyAllWindows()
        return

    def lane_config(self):
        return LaneConfig(v_max=self.get_parameter('v_max').value,
                          v_max_intersection=self.get_parameter('v_max_intersection').value,
                          k_lat=self.get_parameter('k_lat').value,
                          max_ang=self.get_parameter('max_ang').value,
                          lookahead_row=self.get_parameter('lookahead_row').value,
                          single_offset=self.get_parameter('single_offset').value,
                          camera_height=self.get_parameter('camera_height').value,
                          roi_top_row=self.get_parameter('roi_top_row').value,
                          roi_half_width=self.get_parameter('roi_half_width').value)

    def detect(self, bev):
        '''YOLO seg inference on the BEV image -> list of (cls, mask bool HxW, conf)'''
        result = self.model.predict(bev, imgsz=self.imgsz, conf=self.conf, device=self.device,
                                    retina_masks=True, verbose=False)[0]
        if result.masks is None:
            return []
        masks = result.masks.data.cpu().numpy() > 0.5
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        return [(c, m, float(p)) for c, m, p in zip(classes, masks, confs)]

    def videoSubscriber_callback(self, msg):
        if not self.state: return
        self.latest_msg = msg

    def process(self):
        msg, self.latest_msg = self.latest_msg, None
        if msg is None or not self.state: return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            src = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            bev = to_bev(src)
        except Exception as e:
            self.get_logger().error(f'Error decoding compressed image: {e}')
            return

        detections = self.detect(bev)
        v, w = self.estimator.update(detections, lane_mode=self.lane_mode, intersection=self.is_intersection)

        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.velocityPublisher.publish(twist)

        if self.debugPublisher is not None:
            debug = CompressedImage()
            debug.header = msg.header
            debug.format = 'jpeg'
            debug.data = cv2.imencode('.jpg', self.estimator.draw(bev, detections))[1].tobytes()
            self.debugPublisher.publish(debug)

        if self.show:
            cv2.imshow('lane', self.estimator.draw(bev, detections))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                twist = Twist()
                self.velocityPublisher.publish(twist)
                cv2.destroyAllWindows()
                self.destroy_node()
                rclpy.shutdown()

def main():
    rclpy.init()
    node = LaneDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        try:
            node.velocityPublisher.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
