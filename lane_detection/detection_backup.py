import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

class LaneDetection(Node):
    def __init__(self):
        super().__init__('lane_detection_node')
        self.videoSubscriber = self.create_subscription(
            CompressedImage,
            '/camera',
            self.videoSubscriber_callback,
            10
        )
    
    def gaussianBlur(self, src):
        gaussian_src = cv2.GaussianBlur(src, (9,9), sigmaX=0, sigmaY=0)
        return gaussian_src
    
    def yellowHsvInrange(self, src):
        lower_bound = np.array([20,120,120], dtype=np.uint8)
        upper_bound = np.array([35,255,255], dtype=np.uint8)
        hsv_src = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        hsv_dst = cv2.inRange(hsv_src, lower_bound, upper_bound)
        return hsv_dst
    
    def whiteHsvInrange(self, src):
        lower_bound = np.array([0,0,180], dtype=np.uint8)
        upper_bound = np.array([179,10,255], dtype=np.uint8)
        hsv_src = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        hsv_dst = cv2.inRange(hsv_src, lower_bound, upper_bound)
        return hsv_dst
    
    def componentsWithStatsFilter(self, src):
        min_area = 500
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
        valid_labels = np.where(stats[1:, cv2.CC_STAT_AREA] >= min_area)[0] + 1
        mask = np.isin(labels, valid_labels)
        filtered = (mask * 255).astype(np.uint8)
        return filtered
    
    def findYellowMoments(self, src):
        centers = []
        contours, hierarchy = cv2.findContours(src, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if cv2.contourArea(contour) < 2000: continue
            m = cv2.moments(contour)
            cx = int(m['m10']/m['m00'])
            cy = int(m['m01']/m['m00'])
            centers.append((cx, cy))
        return centers
    
    def findWhiteMoments(self, src):
        centers = []
        contours, hierarchy = cv2.findContours(src, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if cv2.contourArea(contour) < 2000: continue
            m = cv2.moments(contour)
            cx = int(m['m10']/m['m00'])
            cy = int(m['m01']/m['m00'])
            centers.append((cx, cy))
        return centers
    
    def videoSubscriber_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            src = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(f'Error decoding compressed image: {e}')
            return
        
        src2 = self.gaussianBlur(src)
        src_white = self.whiteHsvInrange(src2)
        src_white = self.componentsWithStatsFilter(src_white)
        src_yellow = self.yellowHsvInrange(src2)
        src_yellow = self.componentsWithStatsFilter(src_yellow)
        
        cv2.imshow('src', src)
        cv2.imshow('src_white', src_white)
        cv2.imshow('src_yellow', src_yellow)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
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
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
