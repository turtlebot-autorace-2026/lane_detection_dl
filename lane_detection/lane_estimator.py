'''
Deep learning (YOLO seg) based lane following for TurtleBot3 AutoRace.

All geometry is in the BEV image (640x480) made from the undistorted /camera image
with the BEV parameters below. BEV is metric-isotropic: 160 px = 1 camera height.
Left lane = yellow, right lane = white (fixed).

Pipeline (one frame):
  1. keep only the lane masks inside the ROI (near the robot, own road)
  2. fit one curve x(row) per lane color
  3. path center: both lanes -> middle, one lane -> offset from it, none -> stop
  4. pure pursuit on a target point of the path -> (v, w)
'''
import cv2
import numpy as np

# ---- BEV (from bev_params.yaml) ----
BEV_W, BEV_H = 640, 480
BEV_SRC = np.float32([[18, 451], [633, 477], [445, 157], [216, 153]])
BEV_DST = np.float32([[166, 474], [474, 474], [474, 122], [166, 122]])
BEV_M = cv2.getPerspectiveTransform(BEV_SRC, BEV_DST)
BEV_PX_PER_CAMERA_HEIGHT = 160.0   # BEV scale
BEV_BOTTOM_AHEAD = 0.76            # ground distance of the BEV bottom row from the camera [camera heights]

# model class ids
WHITE, YELLOW = 0, 1


def to_bev(undistorted):
    if undistorted.shape[:2] != (480, 640):
        undistorted = cv2.resize(undistorted, (640, 480))
    return cv2.warpPerspective(undistorted, BEV_M, (BEV_W, BEV_H))


class LaneConfig:
    def __init__(self, **kw):
        # geometry
        self.robot_x = 326.0          # BEV x of the robot center line
        self.lane_width = 320.0       # yellow-center to white-center distance [px]
        self.roi_top_row = 220        # ROI: rows below this only (~31 cm ahead of the camera)
        self.roi_half_width = 250     # ROI: +-px around the robot center line (own lane lines at ~+-160 px)
        self.row_step = 10            # rows sampled for the lane curves
        self.min_points = 5           # sampled rows needed to accept a lane
        self.single_offset = 0.6      # distance from the line when only one lane is seen [x lane width]
        # control (pure pursuit)
        self.lookahead_row = 415      # target point row (~15 cm ahead of the camera)
        self.camera_height = 0.129    # m, sets the BEV metric scale
        self.camera_ahead_of_axle = 0.076   # m
        self.k_lat = 1.0              # pure pursuit gain
        self.kappa_max = 7.0          # 1/m
        self.v_max = 0.12             # m/s
        self.v_max_intersection = 0.08
        self.ang_scale_intersection = 0.7
        self.k_v = 0.35               # v = v_max / (1 + k_v*|kappa|): slow down in curves
        self.v_single = 0.8           # speed factor when only one lane is used
        self.max_ang = 0.8            # rad/s
        self.ang_smooth = 0.4         # weight of the new angular command (low-pass)
        for k, v in kw.items():
            setattr(self, k, v)
        self.rows = np.arange(BEV_H - 10, self.roi_top_row - 1, -self.row_step)   # bottom -> top


def _lane_curve(mask, rows, min_points):
    '''x(row) of one lane: per-row center of the mask, fitted with a polynomial. None if too few points.'''
    pts = []
    for r in rows:
        cols = np.nonzero(mask[max(r - 2, 0):r + 3].any(axis=0))[0]
        if len(cols):
            pts.append((r, np.median(cols)))
    if len(pts) < min_points:
        return None
    r, x = np.array(pts).T
    coef = np.polyfit(r, x, 2 if len(pts) >= 8 else 1)
    return np.polyval(coef, rows)


def _offset_curve(xs, rows, d):
    '''Shift a lane curve by d px along its normal (+: to the right when driving up the image).'''
    dxdr = np.gradient(xs, rows)
    n = np.sqrt(1 + dxdr ** 2)
    nx, nr = xs + d / n, rows - d * dxdr / n
    order = np.argsort(nr)
    return np.interp(rows, nr[order], nx[order])


class LaneEstimator:
    def __init__(self, cfg=None):
        self.cfg = cfg or LaneConfig()
        self.prev_ang = 0.0
        self.yellow = self.white = self.center = None
        self.debug = {}

    def roi_mask(self):
        cfg = self.cfg
        roi = np.zeros((BEV_H, BEV_W), bool)
        x0 = int(max(0, cfg.robot_x - cfg.roi_half_width))
        x1 = int(min(BEV_W, cfg.robot_x + cfg.roi_half_width))
        roi[int(cfg.roi_top_row):, x0:x1] = True
        return roi

    def update(self, detections, lane_mode=True, intersection=False):
        '''
        detections: list of (cls, mask HxW bool in BEV coords, conf)
        lane_mode: True -> prefer yellow (left) lane, False -> prefer white (right) lane
        returns (linear_x, angular_z)
        '''
        cfg, rows = self.cfg, self.cfg.rows

        # 1. lane masks inside the ROI, one union mask per color
        roi = self.roi_mask()
        masks = {YELLOW: np.zeros_like(roi), WHITE: np.zeros_like(roi)}
        for cls, mask, _ in detections:
            if cls in masks:
                masks[cls] |= mask & roi

        # 2. one curve per lane
        y = _lane_curve(masks[YELLOW], rows, cfg.min_points)
        w = _lane_curve(masks[WHITE], rows, cfg.min_points)

        # both seen but not a normal lane (order swapped / too narrow / fork): use the lane_mode lane
        if y is not None and w is not None:
            gap = np.median(w - y)
            if not (0.6 * cfg.lane_width < gap < 1.4 * cfg.lane_width):
                y, w = (y, None) if lane_mode else (None, w)
        self.yellow, self.white = y, w

        # 3. path center
        single = cfg.lane_width * cfg.single_offset
        if y is not None and w is not None:
            center, mode = (y + w) / 2, 'both'
        elif y is not None:
            center, mode = _offset_curve(y, rows, +single), 'yellow_only'
        elif w is not None:
            center, mode = _offset_curve(w, rows, -single), 'white_only'
        else:
            # no lane -> stop
            self.center, self.prev_ang = None, 0.0
            self.debug = dict(mode='lost')
            return 0.0, 0.0
        self.center = center

        # 4. pure pursuit on the target point (rotation center = wheel axle)
        x_la = np.interp(cfg.lookahead_row, rows[::-1], center[::-1])
        m_per_px = cfg.camera_height / BEV_PX_PER_CAMERA_HEIGHT
        ahead = (BEV_BOTTOM_AHEAD + (BEV_H - cfg.lookahead_row) / BEV_PX_PER_CAMERA_HEIGHT) * cfg.camera_height \
            + cfg.camera_ahead_of_axle
        lateral = -(x_la - cfg.robot_x) * m_per_px                                  # +: target is left
        kappa = cfg.k_lat * 2 * lateral / (ahead ** 2 + lateral ** 2)
        if intersection:
            kappa *= cfg.ang_scale_intersection
        kappa = float(np.clip(kappa, -cfg.kappa_max, cfg.kappa_max))

        v = (cfg.v_max_intersection if intersection else cfg.v_max) / (1 + cfg.k_v * abs(kappa))
        if mode != 'both':
            v *= cfg.v_single
        ang = float(np.clip(v * kappa, -cfg.max_ang, cfg.max_ang))
        ang = cfg.ang_smooth * ang + (1 - cfg.ang_smooth) * self.prev_ang
        self.prev_ang = ang

        self.debug = dict(mode=mode, x_la=float(x_la), kappa=kappa, v=float(v), w=ang,
                          e_lat=float((x_la - cfg.robot_x) / (cfg.lane_width / 2)))
        return float(v), ang

    def draw(self, bev, detections=None):
        cfg, rows = self.cfg, self.cfg.rows
        img = bev.copy()
        if detections:
            ov = img.copy()
            for cls, mask, _ in detections:
                ov[mask] = (0, 255, 255) if cls == YELLOW else (255, 0, 255)
            img = cv2.addWeighted(ov, 0.4, img, 0.6, 0)
        for xs, col in ((self.yellow, (0, 200, 255)), (self.white, (255, 255, 255)), (self.center, (0, 255, 0))):
            if xs is not None:
                cv2.polylines(img, [np.int32(np.c_[xs, rows])], False, col, 2)
        if 'x_la' in self.debug:
            cv2.circle(img, (int(self.debug['x_la']), cfg.lookahead_row), 7, (0, 0, 255), -1)
        cv2.line(img, (int(cfg.robot_x), BEV_H), (int(cfg.robot_x), BEV_H - 60), (255, 0, 0), 2)
        x0 = int(max(0, cfg.robot_x - cfg.roi_half_width))
        x1 = int(min(BEV_W - 1, cfg.robot_x + cfg.roi_half_width))
        cv2.rectangle(img, (x0, int(cfg.roi_top_row)), (x1, BEV_H - 1), (0, 255, 255), 1)
        d = self.debug
        txt = d.get('mode', '') + (f" v={d['v']:.2f} w={d['w']:+.2f}" if 'v' in d else '')
        cv2.putText(img, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return img
