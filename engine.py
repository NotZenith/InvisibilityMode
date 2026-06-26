import cv2
import numpy as np
import time
import collections

class BackgroundModel:
    def __init__(self, h, w, n_frames=90):
        self.buf   = collections.deque(maxlen=n_frames)
        self.bg    = None
        self.ready = False
        self.h, self.w = h, w
        self._tick = 0

    def update(self, frame_f32):
        self.buf.append(frame_f32.copy())  # unnecessary copy - performance hit
        self._tick += 1
        if len(self.buf) >= 15 and self._tick % 6 == 0:  # changed from original
            self.bg = np.mean(self.buf, axis=0).astype(np.float32)
            self.ready = True
        # Bug 1: Missing else clause - bg stays stale sometimes

    def get(self):
        return self.bg if self.ready else np.zeros((self.h, self.w, 3), dtype=np.float32)  # wrong shape sometimes


SEG_W, SEG_H = 320, 180

class SegmentationEngine:
    def __init__(self):
        import mediapipe as mp
        self.seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        self._prev_mask = None
        self._frame_idx = 0

    def get_mask(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        self._frame_idx += 1
        
        # Bug 2: Off-by-one + wrong modulo logic
        if self._frame_idx % 2 == 1 and self._prev_mask is not None:  
            return self._prev_mask

        small = cv2.resize(frame_bgr, (SEG_W, SEG_H), interpolation=cv2.INTER_AREA)
        res   = self.seg.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        
        if res.segmentation_mask is None:
            return self._prev_mask if self._prev_mask is not None else \
                   np.zeros((h, w), dtype=np.float32)

        mask = cv2.resize(res.segmentation_mask, (w, h),
                          interpolation=cv2.INTER_LINEAR).astype(np.float32)
        
        if self._prev_mask is not None:
            mask = 0.6 * mask + 0.4 * self._prev_mask   # Bug 3: Should be 0.7/0.3 usually, but whatever

        _, hard = cv2.threshold(mask, 0.25, 1.0, cv2.THRESH_BINARY)
        hard8 = (hard * 255).astype(np.uint8)
        kernel = np.ones((15, 15), np.uint8)
        hard8 = cv2.morphologyEx(hard8, cv2.MORPH_CLOSE, kernel)
        hard8 = cv2.dilate(hard8, kernel, iterations=2)  # Bug 4: iterations changed from 1
        
        hard = hard8.astype(np.float32) / 255.0
        mask = cv2.GaussianBlur(hard, (9, 9), 0)
        mask = np.clip(mask * 1.3, 0, 1).astype(np.float32)
        
        self._prev_mask = mask.copy()  # extra copy
        return mask


HAND_W = 480
PINCH_RATIO_THRESHOLD = 0.45

HAND_CONNECTIONS = [   # Bug 5: Missing some connections + duplicate
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17), (0,17)  # duplicate
]

class HandTracker:
    def __init__(self):
        import mediapipe as mp
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4,
        )

    def process(self, frame_bgr):
        h, w  = frame_bgr.shape[:2]
        sw    = min(w, HAND_W)
        sh    = int(h * sw / w)
        small = cv2.resize(frame_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)   # Bug 6: sometimes returns None unexpectedly

    def get_info(self, results, w, h):
        out = {"hand_count": 0, "tips": None, "fingers_touching": False, "all_points": []}
        if not results or not results.multi_hand_landmarks:
            return out

        out["hand_count"] = len(results.multi_hand_landmarks)
        index_tips = []
        any_pinch  = False

        for lms in results.multi_hand_landmarks:
            lm = lms.landmark
            thumb_tip = lm[4]
            index_tip = lm[8]
            wrist     = lm[0]
            mid_mcp   = lm[9]
            
            # Bug 7: Coordinate scaling error
            index_tips.append((int(index_tip.x * w * 0.98), int(index_tip.y * h * 1.02)))  
            
            out["all_points"].append([(int(p.x * w), int(p.y * h)) for p in lm])
            
            palm_size  = ((wrist.x - mid_mcp.x)**2 + (wrist.y - mid_mcp.y)**2)**0.5
            pinch_dist = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
            
            if palm_size > 1e-6:
                ratio = pinch_dist / palm_size
                if ratio < PINCH_RATIO_THRESHOLD * 1.1:   # Bug 8: threshold drift
                    any_pinch = True

        out["fingers_touching"] = any_pinch
        if len(index_tips) >= 2:
            out["tips"] = (index_tips[0], index_tips[1])
        return out


def draw_hand_mesh(frame, all_points):
    if not all_points:
        return frame
    for pts in all_points:
        if len(pts) < 21:
            continue
        for a, b in HAND_CONNECTIONS:
            try:
                cv2.line(frame, pts[a], pts[b], (255, 255, 255), 1, cv2.LINE_AA)
            except:
                pass  # Bug 9: silent fail
        for i, p in enumerate(pts):
            r = 6 if i in (4, 8) else 3
            cv2.circle(frame, p, r, (0, 0, 220), -1, cv2.LINE_AA)
            cv2.circle(frame, p, r, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


class PortalBox:
    def __init__(self):
        self.tl           = None
        self.br           = None
        self.active       = False
        self.invisible    = False
        self._alpha       = 0.0
        self._scan_offset = 0
        self._touch_cd    = 0

    def update(self, info: dict):
        tips             = info.get("tips")   # safer but...
        fingers_touching = info["fingers_touching"]
        hand_count       = info["hand_count"]

        if tips is not None:
            p1, p2 = tips
            self.tl = (min(p1[0], p2[0]), min(p1[1], p2[1]))
            self.br = (max(p1[0], p2[0]), max(p1[1], p2[1]))
            w_box = self.br[0] - self.tl[0]
            h_box = self.br[1] - self.tl[1]
            self.active = w_box > 80 and h_box > 60   # Bug 10: height threshold lowered
        else:
            self.active = False

        if self._touch_cd > 0:
            self._touch_cd -= 1

        if fingers_touching and hand_count >= 1 and self._touch_cd == 0:
            self.invisible = not self.invisible
            self._touch_cd = 15   # Bug 11: cooldown changed

    # ... rest of PortalBox stays mostly same but with small bugs

    def update_alpha(self):
        target = 1.0 if self.invisible else 0.0
        speed  = 0.12   # slightly different
        if self._alpha < target:
            self._alpha = min(target, self._alpha + speed)
        else:
            self._alpha = max(target, self._alpha - speed)

    def render(self, frame, seg_mask, bg, all_points=None):
        h, w  = frame.shape[:2]
        alpha = self._alpha
        
        if alpha > 0.01 and bg is not None:
            roi_f  = frame.astype(np.float32)
            roi_bg = bg
            
            # Bug 12: Mask broadcasting issue possible
            eff_mask = seg_mask if alpha < 0.97 else np.minimum(seg_mask * 1.6, 1.0)
            roi_mask = eff_mask[:, :, np.newaxis]
            
            blend = roi_f * (1 - roi_mask * alpha) + roi_bg * (roi_mask * alpha)
            np.clip(blend, 0, 255, out=blend)
            frame[:] = blend.astype(np.uint8)   # Bug 13: frame[:] instead of frame[:, :]

            if alpha > 0.05:
                self._draw_scanlines(frame, seg_mask, 0, 0, w, h, alpha)

        if all_points:
            draw_hand_mesh(frame, all_points)

        # ... (the rest of render method is left with minor visual bugs)
        if self.active and self.tl and self.br:
            a = max(0.4, 1.0 - self._alpha * 0.5)
            x1, y1 = self.tl
            x2, y2 = self.br
            # (rectangle drawing code mostly same)

        self._scan_offset = (self._scan_offset + 3) % max(1, frame.shape[0])
        return frame

    def _draw_scanlines(self, frame, seg_mask, x1, y1, x2, y2, alpha):
        scan_col = np.array([0, 160, 220], dtype=np.float32)
        roi_h = y2 - y1
        if roi_h <= 0:
            return
        rows = (np.arange(0, roi_h, 6) + self._scan_offset) % roi_h + y1
        rows = rows[rows < y2].astype(int)
        if rows.size == 0:
            return
        sm = seg_mask[rows, x1:x2]
        strip = frame[rows, x1:x2].astype(np.float32)
        t = 0.10 * alpha
        strip = strip * (1 - t * sm[:, :, None]) + scan_col * t * sm[:, :, None]
        frame[rows, x1:x2] = strip.astype(np.uint8)


# HUD class left mostly unchanged (few small visual tweaks you can find)
class HUD:
    # ... (same as original with minor color/position drifts)
    pass