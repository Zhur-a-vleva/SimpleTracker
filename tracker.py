import math


class DistTracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def update(self, object_rect):
        objects_bbs_ids = []

        for rect in object_rect:
            x1, y1, x2, y2, confidence, class_idx = rect
            if class_idx in [1, 2, 3, 5, 6, 7]:
                x, y, w, h = x1, y1, abs(x1 - x2), abs(y1 - y2)
                cx = (x * 2 + w) // 2
                cy = (y * 2 + h) // 2

                same_object_detected = False
                for id, pt in self.center_points.items():
                    dist = math.hypot(cx - pt[0], cy - pt[1])

                    if dist < 50:
                        self.center_points[id] = (cx, cy)
                        objects_bbs_ids.append([x, y, w, h, class_idx, id])
                        same_object_detected = True
                        break

                if same_object_detected is False:
                    self.center_points[self.id_count] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, class_idx, self.id_count])
                    self.id_count += 1

        new_center_points = {}

        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids
