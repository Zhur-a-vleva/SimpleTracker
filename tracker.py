import math


class DistTracker:

    """
    A class for tracking the distance between objects in consecutive frames.
    """

    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def update(self, object_rect, threshold_dist):
        """
        Updates the object IDs based on their center points and distance threshold
        :param object_rect: a list of bounding boxes, confidence scores, and class IDs for the detected objects
        :param threshold_dist: the maximum distance between object center points to be considered the same object
        :return: a list of bounding boxes, class IDs, and object IDs for the detected objects
        """
        objects_bbs_ids = []

        # loop through each detected object
        for rect in object_rect:
            x1, y1, x2, y2, confidence, class_idx = rect

            # compute center points
            if class_idx in [1, 2, 3, 5, 6, 7]:
                x, y, w, h = x1, y1, abs(x1 - x2), abs(y1 - y2)
                cx = (x * 2 + w) // 2
                cy = (y * 2 + h) // 2

                same_object_detected = False

                # loop through each existing object
                for id, pt in self.center_points.items():
                    dist = math.hypot(cx - pt[0], cy - pt[1])

                    # if the distance between the centers is less than the threshold, update the center points
                    if dist < threshold_dist:
                        self.center_points[id] = (cx, cy)
                        objects_bbs_ids.append([x, y, w, h, class_idx, id])
                        same_object_detected = True
                        break

                # if no existing object was found nearby, create a new ID for the object
                if same_object_detected is False:
                    self.center_points[self.id_count] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, class_idx, self.id_count])
                    self.id_count += 1

        # update the center points dictionary
        new_center_points = {}

        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()

        return objects_bbs_ids
