import time
import enum

import numpy
import cv2

class Hand:

    @enum.unique
    class Pose(enum.IntEnum):
        UNDEFINED = -1
        OPEN = 0
        CLOSE = 1
        INDEX_UP = 2
        PINKY_UP = 3
        THUMB_SIDE = 4
        INDEX_MIDDLE_UP = 5

    # Indices of palm landmarks in mediapipe hands
    palm_landmarks = [0, 5, 9, 13, 17]

    # Dimension: only look at X and Y for landmarks (discard Z)
    # If Z must be added at some point, changes will be minor
    dimension = 2

    def __init__(self, pose=None):
        if pose is None:
            self.pose = self.Pose.UNDEFINED
        else:
            if isinstance(pose, self.Pose):
                self.pose = pose
            if isinstance(pose, int):
                self.pose = self.Pose(pose)
            if isinstance(pose, str):
                self.pose = self.Pose[pose]

    def vectorize_landmarks(self, landmarks):
        landmarks_vector = numpy.zeros(self.dimension * len(landmarks))
        for lm_i, landmark in enumerate(landmarks):
            landmarks_vector[self.dimension*lm_i] = landmark.x
            landmarks_vector[self.dimension*lm_i+1] = landmark.y
        return landmarks_vector

    def feature_process_landmarks(self, landmarks_vector):
        processed_landmarks = landmarks_vector.copy()
        for axis in range(self.dimension):
            # Translate center of mass back to origin
            processed_landmarks[axis::self.dimension] -= processed_landmarks[axis::self.dimension].mean()
            # Make scale invariant
            processed_landmarks[axis::self.dimension] /= processed_landmarks[axis::self.dimension].std()
        return processed_landmarks.reshape(1,-1)

    @staticmethod
    def _landmark_xy(landmark):
        return numpy.array([landmark.x, landmark.y])

    @staticmethod
    def _distance(point_0, point_1):
        return numpy.linalg.norm(point_0 - point_1)

    @classmethod
    def infer_pose_from_landmarks(cls, landmarks):
        """Infer simple stop gestures directly from MediaPipe landmarks.

        The trained model handles the main classification path. This heuristic is
        deliberately narrow and only distinguishes a closed fist from the thumb
        side gesture so that CLOSE remains reliable as a safety/reset pose.
        """
        points = [cls._landmark_xy(landmark) for landmark in landmarks]
        wrist = points[0]

        curled_fingers = 0
        for mcp_index, pip_index, _, tip_index in [
            (5, 6, 7, 8),
            (9, 10, 11, 12),
            (13, 14, 15, 16),
            (17, 18, 19, 20),
        ]:
            pip_to_wrist = cls._distance(points[pip_index], wrist)
            tip_to_wrist = cls._distance(points[tip_index], wrist)
            tip_below_pip = points[tip_index][1] >= points[pip_index][1] - 0.015
            folded_toward_palm = tip_to_wrist <= pip_to_wrist * 1.08
            if tip_below_pip and folded_toward_palm:
                curled_fingers += 1

        if curled_fingers < 4:
            return None

        palm_center = numpy.mean([points[index] for index in cls.palm_landmarks], axis=0)
        palm_width = max(cls._distance(points[5], points[17]), 0.001)
        thumb_side_offset = abs(points[4][0] - palm_center[0])
        thumb_vertical_offset = abs(points[4][1] - palm_center[1])
        thumb_sideways = thumb_side_offset > palm_width * 0.65 and thumb_vertical_offset < palm_width * 0.95

        if thumb_sideways:
            return cls.Pose.THUMB_SIDE
        return cls.Pose.CLOSE

class HandSnapshot:

    def __init__(self, hand=None):
        self.time = time.time()
        if hand is None:
            self.hand = Hand()
        else:
            self.hand = hand

    def save_processed_image(self, image, path=None):
        '''Save image with landmarks on them.'''
        if path is None:
            path = 'hand_snapshot'
        cv2.imwrite(f'{path}.jpg', image)

    def save_landmarks_vector(self, landmarks, path=None):
        '''Save the landmarks vector to a text file.'''
        if path is None:
            path = 'defaut_name'
        raw_landmarks_vector = self.hand.vectorize_landmarks(landmarks)
        processed_landmarks_vector = self.hand.feature_process_landmarks(raw_landmarks_vector)
        with open(f'{path}.dat', 'w') as vec_file:
            vec_file.write(f'{self.hand.pose.value}\n')
            for pos in processed_landmarks_vector.flatten():
                vec_file.write(f'{pos:.6f} ')
            vec_file.write('\n')
