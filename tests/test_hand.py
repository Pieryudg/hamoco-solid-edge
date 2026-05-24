#!/usr/bin/env python

import unittest
import os

import cv2
import mediapipe
from hamoco import Hand, HandSnapshot
from hamoco.utils import draw_hand_landmarks

mp_hands = mediapipe.solutions.hands

class Landmark:

    def __init__(self, x, y):
        self.x = x
        self.y = y

class Test(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        # Images
        self.images = os.listdir(self.data_dir)
        self.images = [file for file in self.images if file.startswith('POSE_')]
        self.images.sort()
        
    def test_hand_instantiation(self):
        hand_0 = Hand(pose=Hand.Pose.OPEN)
        hand_1 = Hand(pose=0)
        hand_2 = Hand(pose='OPEN')
        self.assertEqual(hand_0.pose, hand_1.pose)
        self.assertEqual(hand_1.pose, hand_2.pose)

    def test_snapshot(self):

        hands = mp_hands.Hands(static_image_mode=True)
        for index, image_file in enumerate(self.images):

            # Snapshot
            phony_hand = Hand(Hand.Pose(index))
            snapshot = HandSnapshot(hand=phony_hand)
            
            # Process with MediaPipe first to get raw landmarks
            image = cv2.imread(os.path.join(self.data_dir, image_file))
            image = cv2.flip(image, 1)
            mediapipe_results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw_hand_landmarks(image, mediapipe_results.multi_hand_landmarks[0])
            landmarks = mediapipe_results.multi_hand_landmarks[0].landmark
            
            # Image
            image_path = os.path.join(self.data_dir, f'test-snapshot_{phony_hand.pose.name}')
            snapshot.save_processed_image(image, path=image_path)

            # Vector
            vector_path = os.path.join(self.data_dir, f'test-vector_{phony_hand.pose.name}')
            snapshot.save_landmarks_vector(landmarks, path=vector_path)

    def test_closed_fist_landmark_fallback(self):
        landmarks = self._synthetic_curled_landmarks(thumb_tip=(0.48, 0.62))
        self.assertEqual(Hand.infer_pose_from_landmarks(landmarks), Hand.Pose.CLOSE)

    def test_thumb_side_landmark_fallback(self):
        landmarks = self._synthetic_curled_landmarks(thumb_tip=(0.20, 0.60))
        self.assertEqual(Hand.infer_pose_from_landmarks(landmarks), Hand.Pose.THUMB_SIDE)

    def test_open_hand_not_overridden_by_landmark_fallback(self):
        landmarks = self._synthetic_curled_landmarks(thumb_tip=(0.48, 0.62))
        for tip_index in (8, 12, 16, 20):
            landmarks[tip_index].y = 0.20
        self.assertIsNone(Hand.infer_pose_from_landmarks(landmarks))

    def _synthetic_curled_landmarks(self, thumb_tip):
        points = [(0.5, 0.8)] * 21
        points[1] = (0.45, 0.68)
        points[2] = (0.42, 0.64)
        points[3] = (0.45, 0.62)
        points[4] = thumb_tip
        for mcp, pip, dip, tip, x in [
            (5, 6, 7, 8, 0.40),
            (9, 10, 11, 12, 0.50),
            (13, 14, 15, 16, 0.60),
            (17, 18, 19, 20, 0.68),
        ]:
            points[mcp] = (x, 0.55)
            points[pip] = (x, 0.45)
            points[dip] = (x + 0.01, 0.52)
            points[tip] = (x + 0.02, 0.58)
        return [Landmark(x, y) for x, y in points]
            
if __name__ == '__main':
    unittest.main()
