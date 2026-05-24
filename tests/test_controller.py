#!/usr/bin/env python

import unittest
import os
from unittest.mock import patch

from hamoco import Hand, HandyMouseController, ClassificationModel
from hamoco.utils import draw_palm_center, draw_control_bounds, write_pose, draw_scrolling_origin
import numpy
import cv2

# Allow mouse to go on the border of the screen
import pyautogui
pyautogui.FAILSAFE = False

class Test(unittest.TestCase):

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.vectors = os.listdir(self.data_dir)
        self.vectors = [file for file in self.vectors if file.startswith('snapshot_')]
        self.vectors.sort()
        
    def test_controller(self):

        # Controller
        controller = HandyMouseController(sensitivity=0.15)
        controller.sensitivity = 1.5
        self.assertEqual(controller.sensitivity, 1.0) # value is clamped

        model = ClassificationModel()
        event_sequence = ['OPEN',
                          'CLOSE',
                          'THUMB_SIDE',
                          'THUMB_SIDE',
                          'THUMB_SIDE',
                          'CLOSE',
                          'OPEN',
                          'CLOSE',
                          'PINKY_UP',
                          'CLOSE',
                          'INDEX_UP',
                          'CLOSE',
                          'INDEX_MIDDLE_UP',
                          'INDEX_MIDDLE_UP',
                          'INDEX_MIDDLE_UP',
                          'CLOSE']

        phony_image = cv2.imread(os.path.join(self.data_dir, 'POSE_1_OPEN.jpg'))
        for event in event_sequence:

            # Pose name from event
            pose = Hand.Pose[event]
            vector_file = self.vectors[pose.value]
            
            # Get hand pose and palm center
            vector, _ = model.read_sample(os.path.join(self.data_dir, vector_file))
            vector = numpy.array(vector)
            palm_center = controller.palm_center(vector)
            confidence = 0.99
            
            # Operate mouse
            hand = Hand(pose=pose)
            controller.operate_mouse(hand, palm_center, confidence)

            # Drawing functions
            bounds = controller.accessible_area(phony_image)
            draw_control_bounds(phony_image, bounds)
            draw_palm_center(phony_image, palm_center, size=20)
            write_pose(phony_image, hand.pose.name)
            # Draw scrolling origin
            if controller.current_mouse_state == HandyMouseController.MouseState.SCROLLING:
                draw_scrolling_origin(phony_image, controller.scrolling_origin, controller.scrolling_threshold)

    def test_drag_button(self):
        controller = HandyMouseController(drag_button='middle')

        with patch('hamoco.controller.pyautogui.keyDown') as key_down, \
             patch('hamoco.controller.pyautogui.mouseDown') as mouse_down, \
             patch('hamoco.controller.pyautogui.mouseUp') as mouse_up, \
             patch('hamoco.controller.pyautogui.keyUp') as key_up:
            controller.begin_drag()
            controller.begin_drag()
            controller.end_drag()
            controller.end_drag()

        key_down.assert_not_called()
        mouse_down.assert_called_once_with(button='middle', _pause=False)
        mouse_up.assert_called_once_with(button='middle', _pause=False)
        key_up.assert_not_called()

    def test_release_controls(self):
        controller = HandyMouseController(drag_button='middle')
        controller.enter_state(HandyMouseController.MouseState.DRAGGING)

        with patch('hamoco.controller.pyautogui.keyDown'), \
             patch('hamoco.controller.pyautogui.mouseDown'), \
             patch('hamoco.controller.pyautogui.mouseUp') as mouse_up, \
             patch('hamoco.controller.pyautogui.keyUp') as key_up:
            controller.begin_drag()
            controller.release_controls()

        mouse_up.assert_called_once_with(button='middle', _pause=False)
        key_up.assert_not_called()
        self.assertEqual(controller.current_mouse_state, HandyMouseController.MouseState.STANDARD)

    def test_drag_modifier(self):
        controller = HandyMouseController(drag_modifier='ctrl', drag_button='left')

        with patch('hamoco.controller.pyautogui.keyDown') as key_down, \
             patch('hamoco.controller.pyautogui.mouseDown') as mouse_down, \
             patch('hamoco.controller.pyautogui.mouseUp') as mouse_up, \
             patch('hamoco.controller.pyautogui.keyUp') as key_up:
            controller.begin_drag()
            controller.end_drag()

        key_down.assert_called_once_with('ctrl', _pause=False)
        mouse_down.assert_called_once_with(button='left', _pause=False)
        mouse_up.assert_called_once_with(button='left', _pause=False)
        key_up.assert_called_once_with('ctrl', _pause=False)

    def test_begin_drag_releases_modifier_if_mouse_down_fails(self):
        controller = HandyMouseController(drag_modifier='ctrl', drag_button='left')

        with patch('hamoco.controller.pyautogui.keyDown'), \
             patch('hamoco.controller.pyautogui.mouseDown', side_effect=RuntimeError), \
             patch('hamoco.controller.pyautogui.keyUp') as key_up:
            with self.assertRaises(RuntimeError):
                controller.begin_drag()

        key_up.assert_called_once_with('ctrl', _pause=False)
        self.assertFalse(controller.drag_active)

if __name__ == '__main':
    unittest.main()
