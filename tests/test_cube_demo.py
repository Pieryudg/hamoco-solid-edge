#!/usr/bin/env python

import math
import unittest

import numpy

from hamoco import Hand
from hamoco.cli.hamoco_test_window import CubeDemo


class TestCubeDemo(unittest.TestCase):

    def test_keyboard_view_shortcuts(self):
        cube = CubeDemo()

        self.assertTrue(cube.handle_keyboard(ord('1')))
        self.assertAlmostEqual(cube.yaw, 0.0)
        self.assertAlmostEqual(cube.pitch, 0.0)

        self.assertTrue(cube.handle_keyboard(ord('2')))
        self.assertAlmostEqual(cube.yaw, math.radians(-90.0))
        self.assertAlmostEqual(cube.pitch, 0.0)

        self.assertTrue(cube.handle_keyboard(ord('3')))
        self.assertAlmostEqual(cube.yaw, 0.0)
        self.assertAlmostEqual(cube.pitch, math.radians(90.0))

        cube.zoom = 2.0
        cube.pan = numpy.array([120.0, -80.0])
        self.assertTrue(cube.handle_keyboard(ord('f')))
        self.assertAlmostEqual(cube.zoom, 1.0)
        numpy.testing.assert_allclose(cube.pan, numpy.zeros(2))

    def test_gesture_updates_move_zoom_and_rotation(self):
        cube = CubeDemo()
        start = numpy.array([0.5, 0.5])

        cube.update(Hand.Pose.OPEN, start, 1.0, 0.8, now=1.0)
        cube.update(Hand.Pose.OPEN, numpy.array([0.6, 0.4]), 1.0, 0.8, now=1.1)
        numpy.testing.assert_allclose(cube.pan, numpy.array([62.0, -62.0]))

        cube.update(Hand.Pose.THUMB_SIDE, start, 1.0, 0.8, now=2.0)
        cube.update(Hand.Pose.THUMB_SIDE, numpy.array([0.5, 0.3]), 1.0, 0.8, now=2.1)
        self.assertGreater(cube.zoom, 1.0)

        yaw = cube.yaw
        pitch = cube.pitch
        cube.update(Hand.Pose.INDEX_MIDDLE_UP, start, 1.0, 0.8, now=3.0)
        cube.update(Hand.Pose.INDEX_MIDDLE_UP, numpy.array([0.6, 0.6]), 1.0, 0.8, now=3.1)
        self.assertNotEqual(cube.yaw, yaw)
        self.assertNotEqual(cube.pitch, pitch)

    def test_index_up_selects_view_zone_and_close_fits(self):
        cube = CubeDemo()

        cube.update(Hand.Pose.INDEX_UP, numpy.array([0.1, 0.5]), 1.0, 0.8, now=1.0)
        self.assertEqual(cube.last_action, 'Front view')

        cube.update(Hand.Pose.INDEX_UP, numpy.array([0.5, 0.5]), 1.0, 0.8, now=2.0)
        self.assertEqual(cube.last_action, 'Side view')

        cube.update(Hand.Pose.INDEX_UP, numpy.array([0.9, 0.5]), 1.0, 0.8, now=3.0)
        self.assertEqual(cube.last_action, 'Top view')

        cube.zoom = 2.0
        cube.pan = numpy.array([80.0, -30.0])
        cube.update(Hand.Pose.CLOSE, numpy.array([0.5, 0.5]), 1.0, 0.8, now=4.0)
        self.assertEqual(cube.last_action, 'Fit screen')
        self.assertAlmostEqual(cube.zoom, 1.0)
        numpy.testing.assert_allclose(cube.pan, numpy.zeros(2))


if __name__ == '__main__':
    unittest.main()
