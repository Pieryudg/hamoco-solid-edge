#!/usr/bin/env python

import unittest
from unittest.mock import Mock

import numpy

from hamoco import Hand
from hamoco.interfaces import (
    CADCommand,
    CADGestureMapper,
    SolidEdgeCommandClient,
    SolidEdgeHybridAdapter,
)


class TestCADInterfaces(unittest.TestCase):

    def test_gesture_mapper_creates_spacemouse_style_vectors(self):
        mapper = CADGestureMapper()
        anchor = numpy.array([0.5, 0.5])

        first = mapper.update(Hand.Pose.OPEN, anchor, 1.0, 0.8, now=1.0)
        self.assertTrue(first.pose_started)
        self.assertEqual(first.source_pose, Hand.Pose.OPEN)

        pan = mapper.update(Hand.Pose.OPEN, numpy.array([0.6, 0.4]), 1.0, 0.8, now=1.1)
        self.assertAlmostEqual(pan.pan_x, 0.1)
        self.assertAlmostEqual(pan.pan_y, -0.1)

        zoom = mapper.update(Hand.Pose.THUMB_SIDE, anchor, 1.0, 0.8, now=2.0)
        self.assertTrue(zoom.pose_started)
        zoom = mapper.update(Hand.Pose.THUMB_SIDE, numpy.array([0.5, 0.3]), 1.0, 0.8, now=2.1)
        self.assertAlmostEqual(zoom.zoom_z, 0.2)

        rotate = mapper.update(Hand.Pose.INDEX_MIDDLE_UP, anchor, 1.0, 0.8, now=3.0)
        self.assertTrue(rotate.pose_started)
        rotate = mapper.update(Hand.Pose.INDEX_MIDDLE_UP, numpy.array([0.6, 0.7]), 1.0, 0.8, now=3.1)
        self.assertAlmostEqual(rotate.rotate_y, 0.1)
        self.assertAlmostEqual(rotate.rotate_x, 0.2)

    def test_gesture_mapper_creates_discrete_view_commands(self):
        mapper = CADGestureMapper(command_cooldown=0.5)

        front = mapper.update(Hand.Pose.INDEX_UP, numpy.array([0.1, 0.5]), 1.0, 0.8, now=1.0)
        self.assertEqual(front.command, CADCommand.FRONT_VIEW)
        self.assertEqual(front.view_zone, 'FRONT')

        blocked = mapper.update(Hand.Pose.INDEX_UP, numpy.array([0.9, 0.5]), 1.0, 0.8, now=1.2)
        self.assertEqual(blocked.command, CADCommand.NONE)
        self.assertEqual(blocked.view_zone, 'TOP')

        top = mapper.update(Hand.Pose.INDEX_UP, numpy.array([0.9, 0.5]), 1.0, 0.8, now=1.6)
        self.assertEqual(top.command, CADCommand.TOP_VIEW)

        fit = mapper.update(Hand.Pose.CLOSE, numpy.array([0.5, 0.5]), 1.0, 0.8, now=2.2)
        self.assertEqual(fit.command, CADCommand.FIT_VIEW)

    def test_command_client_starts_configured_command(self):
        client = SolidEdgeCommandClient(command_ids={'FIT_VIEW': 44001}, logger=None, connect=False)
        client.application = Mock()
        client.available = True

        self.assertTrue(client.start_command(CADCommand.FIT_VIEW))
        client.application.StartCommand.assert_called_once_with(44001)

    def test_command_client_reports_missing_command(self):
        logger = Mock()
        client = SolidEdgeCommandClient(command_ids={}, logger=logger, connect=False)

        self.assertFalse(client.start_command(CADCommand.TOP_VIEW))
        logger.assert_called_once()

    def test_hybrid_adapter_routes_discrete_commands_without_mouse_clicks(self):
        controller = Mock()
        command_client = Mock()
        adapter = SolidEdgeHybridAdapter(
            controller,
            command_client=command_client,
            gesture_mapper=CADGestureMapper(command_cooldown=0.0),
        )
        hand = Hand(Hand.Pose.INDEX_UP)

        adapter.operate(hand, numpy.array([0.2, 0.5]), 1.0, 0.8)

        command_client.start_command.assert_called_once_with(CADCommand.FRONT_VIEW)
        controller.operate_mouse.assert_not_called()
        controller.release_controls.assert_called_once()


if __name__ == '__main__':
    unittest.main()
