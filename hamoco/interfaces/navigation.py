import enum
import time
from dataclasses import dataclass

import numpy

from hamoco import Hand


class CADCommand(enum.Enum):
    NONE = 'NONE'
    FRONT_VIEW = 'FRONT_VIEW'
    SIDE_VIEW = 'SIDE_VIEW'
    TOP_VIEW = 'TOP_VIEW'
    FIT_VIEW = 'FIT_VIEW'


@dataclass(frozen=True)
class CADNavigationVector:
    pan_x: float = 0.0
    pan_y: float = 0.0
    zoom_z: float = 0.0
    rotate_x: float = 0.0
    rotate_y: float = 0.0
    rotate_z: float = 0.0
    command: CADCommand = CADCommand.NONE
    source_pose: Hand.Pose = Hand.Pose.UNDEFINED
    view_zone: str = None
    active: bool = False
    pose_started: bool = False

    @property
    def has_motion(self):
        return any(
            abs(value) > 0.0
            for value in (
                self.pan_x,
                self.pan_y,
                self.zoom_z,
                self.rotate_x,
                self.rotate_y,
                self.rotate_z,
            )
        )

    @property
    def has_command(self):
        return self.command is not CADCommand.NONE


class CADGestureMapper:

    def __init__(self, command_cooldown=0.55):
        self.command_cooldown = command_cooldown
        self.active_pose = Hand.Pose.UNDEFINED
        self.anchor_center = None
        self.last_command_at = 0.0
        self.active_view_zone = None

    def reset_interaction(self):
        self.active_pose = Hand.Pose.UNDEFINED
        self.anchor_center = None
        self.active_view_zone = None

    def update(self, pose, palm_center, confidence, min_confidence, now=None):
        if now is None:
            now = time.perf_counter()
        if palm_center is None or confidence < min_confidence or pose == Hand.Pose.UNDEFINED:
            self.reset_interaction()
            return CADNavigationVector()

        palm_center = numpy.asarray(palm_center, dtype=float)
        pose_started = pose != self.active_pose or self.anchor_center is None
        if pose_started:
            self.active_pose = pose
            self.anchor_center = palm_center.copy()

        delta = palm_center - self.anchor_center
        self.active_view_zone = None

        if pose == Hand.Pose.OPEN:
            return CADNavigationVector(
                pan_x=float(delta[0]),
                pan_y=float(delta[1]),
                source_pose=pose,
                active=True,
                pose_started=pose_started,
            )
        if pose == Hand.Pose.THUMB_SIDE:
            return CADNavigationVector(
                zoom_z=float(self.anchor_center[1] - palm_center[1]),
                source_pose=pose,
                active=True,
                pose_started=pose_started,
            )
        if pose == Hand.Pose.INDEX_MIDDLE_UP:
            return CADNavigationVector(
                rotate_x=float(delta[1]),
                rotate_y=float(delta[0]),
                source_pose=pose,
                active=True,
                pose_started=pose_started,
            )
        if pose == Hand.Pose.INDEX_UP:
            self.active_view_zone = self.view_zone(palm_center)
            command = CADCommand.NONE
            if now - self.last_command_at >= self.command_cooldown:
                command = self.command_for_view_zone(self.active_view_zone)
                self.last_command_at = now
            return CADNavigationVector(
                command=command,
                source_pose=pose,
                view_zone=self.active_view_zone,
                active=True,
                pose_started=pose_started,
            )
        if pose == Hand.Pose.CLOSE:
            command = CADCommand.NONE
            if now - self.last_command_at >= self.command_cooldown:
                command = CADCommand.FIT_VIEW
                self.last_command_at = now
            return CADNavigationVector(
                command=command,
                source_pose=pose,
                active=True,
                pose_started=pose_started,
            )
        if pose == Hand.Pose.PINKY_UP:
            command = CADCommand.NONE
            if now - self.last_command_at >= self.command_cooldown:
                command = CADCommand.SIDE_VIEW
                self.last_command_at = now
            return CADNavigationVector(
                command=command,
                source_pose=pose,
                active=True,
                pose_started=pose_started,
            )

        return CADNavigationVector(source_pose=pose, active=True, pose_started=pose_started)

    def view_zone(self, palm_center):
        if palm_center[0] < 1.0 / 3.0:
            return 'FRONT'
        if palm_center[0] < 2.0 / 3.0:
            return 'SIDE'
        return 'TOP'

    def command_for_view_zone(self, zone):
        if zone == 'FRONT':
            return CADCommand.FRONT_VIEW
        if zone == 'SIDE':
            return CADCommand.SIDE_VIEW
        if zone == 'TOP':
            return CADCommand.TOP_VIEW
        return CADCommand.NONE
