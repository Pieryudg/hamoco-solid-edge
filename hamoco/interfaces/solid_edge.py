import os

from hamoco import Hand

from .navigation import CADCommand, CADGestureMapper


class SolidEdgeCommandClient:

    def __init__(self, command_ids=None, logger=print, connect=True):
        self.command_ids = command_ids or {}
        self.logger = logger
        self.application = None
        self.available = False
        self._reported = set()
        if connect:
            self.connect()

    def connect(self):
        if os.name != 'nt':
            self._report_once('platform', '# hamoco: Solid Edge COM interface is only available on Windows.')
            return False
        try:
            import win32com.client
        except ImportError:
            self._report_once('pywin32', '# hamoco: pywin32 is not installed; Solid Edge COM commands are disabled.')
            return False

        try:
            self.application = win32com.client.GetActiveObject('SolidEdge.Application')
        except Exception:
            try:
                self.application = win32com.client.Dispatch('SolidEdge.Application')
            except Exception as exc:
                self._report_once('connect', f'# hamoco: unable to connect to Solid Edge COM: {exc}')
                return False

        self.available = True
        self._report_once('connected', '# hamoco: connected to Solid Edge COM interface.')
        return True

    def start_command(self, command):
        if command is CADCommand.NONE:
            return True

        command_id = self.command_id(command)
        if command_id is None:
            self._report_once(
                f'missing:{command.name}',
                f'# hamoco: no Solid Edge command id configured for {command.name}.',
            )
            return False

        if not self.available and not self.connect():
            return False

        try:
            self.application.StartCommand(int(command_id))
            return True
        except Exception as exc:
            self._report_once(command.name, f'# hamoco: Solid Edge StartCommand failed for {command.name}: {exc}')
            return False

    def command_id(self, command):
        if command.name in self.command_ids:
            return self.command_ids[command.name]
        if command.value in self.command_ids:
            return self.command_ids[command.value]
        return None

    def _report_once(self, key, message):
        if key in self._reported:
            return
        self._reported.add(key)
        if self.logger:
            self.logger(message)


class SolidEdgeHybridAdapter:

    continuous_poses = {
        Hand.Pose.OPEN,
        Hand.Pose.THUMB_SIDE,
        Hand.Pose.INDEX_MIDDLE_UP,
    }

    def __init__(self, controller, command_client=None, gesture_mapper=None):
        self.controller = controller
        self.command_client = command_client or SolidEdgeCommandClient()
        self.gesture_mapper = gesture_mapper or CADGestureMapper()

    def operate(self, hand, palm_center, confidence, min_confidence):
        navigation = self.gesture_mapper.update(hand.pose, palm_center, confidence, min_confidence)
        if navigation.has_command:
            self.command_client.start_command(navigation.command)

        if hand.pose in self.continuous_poses:
            self.controller.operate_mouse(hand, palm_center, confidence, min_confidence=min_confidence)
        else:
            self.controller.release_controls()

    def release_controls(self):
        self.gesture_mapper.reset_interaction()
        self.controller.release_controls()
