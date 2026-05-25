#!/usr/bin/env python

import argparse
import atexit
import json
import math
import time

import cv2
import keras
import mediapipe as mp
import numpy

from hamoco import Hand, HandyMouseController
from hamoco.config import __default_config__
from hamoco.interfaces import CADCommand, CADGestureMapper
from hamoco.models import __default_model__
from hamoco.utils import clamp, draw_control_bounds, draw_hand_landmarks
from hamoco.utils import draw_palm_center, draw_scrolling_origin


mp_hands = mp.solutions.hands

WINDOW_NAME = 'Hamoco test window'
POSE_COLORS = {
    Hand.Pose.OPEN: (58, 195, 102),
    Hand.Pose.CLOSE: (238, 238, 238),
    Hand.Pose.INDEX_UP: (255, 183, 77),
    Hand.Pose.PINKY_UP: (80, 157, 255),
    Hand.Pose.THUMB_SIDE: (196, 118, 255),
    Hand.Pose.INDEX_MIDDLE_UP: (255, 122, 122),
    Hand.Pose.UNDEFINED: (190, 190, 190),
}


class CubeDemo:

    vertices = numpy.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    faces = [
        ((0, 1, 2, 3), (70, 132, 224)),
        ((4, 5, 6, 7), (76, 190, 110)),
        ((0, 1, 5, 4), (236, 180, 82)),
        ((2, 3, 7, 6), (214, 103, 103)),
        ((1, 2, 6, 5), (168, 112, 226)),
        ((0, 3, 7, 4), (82, 175, 214)),
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    def __init__(self):
        self.yaw = math.radians(-34.0)
        self.pitch = math.radians(24.0)
        self.roll = 0.0
        self.zoom = 1.0
        self.pan = numpy.zeros(2, dtype=float)
        self.anchor_pan = self.pan.copy()
        self.anchor_zoom = self.zoom
        self.anchor_yaw = self.yaw
        self.anchor_pitch = self.pitch
        self.last_action = 'Ready'
        self.active_view_zone = None
        self.gesture_mapper = CADGestureMapper()

    def reset_interaction(self):
        self.gesture_mapper.reset_interaction()
        self.active_view_zone = None

    def begin_navigation(self):
        self.anchor_pan = self.pan.copy()
        self.anchor_zoom = self.zoom
        self.anchor_yaw = self.yaw
        self.anchor_pitch = self.pitch

    def apply_front_view(self):
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.last_action = 'Front view'

    def apply_side_view(self):
        self.yaw = math.radians(-90.0)
        self.pitch = 0.0
        self.roll = 0.0
        self.last_action = 'Side view'

    def apply_top_view(self):
        self.yaw = 0.0
        self.pitch = math.radians(90.0)
        self.roll = 0.0
        self.last_action = 'Top view'

    def fit_screen(self):
        self.zoom = 1.0
        self.pan = numpy.zeros(2, dtype=float)
        self.last_action = 'Fit screen'

    def apply_isometric_view(self):
        self.yaw = math.radians(-34.0)
        self.pitch = math.radians(24.0)
        self.roll = 0.0
        self.last_action = 'Isometric view'

    def handle_keyboard(self, key):
        if key == ord('1'):
            self.apply_front_view()
            return True
        if key == ord('2'):
            self.apply_side_view()
            return True
        if key == ord('3'):
            self.apply_top_view()
            return True
        if key == ord('f'):
            self.fit_screen()
            return True
        if key == ord('r'):
            self.apply_isometric_view()
            self.fit_screen()
            self.last_action = 'Reset view'
            return True
        return False

    def update(self, pose, palm_center, confidence, min_confidence, now=None):
        navigation = self.gesture_mapper.update(pose, palm_center, confidence, min_confidence, now=now)
        self.active_view_zone = navigation.view_zone
        if not navigation.active:
            return

        if navigation.pose_started:
            self.begin_navigation()

        if navigation.source_pose == Hand.Pose.OPEN:
            self.pan = self.anchor_pan + numpy.array([navigation.pan_x, navigation.pan_y]) * 620.0
            self.last_action = 'Move'
        elif navigation.source_pose == Hand.Pose.THUMB_SIDE:
            zoom_factor = 1.0 + navigation.zoom_z * 3.2
            self.zoom = clamp(self.anchor_zoom * zoom_factor, 0.35, 3.4)
            self.last_action = 'Zoom'
        elif navigation.source_pose == Hand.Pose.INDEX_MIDDLE_UP:
            self.yaw = self.anchor_yaw + navigation.rotate_y * math.tau * 1.35
            self.pitch = clamp(self.anchor_pitch + navigation.rotate_x * math.tau * 0.95, -1.48, 1.48)
            self.last_action = 'Rotation'
        if navigation.command is CADCommand.FRONT_VIEW:
            self.apply_front_view()
        elif navigation.command is CADCommand.SIDE_VIEW:
            self.apply_side_view()
        elif navigation.command is CADCommand.TOP_VIEW:
            self.apply_top_view()
        elif navigation.command is CADCommand.FIT_VIEW:
            self.fit_screen()

    def rotation_matrix(self):
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        cr = math.cos(self.roll)
        sr = math.sin(self.roll)
        yaw = numpy.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        pitch = numpy.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
        roll = numpy.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
        return roll @ pitch @ yaw

    def project(self, width, height):
        rotated = self.vertices @ self.rotation_matrix().T
        depth = rotated[:, 2] + 4.2
        focal = min(width, height) * 1.28 * self.zoom
        center = numpy.array([width / 2.0, height / 2.0]) + self.pan
        projected = numpy.column_stack(
            [
                center[0] + rotated[:, 0] * focal / depth,
                center[1] - rotated[:, 1] * focal / depth,
            ]
        )
        return projected.astype(numpy.int32), rotated

    def render(self, height, width, pose, confidence):
        image = numpy.zeros((height, width, 3), dtype=numpy.uint8)
        image[:] = (36, 39, 45)
        cv2.rectangle(image, (0, 0), (width - 1, height - 1), (70, 76, 86), 1)

        projected, rotated = self.project(width, height)
        face_order = sorted(
            range(len(self.faces)),
            key=lambda index: numpy.mean(rotated[list(self.faces[index][0]), 2]),
        )
        for face_index in face_order:
            indices, color = self.faces[face_index]
            polygon = projected[list(indices)]
            cv2.fillConvexPoly(image, polygon, color)
            cv2.polylines(image, [polygon], True, (245, 247, 250), 1, cv2.LINE_AA)

        for edge_start, edge_end in self.edges:
            cv2.line(image, tuple(projected[edge_start]), tuple(projected[edge_end]), (28, 31, 36), 2, cv2.LINE_AA)

        self.draw_view_selector(image)
        self.draw_cube_status(image, pose, confidence)
        return image

    def draw_view_selector(self, image):
        height, width = image.shape[:2]
        y0 = 12
        y1 = 47
        labels = ['FRONT', 'SIDE', 'TOP']
        for index, label in enumerate(labels):
            x0 = int(index * width / 3)
            x1 = int((index + 1) * width / 3)
            active = self.active_view_zone == label
            color = (66, 98, 148) if active else (48, 53, 61)
            cv2.rectangle(image, (x0 + 8, y0), (x1 - 8, y1), color, -1)
            cv2.rectangle(image, (x0 + 8, y0), (x1 - 8, y1), (95, 104, 118), 1)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
            text_x = x0 + (x1 - x0 - text_size[0]) // 2
            draw_text(image, label, (text_x, y0 + 23), scale=0.48, color=(238, 240, 244))

        draw_text(image, 'INDEX_UP selects view zone', (14, y1 + 24), scale=0.46, color=(210, 214, 220))

    def draw_cube_status(self, image, pose, confidence):
        height, width = image.shape[:2]
        panel_y = height - 96
        overlay = image.copy()
        cv2.rectangle(overlay, (0, panel_y), (width, height), (28, 31, 36), -1)
        cv2.addWeighted(overlay, 0.76, image, 0.24, 0, image)

        pose_name = pose.name.replace('_', ' ')
        color = POSE_COLORS.get(pose, POSE_COLORS[Hand.Pose.UNDEFINED])
        draw_text(image, '3D CUBE TEST', (16, panel_y + 25), scale=0.62, color=(245, 247, 250), thickness=2)
        draw_text(image, self.last_action, (16, panel_y + 52), scale=0.58, color=color, thickness=2)
        draw_text(image, f'{pose_name} {confidence * 100:4.0f}%', (16, panel_y + 78), scale=0.46, color=(220, 224, 230))

        draw_text(image, 'OPEN move  THUMB zoom  INDEX+MIDDLE rotate', (width - 374, panel_y + 31), scale=0.42, color=(220, 224, 230))
        draw_text(image, 'CLOSE fit  1/2/3 views  f fit  r reset', (width - 374, panel_y + 57), scale=0.42, color=(220, 224, 230))


def optional_modifier(value):
    if value is None:
        return None
    if value.lower() in ('', 'none', 'null'):
        return None
    return value


def draw_text(image, text, origin, scale=0.55, color=(255, 255, 255), thickness=1):
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_status_bar(image, pose, confidence, fps, control_enabled, inferred=False):
    height, width = image.shape[:2]
    color = POSE_COLORS.get(pose, POSE_COLORS[Hand.Pose.UNDEFINED])
    panel_height = 76
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (28, 31, 36), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)

    pose_name = pose.name.replace('_', ' ')
    draw_text(image, pose_name, (18, 31), scale=0.78, color=color, thickness=2)
    source = 'landmarks' if inferred else 'model'
    draw_text(image, f'{source} confidence {confidence * 100:5.1f}%', (18, 60), color=(235, 235, 235))

    control_label = 'CONTROL ON' if control_enabled else 'PREVIEW'
    control_color = (94, 220, 117) if control_enabled else (220, 220, 220)
    text_size = cv2.getTextSize(control_label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0]
    draw_text(image, control_label, (width - text_size[0] - 18, 31), scale=0.62, color=control_color, thickness=2)
    draw_text(image, f'{fps:4.1f} fps', (width - 94, 60), color=(220, 220, 220))


def draw_probability_panel(image, probabilities, active_pose=None, inferred_pose=None):
    if probabilities is None:
        return

    height, width = image.shape[:2]
    panel_width = min(300, max(230, width // 3))
    row_height = 25
    panel_height = 26 + row_height * len(Hand.Pose)
    x0 = width - panel_width - 14
    y0 = height - panel_height - 14

    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (width - 14, height - 14), (26, 29, 34), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)
    draw_text(image, 'POSE SCORES', (x0 + 12, y0 + 19), scale=0.48, color=(230, 230, 230))

    poses = [pose for pose in Hand.Pose if pose is not Hand.Pose.UNDEFINED]
    for index, pose in enumerate(poses):
        probability = float(probabilities[pose.value])
        active = pose == active_pose
        if active and pose == inferred_pose:
            probability = 1.0
        y = y0 + 42 + index * row_height
        bar_x = x0 + 128
        bar_width = panel_width - 148
        color = POSE_COLORS.get(pose, (220, 220, 220))
        if active:
            cv2.rectangle(image, (x0 + 6, y - 15), (width - 20, y + 11), (48, 54, 62), -1)
            cv2.circle(image, (x0 + 11, y - 2), 4, color, -1)
        label_color = color if active else (230, 230, 230)
        draw_text(image, pose.name, (x0 + 20, y + 5), scale=0.40, color=label_color)
        cv2.rectangle(image, (bar_x, y - 10), (bar_x + bar_width, y + 6), (70, 75, 84), -1)
        cv2.rectangle(image, (bar_x, y - 10), (bar_x + int(bar_width * probability), y + 6), color, -1)
        if active:
            cv2.rectangle(image, (bar_x, y - 10), (bar_x + bar_width, y + 6), color, 1)


def build_parser(default_config):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.description = (
        'Open a camera preview for testing Hamoco hand tracking and pose prediction. '
        'Mouse control is disabled unless --control is passed.'
    )
    parser.add_argument('-c', '--camera', type=int, default=0, help='Camera device index')
    parser.add_argument('--width', type=int, default=960, help='Requested capture width')
    parser.add_argument('--height', type=int, default=540, help='Requested capture height')
    parser.add_argument('--cube_width', type=int, default=560, help='Width of the 3D cube test panel')
    parser.add_argument('--control', action='store_true', help='Enable mouse control while previewing')
    parser.add_argument('-m', '--model', default=default_config['model'], type=str, help='Path to the Keras model')
    parser.add_argument(
        '--minimum_prediction_confidence',
        default=default_config['minimum_prediction_confidence'],
        type=float,
        help='Minimum prediction confidence before mouse control is applied',
    )
    parser.add_argument('-S', '--sensitivity', default=default_config['sensitivity'], type=float)
    parser.add_argument('-M', '--margin', default=default_config['margin'], type=float)
    parser.add_argument('--scrolling_threshold', default=default_config['scrolling_threshold'], type=float)
    parser.add_argument('--scrolling_speed', default=default_config['scrolling_speed'], type=float)
    parser.add_argument('--drag_modifier', default=default_config['drag_modifier'], type=optional_modifier)
    parser.add_argument('--drag_button', default=default_config['drag_button'], type=str)
    parser.add_argument('--min_cutoff_filter', default=default_config['min_cutoff_filter'], type=float)
    parser.add_argument('--beta_filter', default=default_config['beta_filter'], type=float)
    return parser


def main():
    with open(__default_config__) as cfg:
        default_config = json.load(cfg)

    parser = build_parser(default_config)
    args = parser.parse_args()
    model_path = __default_model__ if args.model is None else args.model
    cube_width = max(420, args.cube_width)
    trained_model = keras.models.load_model(model_path)

    hand_controller = HandyMouseController(
        sensitivity=args.sensitivity,
        margin=args.margin,
        scrolling_threshold=args.scrolling_threshold,
        scrolling_speed=args.scrolling_speed,
        min_cutoff_filter=args.min_cutoff_filter,
        beta_filter=args.beta_filter,
        drag_modifier=args.drag_modifier,
        drag_button=args.drag_button,
    )
    atexit.register(hand_controller.release_controls)

    capture = cv2.VideoCapture(args.camera)
    if args.width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not capture.isOpened():
        raise SystemExit(f'Unable to open camera {args.camera}')

    control_enabled = args.control
    cube_demo = CubeDemo()
    last_frame_at = time.perf_counter()
    fps = 0.0
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print('# Hamoco test window is open.')
    print('# Press ESC or q in the preview window to exit.')
    print('# Press c in the preview window to toggle mouse control.')
    print('# Cube test: OPEN move, THUMB_SIDE zoom, INDEX_MIDDLE_UP rotate, INDEX_UP front/side/top zones, CLOSE fit.')
    print('# Keyboard shortcuts: 1 front, 2 side, 3 top, f fit, r reset.')

    with mp_hands.Hands(
        static_image_mode=False,
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while capture.isOpened():
            success, image = capture.read()
            if not success:
                print('Ignoring empty camera frame.')
                continue

            now = time.perf_counter()
            elapsed = now - last_frame_at
            last_frame_at = now
            if elapsed > 0:
                current_fps = 1.0 / elapsed
                fps = current_fps if fps == 0.0 else (0.9 * fps + 0.1 * current_fps)

            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_image)
            hand_detected = bool(results.multi_hand_landmarks)

            pose = Hand.Pose.UNDEFINED
            confidence = 0.0
            probabilities = None
            display_probabilities = None
            inferred_pose = None
            palm_center = None
            draw_control_bounds(image, hand_controller.accessible_area(image))

            if hand_detected:
                draw_hand_landmarks(image, results.multi_hand_landmarks[0])
                landmarks = results.multi_hand_landmarks[0].landmark
                hand = Hand()
                raw_landmark_vector = hand.vectorize_landmarks(landmarks)
                palm_center = hand_controller.palm_center(raw_landmark_vector)
                processed_landmark_vector = hand.feature_process_landmarks(raw_landmark_vector)

                probabilities = trained_model.predict(processed_landmark_vector, verbose=0).flatten()
                display_probabilities = probabilities.copy()
                confidence = float(numpy.max(probabilities))
                pose = Hand.Pose(int(numpy.argmax(probabilities)))
                inferred_pose = Hand.infer_pose_from_landmarks(landmarks)
                if inferred_pose in (Hand.Pose.CLOSE, Hand.Pose.THUMB_SIDE):
                    pose = inferred_pose
                    confidence = max(confidence, 0.95)
                    display_probabilities[inferred_pose.value] = 1.0
                hand.pose = pose

                draw_palm_center(image, palm_center, size=0.03)
                if control_enabled and confidence >= args.minimum_prediction_confidence:
                    hand_controller.operate_mouse(
                        hand,
                        palm_center,
                        confidence,
                        min_confidence=args.minimum_prediction_confidence,
                    )
                else:
                    hand_controller.release_controls()
            else:
                hand_controller.release_controls()

            if hand_controller.current_mouse_state == HandyMouseController.MouseState.SCROLLING:
                draw_scrolling_origin(
                    image,
                    hand_controller.scrolling_origin,
                    hand_controller.scrolling_threshold,
                )

            cube_demo.update(pose, palm_center, confidence, args.minimum_prediction_confidence, now=now)
            draw_status_bar(image, pose, confidence, fps, control_enabled, inferred=inferred_pose == pose)
            draw_probability_panel(image, display_probabilities, active_pose=pose, inferred_pose=inferred_pose)
            cube_image = cube_demo.render(image.shape[0], cube_width, pose, confidence)
            cv2.imshow(WINDOW_NAME, numpy.hstack([image, cube_image]))

            key = cv2.waitKey(5) & 0xFF
            if key in (27, ord('q')):
                break
            if key == ord('c'):
                control_enabled = not control_enabled
                if not control_enabled:
                    hand_controller.release_controls()
            else:
                cube_demo.handle_keyboard(key)

    hand_controller.release_controls()
    capture.release()
    cv2.destroyWindow(WINDOW_NAME)


if __name__ == '__main__':
    main()
