#!/usr/bin/env python

import argparse
import atexit
import json
import time

import cv2
import keras
import mediapipe as mp
import numpy

from hamoco import Hand, HandyMouseController
from hamoco.config import __default_config__
from hamoco.models import __default_model__
from hamoco.utils import draw_control_bounds, draw_hand_landmarks
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
    last_frame_at = time.perf_counter()
    fps = 0.0
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print('# Hamoco test window is open.')
    print('# Press ESC or q in the preview window to exit.')
    print('# Press c in the preview window to toggle mouse control.')

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

            draw_status_bar(image, pose, confidence, fps, control_enabled, inferred=inferred_pose == pose)
            draw_probability_panel(image, display_probabilities, active_pose=pose, inferred_pose=inferred_pose)
            cv2.imshow(WINDOW_NAME, image)

            key = cv2.waitKey(5) & 0xFF
            if key in (27, ord('q')):
                break
            if key == ord('c'):
                control_enabled = not control_enabled
                if not control_enabled:
                    hand_controller.release_controls()

    hand_controller.release_controls()
    capture.release()
    cv2.destroyWindow(WINDOW_NAME)


if __name__ == '__main__':
    main()
