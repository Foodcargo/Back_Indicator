
import argparse
from collections import Counter, deque
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

from auto_ptz import AutoPTZ
from onvif_controller import ONVIFController


mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

LEFT = [11, 13, 15, 23, 25, 27]
RIGHT = [12, 14, 16, 24, 26, 28]

VISIBILITY_THRESHOLD = 0.5


@dataclass
class ExerciseResult:
    exercise: str = "Unknown"
    stable_exercise: str = "Unknown"
    side: str = "Unknown"
    elbow_angle: float | None = None
    knee_angle: float | None = None
    landmarks: object | None = None
    ptz_status: str = "OFF"


class DigitalPTZ:
    def __init__(
        self,
        padding=2.2,
        min_crop_ratio=0.45,
        smoothing=0.18
    ):
        self.padding = padding
        self.min_crop_ratio = min_crop_ratio
        self.smoothing = smoothing

        self.center_x = None
        self.center_y = None
        self.crop_width = None

    def apply(self, frame, landmarks):

        if landmarks is None:
            return frame, "DIGITAL SEARCH"

        height, width = frame.shape[:2]

        visible = [
            item for item in landmarks
            if item.visibility >= VISIBILITY_THRESHOLD
        ]

        if len(visible) < 6:
            return frame, "DIGITAL LOW VIS"

        xs = np.array([item.x * width for item in visible])
        ys = np.array([item.y * height for item in visible])

        target_x = float(
            np.clip((xs.min() + xs.max()) / 2, 0, width)
        )
        target_y = float(
            np.clip((ys.min() + ys.max()) / 2, 0, height)
        )

        aspect = width / height

        body_width = xs.max() - xs.min()
        body_height = ys.max() - ys.min()

        target_crop_width = max(
            body_width * self.padding,
            body_height * aspect * self.padding,
            width * self.min_crop_ratio
        )

        target_crop_width = float(
            np.clip(
                target_crop_width,
                width * 0.35,
                width
            )
        )

        if self.center_x is None:
            self.center_x = target_x
            self.center_y = target_y
            self.crop_width = target_crop_width

        else:
            alpha = self.smoothing

            self.center_x = (
                self.center_x * (1 - alpha)
                + target_x * alpha
            )
            self.center_y = (
                self.center_y * (1 - alpha)
                + target_y * alpha
            )
            self.crop_width = (
                self.crop_width * (1 - alpha)
                + target_crop_width * alpha
            )

        crop_width = int(round(self.crop_width))
        crop_height = int(round(crop_width / aspect))

        if crop_width >= width or crop_height >= height:
            return frame, "DIGITAL FULL"

        x1 = int(round(self.center_x - crop_width / 2))
        y1 = int(round(self.center_y - crop_height / 2))

        x1 = max(0, min(x1, width - crop_width))
        y1 = max(0, min(y1, height - crop_height))

        cropped = frame[
            y1:y1 + crop_height,
            x1:x1 + crop_width
        ]

        tracked = cv2.resize(
            cropped,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

        zoom = width / crop_width

        return tracked, f"DIGITAL {zoom:.1f}x"


def angle_3p(a, b, c):
    """세 점의 2D 내각 계산: b가 꼭짓점"""

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    denom = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denom < 1e-8:
        return 180.0

    cosine = np.dot(ba, bc) / denom

    return float(
        np.degrees(
            np.arccos(np.clip(cosine, -1, 1))
        )
    )


def classify_exercise(landmarks, frame_width, frame_height):
    """
    운동 분류:
    1. Push-up
    2. Lunge
    3. Squat
    4. Unknown

    런지:
    - 양쪽 다리 인식
    - 뒤쪽 무릎 각도 < 150도
    - 뒤쪽 어깨-엉덩이-무릎 각도 < 45도
    """

    def visible(ids):
        return all(
            landmarks[i].visibility >= VISIBILITY_THRESHOLD
            for i in ids
        )

    def pt(i):
        return np.array([
            landmarks[i].x * frame_width,
            landmarks[i].y * frame_height
        ], dtype=float)

    left_score = np.mean([
        landmarks[i].visibility for i in LEFT
    ])
    right_score = np.mean([
        landmarks[i].visibility for i in RIGHT
    ])

    # 가시성이 높은 쪽 선택
    if left_score >= right_score:
        ids = LEFT
        side = "LEFT"
    else:
        ids = RIGHT
        side = "RIGHT"

    if not visible(ids):
        return "Unknown", side, None, None

    shoulder = pt(ids[0])
    elbow = pt(ids[1])
    wrist = pt(ids[2])
    hip = pt(ids[3])
    knee = pt(ids[4])
    ankle = pt(ids[5])

    elbow_angle = angle_3p(
        shoulder, elbow, wrist
    )

    knee_angle = angle_3p(
        hip, knee, ankle
    )

    dx = abs(shoulder[0] - hip[0])
    dy = abs(shoulder[1] - hip[1])

    torso_angle = np.degrees(
        np.arctan2(dy, dx + 1e-8)
    )

    torso_length = np.linalg.norm(
        shoulder - hip
    )

    if torso_length < 1e-6:
        return (
            "Unknown",
            side,
            elbow_angle,
            knee_angle
        )

    # 1. Push-up
    if (
        torso_angle < 35
        and wrist[1] > shoulder[1]
    ):
        return (
            "Push-up",
            side,
            elbow_angle,
            knee_angle
        )

    # 2. Lunge
    # 양쪽 다리 모두 인식되어야 함
    both_legs_visible = visible([
        23, 25, 27,
        24, 26, 28,
        11, 12
    ])

    if both_legs_visible:

        left_shoulder = pt(11)
        right_shoulder = pt(12)

        left_hip = pt(23)
        right_hip = pt(24)

        left_knee = pt(25)
        right_knee = pt(26)

        left_ankle = pt(27)
        right_ankle = pt(28)

        left_knee_angle = angle_3p(
            left_hip,
            left_knee,
            left_ankle
        )

        right_knee_angle = angle_3p(
            right_hip,
            right_knee,
            right_ankle
        )

        # 어깨-엉덩이-무릎 각도
        left_hip_angle = angle_3p(
            left_shoulder,
            left_hip,
            left_knee
        )

        right_hip_angle = angle_3p(
            right_shoulder,
            right_hip,
            right_knee
        )

        # 엉덩이 기준으로 발목이 더 뒤에 있는
        # 다리를 뒤쪽 다리로 판단
        # 2D 영상에서는 전후 방향을 확정할 수
        # 없으므로 두 다리의 수평 위치 차이와
        # 엉덩이 각도를 함께 활용
        ankle_gap = abs(
            left_ankle[0] - right_ankle[0]
        ) / torso_length

        # 뒤로 뻗은 다리는 보통 엉덩이 각도가
        # 더 작다는 기하학적 특징을 활용
        if left_hip_angle < right_hip_angle:
            rear_side = "LEFT"
            rear_knee_angle = left_knee_angle
            rear_hip_angle = left_hip_angle
        else:
            rear_side = "RIGHT"
            rear_knee_angle = right_knee_angle
            rear_hip_angle = right_hip_angle

        if (
            torso_angle > 45
            and ankle_gap > 0.45
            and rear_knee_angle < 150
            and rear_hip_angle < 45
        ):
            return (
                "Lunge",
                rear_side,
                elbow_angle,
                rear_knee_angle
            )

    # 3. Squat
    if (
        torso_angle > 45
        and knee_angle < 150
    ):
        return (
            "Squat",
            side,
            elbow_angle,
            knee_angle
        )

    return (
        "Unknown",
        side,
        elbow_angle,
        knee_angle
    )


def draw_exercise_info(frame, result):
    """운동 분류 결과 표시"""

    cv2.putText(
        frame,
        f"Exercise: {result.exercise}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Side: {result.side}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    if result.elbow_angle is not None:
        cv2.putText(
            frame,
            f"Elbow: {result.elbow_angle:.1f} deg",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    if result.knee_angle is not None:
        cv2.putText(
            frame,
            f"Knee: {result.knee_angle:.1f} deg",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    cv2.putText(
        frame,
        f"PTZ: {result.ptz_status}",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default="0",
        help="Camera index or RTSP URL"
    )

    parser.add_argument(
        "--digital-ptz",
        action="store_true"
    )

    args = parser.parse_args()

    source = (
        int(args.source)
        if args.source.isdigit()
        else args.source
    )

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera: {source}"
        )

    digital_ptz = DigitalPTZ()

    history = deque(maxlen=7)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = pose.process(rgb)

            exercise_result = ExerciseResult()

            if results.pose_landmarks:

                landmarks = (
                    results.pose_landmarks.landmark
                )

                exercise, side, elbow, knee = (
                    classify_exercise(
                        landmarks,
                        width,
                        height
                    )
                )

                history.append(exercise)

                stable_exercise = Counter(
                    history
                ).most_common(1)[0][0]

                exercise_result.exercise = exercise
                exercise_result.stable_exercise = (
                    stable_exercise
                )
                exercise_result.side = side
                exercise_result.elbow_angle = elbow
                exercise_result.knee_angle = knee
                exercise_result.landmarks = landmarks

                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                if args.digital_ptz:
                    frame, ptz_status = digital_ptz.apply(
                        frame,
                        landmarks
                    )
                    exercise_result.ptz_status = ptz_status

            else:
                history.clear()

            # 안정화된 운동 종류 표시
            exercise_result.exercise = (
                exercise_result.stable_exercise
            )

            draw_exercise_info(
                frame,
                exercise_result
            )

            cv2.imshow(
                "Exercise Recognition",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


class ExerciseAnalyzer:
    def __init__(self, auto_ptz=None, digital_ptz=False):
        self.auto_ptz = auto_ptz
        self.digital_ptz = DigitalPTZ() if digital_ptz else None
        self.history = deque(maxlen=7)
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self):
        self.pose.close()

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        rgb.flags.writeable = True

        result = ExerciseResult()

        if results.pose_landmarks:
            result.landmarks = results.pose_landmarks.landmark
            (
                result.exercise,
                result.side,
                result.elbow_angle,
                result.knee_angle,
            ) = classify_exercise(result.landmarks, width, height)

            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
            )

        self.history.append(result.exercise)
        result.stable_exercise = Counter(self.history).most_common(1)[0][0]

        if self.auto_ptz:
            result.ptz_status = self.auto_ptz.update(result.landmarks)

        if self.digital_ptz:
            frame, result.ptz_status = self.digital_ptz.apply(frame, result.landmarks)

        draw_overlay(frame, result)
        return frame, result


def draw_overlay(frame, result):
    cv2.rectangle(frame, (0, 0), (470, 215), (25, 25, 25), -1)

    lines = [
        ("Exercise", result.stable_exercise, (0, 255, 0)),
        ("Detected Side", result.side, (255, 255, 0)),
        (
            "Elbow Angle",
            f"{result.elbow_angle:.1f}" if result.elbow_angle is not None else "N/A",
            (255, 255, 255),
        ),
        (
            "Knee Angle",
            f"{result.knee_angle:.1f}" if result.knee_angle is not None else "N/A",
            (255, 255, 255),
        ),
        ("PTZ", result.ptz_status, (0, 200, 255)),
    ]

    for index, (label, value, color) in enumerate(lines):
        cv2.putText(
            frame,
            f"{label}: {value}",
            (20, 40 + index * 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )


def create_auto_ptz(enabled, ip, port, username, password):
    if not enabled:
        return None

    controller = ONVIFController(
        ip=ip,
        port=port,
        username=username,
        password=password,
    )
    return AutoPTZ(controller)


def open_camera(camera_index):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if cap.isOpened():
        return cap

    cap.release()
    return cv2.VideoCapture(camera_index)


def run_camera(camera_index=0, auto_ptz=None, digital_ptz=False):
    print(f"Opening camera index {camera_index}...")
    cap = open_camera(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Camera not found: index {camera_index}")

    analyzer = ExerciseAnalyzer(auto_ptz=auto_ptz, digital_ptz=digital_ptz)

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame, _ = analyzer.process_frame(frame)
            cv2.imshow("Exercise Classification", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        analyzer.close()
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=2)
    parser.add_argument("--digital-ptz", action="store_true")

    parser.add_argument("--ptz", action="store_true")
    parser.add_argument("--ptz-ip", default="192.168.0.20")
    parser.add_argument("--ptz-port", type=int, default=80)
    parser.add_argument("--ptz-user", default="admin")
    parser.add_argument("--ptz-password", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    auto_ptz = create_auto_ptz(
        enabled=args.ptz,
        ip=args.ptz_ip,
        port=args.ptz_port,
        username=args.ptz_user,
        password=args.ptz_password,
    )
    run_camera(
        camera_index=args.camera,
        auto_ptz=auto_ptz,
        digital_ptz=args.digital_ptz,
    )


if __name__ == "__main__":
    main()
