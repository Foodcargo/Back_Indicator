
import time

class AutoPTZ:
    def __init__(self, controller):
        self.controller = controller
        self.last_move = 0
        self.cooldown = 1.0

    def update(self, landmarks):
        now = time.monotonic()

        if now - self.last_move < self.cooldown:
            return "WAIT"

        if landmarks is None:
            # 전신이 안 보이면 우선 축소
            self.controller.move(
                zoom=-0.2,
                duration=0.2
            )
            self.last_move = now
            return "ZOOM OUT"

        ids = [11, 12, 23, 24, 25, 26, 27, 28]

        points = [landmarks[i] for i in ids]

        visible = [
            p for p in points
            if p.visibility >= 0.6
        ]

        if len(visible) < 6:
            self.controller.move(
                zoom=-0.2,
                duration=0.2
            )
            self.last_move = now
            return "LOW VISIBILITY"

        xs = [p.x for p in visible]
        ys = [p.y for p in visible]

        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2

        pan = 0.0
        tilt = 0.0

        if cx < 0.35:
            pan = -0.15
        elif cx > 0.65:
            pan = 0.15

        if cy < 0.35:
            tilt = 0.15
        elif cy > 0.65:
            tilt = -0.15

        if pan != 0 or tilt != 0:
            self.controller.move(
                pan=pan,
                tilt=tilt,
                duration=0.15
            )
            self.last_move = now
            return "PAN/TILT"

        body_height = max(ys) - min(ys)

        if body_height < 0.5:
            self.controller.move(
                zoom=0.15,
                duration=0.15
            )
            self.last_move = now
            return "ZOOM IN"

        return "STABLE"
