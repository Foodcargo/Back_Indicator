import time

from onvif import ONVIFCamera


class ONVIFController:
    def __init__(self, ip, port=80, username="admin", password=""):
        self.camera = ONVIFCamera(ip, port, username, password)

        media = self.camera.create_media_service()
        self.ptz = self.camera.create_ptz_service()

        profiles = media.GetProfiles()
        profile = next(
            (
                item
                for item in profiles
                if getattr(item, "PTZConfiguration", None)
            ),
            None,
        )

        if profile is None:
            raise RuntimeError(
                "No ONVIF profile with PTZ configuration was found."
            )

        self.token = profile.token

    def move(self, pan=0.0, tilt=0.0, zoom=0.0, duration=0.2):
        request = self.ptz.create_type("ContinuousMove")
        request.ProfileToken = self.token
        request.Velocity = {
            "PanTilt": {
                "x": pan,
                "y": tilt,
            },
            "Zoom": {
                "x": zoom,
            },
        }

        try:
            self.ptz.ContinuousMove(request)
            time.sleep(duration)
        finally:
            self.ptz.Stop(
                {
                    "ProfileToken": self.token,
                    "PanTilt": True,
                    "Zoom": True,
                }
            )


if __name__ == "__main__":
    controller = ONVIFController(ip="192.168.0.20")
    controller.move(pan=0.2)
    controller.move(tilt=-0.2)
    controller.move(zoom=0.2)
