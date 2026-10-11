import argparse
import asyncio
from pathlib import Path

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

from pose_analysis import ExerciseAnalyzer, create_auto_ptz, open_camera


pcs = set()


class CameraExerciseTrack(VideoStreamTrack):
    def __init__(self, camera_index=0, auto_ptz=None, digital_ptz=False):
        super().__init__()
        self.cap = open_camera(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera not found: index {camera_index}")
        self.analyzer = ExerciseAnalyzer(
            auto_ptz=auto_ptz,
            digital_ptz=digital_ptz,
        )

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read camera frame")

        frame, _ = self.analyzer.process_frame(frame)
        video = VideoFrame.from_ndarray(frame, format="bgr24")
        video.pts = pts
        video.time_base = time_base
        return video

    def close(self):
        self.analyzer.close()
        self.cap.release()


async def index(request):
    return web.FileResponse(Path(__file__).with_name("index.html"))


async def offer(request):
    params = await request.json()
    settings = request.app["settings"]

    pc = RTCPeerConnection()
    pcs.add(pc)

    track = CameraExerciseTrack(
        camera_index=settings.camera,
        auto_ptz=request.app["auto_ptz"],
        digital_ptz=settings.digital_ptz,
    )

    @pc.on("connectionstatechange")
    async def on_state_change():
        print("Connection:", pc.connectionState)
        if pc.connectionState in ("failed", "closed"):
            track.close()
            await pc.close()
            pcs.discard(pc)

    pc.addTrack(track)

    await pc.setRemoteDescription(
        RTCSessionDescription(
            sdp=params["sdp"],
            type=params["type"],
        )
    )

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response(
        {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }
    )


async def cleanup(app):
    await asyncio.gather(*(pc.close() for pc in pcs))
    pcs.clear()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--camera", type=int, default=2)
    parser.add_argument("--digital-ptz", action="store_true")
    parser.add_argument("--ptz", action="store_true")
    parser.add_argument("--ptz-ip", default="192.168.0.20")
    parser.add_argument("--ptz-port", type=int, default=80)
    parser.add_argument("--ptz-user", default="admin")
    parser.add_argument("--ptz-password", default="")
    return parser.parse_args()


def create_app(settings):
    app = web.Application()
    app["settings"] = settings
    app["auto_ptz"] = create_auto_ptz(
        enabled=settings.ptz,
        ip=settings.ptz_ip,
        port=settings.ptz_port,
        username=settings.ptz_user,
        password=settings.ptz_password,
    )
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.on_cleanup.append(cleanup)
    return app


def main():
    settings = parse_args()
    web.run_app(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
