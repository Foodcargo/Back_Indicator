import sys
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib
import aiortc
import aiohttp
import av

from importlib.metadata import version

print("Python:", sys.version.split()[0])
print("OpenCV:", cv2.__version__)
print("MediaPipe:", mp.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Matplotlib:", matplotlib.__version__)
print("aiortc:", aiortc.__version__)
print("aiohttp:", aiohttp.__version__)
print("PyAV:", av.__version__)
print("ONVIF:", version("onvif-zeep"))