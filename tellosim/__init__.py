"""tellosim: a DJI Tello simulator tuned to real flights of the DroneOps lab Tellos.

Swap ``from djitellopy import Tello`` for ``from tellosim import Tello`` and the same code
flies in simulation. See README.md and docs/.
"""

from .camera import Camera
from .drone import Faults, SimDrone
from .model import DroneModel, load_model
from .objects import (AprilTag, ArucoMarker, Box, ColorPanel, FloorLine, Gate, LandingPad, Marker, SimObject,
                      Wall)
from .render import Renderer
from .tello import BackgroundFrameRead, Sim, Tello, TelloException
from .world import Floor, World, builtin_maps, load_map

__version__ = "0.1.0"

__all__ = [
    "Tello", "TelloException", "BackgroundFrameRead", "Sim", "World", "Floor", "load_map", "builtin_maps",
    "Camera", "DroneModel", "load_model", "Faults", "SimDrone", "Renderer", "SimObject", "Marker", "AprilTag",
    "ArucoMarker", "FloorLine", "LandingPad", "ColorPanel", "Gate", "Box", "Wall",
]
