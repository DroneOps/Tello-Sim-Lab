"""tellolab: controllers that keep a DJI Tello stable, tuned on the DroneOps lab Tellos.

They take readings (state dict, positions) and return rc commands, so they work the same with
the real drone (djitellopy) and the simulator (tellosim). See docs/controllers.md.

- AltitudeHold, AltitudeFilter, GoToAltitude: height with the ToF, without its false jumps
- HeadingHold: heading, with the yaw deadband in mind
- FloorFlow, point_below_drone: how far it moved, from the floor seen by a downward mirror
- PositionHold: PID per axis with integral (constant drift) and a runaway cutoff
- TargetCentering: get over a visual target and stay, braking with its velocity
- SafeFlip: climb, flip, and recover from the post-flip fall
- SafetyMonitor: when to refuse a takeoff and when to land
- Pilot: all of the above in a 20 Hz loop; StopWindow: video + STOP button
"""

from .altitude import (AltitudeFilter, AltitudeFilterConfig, AltitudeHold, AltitudeHoldConfig, GoToAltitude,
                       GoToAltitudeConfig)
from .flip import SafeFlip, SafeFlipConfig
from .floor_flow import FloorFlow, point_below_drone
from .heading import HeadingConfig, HeadingHold
from .pilot import Pilot, PilotConfig
from .position import PositionHold, PositionHoldConfig
from .safety import SafetyConfig, SafetyMonitor
from .target import TargetCentering, TargetCenteringConfig
from .telemetry import Telemetry
from .util import altitude_cm, clamp, with_min, wrap_deg

__version__ = "0.1.0"


def __getattr__(name):  # the window needs a display: import it only when asked
    if name == "StopWindow":
        from .hmi import StopWindow
        return StopWindow
    raise AttributeError(name)
