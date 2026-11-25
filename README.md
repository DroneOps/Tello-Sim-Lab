# Tello-Sim-Lab

A DJI Tello simulator tuned to real flights of the DroneOps lab Tellos, plus reusable
stabilization controllers for the Tello.

> Work in progress. Extracted and generalized from
> [DroneOps/AprilTag-Detection](https://github.com/DroneOps/AprilTag-Detection).

- `tellosim/`: the simulator. Drop-in replacement for `djitellopy.Tello`.
- `tellolab/`: controllers that keep the Tello stable (altitude, heading, position hold,
  visual target centering, safe flips, safety watchdogs).

## Install

```bash
pip install -e ".[dev]"
```
