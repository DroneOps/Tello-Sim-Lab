"""python -m tellosim.run: an unmodified djitellopy program, in the simulator."""
import subprocess
import sys
import textwrap


def test_run_unmodified_program(tmp_path):
    prog = tmp_path / "prog.py"
    prog.write_text(textwrap.dedent("""
        import sys, time
        import cv2
        from djitellopy import Tello
        t = Tello()
        t.connect()
        t.streamon()
        frame = None
        for _ in range(100):
            frame = t.get_frame_read().frame
            if frame.any():
                break
            time.sleep(0.05)
        print("FRAME", frame.shape, "SIM", type(t).__mro__[1].__module__, "ARGS", sys.argv[1:])
        cv2.imshow("x", frame)                      # --headless: does nothing
        keys = [cv2.waitKey(50) & 0xFF for _ in range(40)]
        print("KEY", "".join(chr(k) for k in keys if k != 255))
        t.end()
    """))
    out = subprocess.run(
        [sys.executable, "-m", "tellosim.run", "--world", "tmr_landing", "--headless", "--keys", "t@0.5",
         "--camera-set", "mirror_axis=h", str(prog), "--mine", "1"],
        capture_output=True, text=True, timeout=60).stdout
    assert "FRAME (720, 960, 3) SIM tellosim.tello ARGS ['--mine', '1']" in out
    assert "KEY t" in out
    assert "crash: none" in out


def test_run_options_from_a_json_file(tmp_path):
    """--fog and --faults also take a .json file (quotes in commands are hard on Windows)."""
    (tmp_path / "fog.json").write_text('{"visibility_cm": 300, "patches": 0.5}')
    prog = tmp_path / "prog.py"
    prog.write_text("from djitellopy import Tello\nt = Tello()\nprint('FOG', t.sim.world.fog.visibility_cm, t.sim.world.fog.patches)\n")
    out = subprocess.run([sys.executable, "-m", "tellosim.run", "--headless", "--fog", str(tmp_path / "fog.json"), str(prog)],
                         capture_output=True, text=True, timeout=60).stdout
    assert "FOG 300 0.5" in out


def test_run_report_json(tmp_path):
    """--report-json writes the end-of-run results as data, the same ones that are printed."""
    import json
    prog = tmp_path / "prog.py"
    prog.write_text("from djitellopy import Tello\nt = Tello()\nt.connect()\nt.end()\n")
    report = tmp_path / "r.json"
    out = subprocess.run([sys.executable, "-m", "tellosim.run", "--headless", "--world", "tmr_landing", "--seed", "3",
                          "--report-json", str(report), str(prog)], capture_output=True, text=True, timeout=60).stdout
    assert "crash: none" in out
    r = json.loads(report.read_text())
    assert r["world"] == "tmr_landing" and r["seed"] == 3
    assert r["drones"][0]["crash"] is None and r["drones"][0]["landings"] == []


def test_batch_helpers():
    from tellosim.batch import aggregate, parse_seeds, row
    assert parse_seeds("1-3,7") == [1, 2, 3, 7]
    run = {"seed": 4, "drones": [{"flight_time_s": 30, "crash": None, "gates": [],
                                  "landings": [{"pad": "pad", "distance_cm": 8.0, "on_pad": True, "t_s": 40}]}]}
    rows = [row(run), row({"seed": 5, "drones": [{"flight_time_s": 9, "crash": "wall", "gates": [], "landings": []}]}),
            row({"seed": 6, "error": "no report"})]
    total = aggregate(rows)
    assert total == {"runs": 3, "errors": 1, "crashes": 1, "landed": 1, "on_pad": 1,
                     "distance_cm_median": 8.0, "distance_cm_max": 8.0}


def test_batch_runs_every_seed(tmp_path):
    import json
    prog = tmp_path / "prog.py"
    prog.write_text("from djitellopy import Tello\nt = Tello()\nt.connect()\nt.end()\n")
    out = subprocess.run([sys.executable, "-m", "tellosim.batch", "--seeds", "1-2", "--jobs", "2",
                          "--out", str(tmp_path / "b"), "--world", "tmr_landing", str(prog)],
                         capture_output=True, text=True, timeout=120).stdout
    assert "2 runs" in out
    s = json.loads((tmp_path / "b" / "summary.json").read_text())
    assert [r["seed"] for r in s["runs"]] == [1, 2] and s["total"]["errors"] == 0


def test_run_glare_option(tmp_path):
    prog = tmp_path / "prog.py"
    prog.write_text("from djitellopy import Tello\nt = Tello()\nprint('LIGHTS', len(t.sim.world.lights) > 0, "
                    "bool(t.sim.world.objects[0].glossy_surfaces()))\n")
    out = subprocess.run([sys.executable, "-m", "tellosim.run", "--headless", "--world", "tmr_landing",
                          "--glare", "0.7", str(prog)], capture_output=True, text=True, timeout=60).stdout
    assert "LIGHTS True True" in out
