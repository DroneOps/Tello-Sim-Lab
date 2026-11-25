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
