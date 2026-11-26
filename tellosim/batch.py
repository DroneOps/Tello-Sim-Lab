"""Run the same UNMODIFIED program many times (one seed each, in parallel) and sum up the results.

    python -m tellosim.batch --seeds 1-8 --jobs 4 --world tmr_landing --keys t@5 --seconds 120 main.py
    python -m tellosim.batch --seeds 1,2,5 --out runs/ --fog 300 my_program.py --its-own-args

One flight tells you little: the drone drifts, the video comes late, and your program runs in real
time next to the simulator, so even the same seed does not give exactly the same flight twice.
This runs it once per seed, each one with ``python -m tellosim.run --headless --seed N
--report-json``, and prints a table (and writes summary.json) with what really happened.

Options of its own:
  --seeds 1-8 or 1,3,7   which seeds (one run each)                     default 1-5
  --jobs N               runs at the same time                          default: half the CPUs
  --out DIR              where each run's log and JSON report go        default: tellosim_batch/
Everything else (--world, --model, --fog, --keys, --seconds, the program and its arguments) is
passed to tellosim.run as it is.

If your program saves its own logs or videos, all the runs write them to the same place: send them
somewhere else with your program's own options if you need to tell them apart.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def parse_seeds(text: str) -> list[int]:
    """'1-5' -> [1, 2, 3, 4, 5]; '1,3,7' -> [1, 3, 7]; '2-3,9' -> [2, 3, 9]."""
    seeds = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        seeds.extend(range(int(lo), int(hi) + 1) if sep else [int(lo)])
    if not seeds:
        raise ValueError(f"--seeds: no seeds in '{text}'")
    return seeds


def run_one(seed: int, forward: list[str], out: Path) -> dict:
    """One tellosim.run with this seed; returns its JSON report (or what went wrong)."""
    report, log = out / f"seed_{seed}.json", out / f"seed_{seed}.log"
    if report.exists():
        report.unlink()
    cmd = [sys.executable, "-m", "tellosim.run", "--headless", "--seed", str(seed),
           "--report-json", str(report), *forward]
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if not report.exists():
        return {"seed": seed, "error": f"no report (exit code {rc}); see {log}"}
    data = json.loads(report.read_text(encoding="utf-8"))
    data["seed"] = seed
    return data


def row(run: dict) -> dict:
    """The few numbers of one run that matter for the table."""
    if "error" in run:
        return {"seed": run["seed"], "error": run["error"]}
    d = (run.get("drones") or [{}])[0]
    landing = (d.get("landings") or [None])[0]
    return {"seed": run["seed"], "flight_time_s": d.get("flight_time_s"), "crash": d.get("crash"),
            "landed": landing is not None,
            "on_pad": landing["on_pad"] if landing else None,
            "distance_cm": landing["distance_cm"] if landing else None,
            "gates": sum(g["through"] for g in d.get("gates", [])), "gates_total": len(d.get("gates", []))}


def aggregate(rows: list[dict]) -> dict:
    """Totals over the runs: how many touched down on the pad, distances, crashes, errors."""
    ok = [r for r in rows if "error" not in r]
    dist = [r["distance_cm"] for r in ok if r["distance_cm"] is not None]
    on_pad = [r for r in ok if r["on_pad"]]
    return {"runs": len(rows), "errors": len(rows) - len(ok),
            "crashes": sum(1 for r in ok if r["crash"]),
            "landed": sum(1 for r in ok if r["landed"]), "on_pad": len(on_pad),
            "distance_cm_median": round(statistics.median(dist), 1) if dist else None,
            "distance_cm_max": round(max(dist), 1) if dist else None}


def table(rows: list[dict], total: dict) -> str:
    lines = [f"{'seed':>5}  {'time s':>6}  {'touchdown':<22} {'gates':<6} crash"]
    for r in sorted(rows, key=lambda r: r["seed"]):
        if "error" in r:
            lines.append(f"{r['seed']:>5}  ERROR: {r['error']}")
            continue
        td = "-" if not r["landed"] else f"{r['distance_cm']:.0f} cm ({'on the pad' if r['on_pad'] else 'OFF the pad'})"
        gates = f"{r['gates']}/{r['gates_total']}" if r["gates_total"] else "-"
        lines.append(f"{r['seed']:>5}  {r['flight_time_s'] or 0:>6.0f}  {td:<22} {gates:<6} {r['crash'] or 'none'}")
    s = f"\n{total['runs']} runs: {total['on_pad']} touched down on the pad, {total['landed']} touched down, " \
        f"{total['crashes']} crashed, {total['errors']} errors"
    if total["distance_cm_median"] is not None:
        s += f"; distance to the pad center: median {total['distance_cm_median']:.0f} cm, max {total['distance_cm_max']:.0f} cm"
    return "\n".join(lines) + s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 usage="python -m tellosim.batch [--seeds 1-5] [--jobs N] [--out DIR] "
                                       "[tellosim.run options] program.py [its arguments]")
    ap.add_argument("--seeds", default="1-5")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--out", default="tellosim_batch")
    a, forward = ap.parse_known_args(argv)
    for own in ("--seed", "--report-json", "--viewer", "--record"):
        if any(f == own or f.startswith(own + "=") for f in forward):
            ap.error(f"{own} is set by the batch (one per run); leave it out")
    if not any(f.endswith(".py") for f in forward):
        ap.error("which program? e.g. python -m tellosim.batch --world tmr_landing main.py")
    seeds = parse_seeds(a.seeds)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[tellosim.batch] {len(seeds)} runs, {a.jobs} at a time; logs and reports in {out}/", flush=True)

    def one(seed):
        r = row(run_one(seed, forward, out))
        print(f"[tellosim.batch] seed {seed} done", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        rows = list(pool.map(one, seeds))
    total = aggregate(rows)
    print(table(rows, total), flush=True)
    (out / "summary.json").write_text(json.dumps({"runs": rows, "total": total}, indent=2), encoding="utf-8")
    return 1 if total["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
