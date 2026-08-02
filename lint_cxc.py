#!/usr/bin/env python3
"""
lint_cxc.py -- static checks on the ChimeraX scripts.

A 1080p supersample-3 pass is roughly half an hour. Finding out at minute
twenty-eight that a selection name was misspelled three shots ago is the
failure mode this exists to prevent, so the checks below are exactly the
mistakes that produce a silently empty or wrong frame rather than an error:

  * a model number that nothing ever opened  -> that shot renders empty
  * a selection or alias name never defined  -> ChimeraX errors mid-record
  * a trajectory opened without `coordset true` -> playback has nothing to play
  * a motion command with no matching `wait`  -> commands pile up, timing dies
  * alias frame counts that disagree with metrics.json -> playback and
    recording drift apart, and the drift is only visible on the last shot

Exit status is 1 if anything at ERROR level is found, so it can gate a render.

    python3 lint_cxc.py
    python3 lint_cxc.py --strict      # warnings are errors too
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

# Commands that animate over N frames and therefore need a following `wait N`.
MOTION = {"turn", "move", "roll", "zoom", "view", "perframe", "coordset",
          "fly", "wobble"}

# Commands that start a per-frame loop which must be stopped with ~perframe.
PERFRAME_START = "perframe"


class Report:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.notes: list[str] = []

    def error(self, where: str, msg: str) -> None:
        self.errors.append(f"{where}: {msg}")

    def warn(self, where: str, msg: str) -> None:
        self.warnings.append(f"{where}: {msg}")

    def note(self, msg: str) -> None:
        self.notes.append(msg)


def read_lines(path: str) -> list[tuple[int, str]]:
    """
    Non-comment lines, with line numbers.

    Only a leading '#' starts a comment. ChimeraX uses '#' for model specs
    (#5/A:312) and for hex colours (#2E6FA8), so stripping it inline the way
    a shell linter would truncates half the real commands in these files --
    and then every name definition and every quoted perframe body looks
    malformed.
    """
    out = []
    with open(path, "r", errors="replace") as fh:
        for i, raw in enumerate(fh, start=1):
            line = raw.strip()
            if line and not line.startswith("#"):
                out.append((i, line))
    return out


def parse_open_models(path: str, rep: Report) -> dict[int, dict]:
    """Model number -> {file, coordset} from 00_open.cxc, by open order."""
    models: dict[int, dict] = {}
    if not os.path.exists(path):
        rep.error(os.path.basename(path), "missing -- run prep_shrink.py first")
        return models
    n = 0
    for lineno, line in read_lines(path):
        if not line.startswith("open "):
            continue
        n += 1
        parts = line.split()
        models[n] = {
            "file": parts[1],
            "coordset": "coordset" in line and "true" in line,
            "line": lineno,
        }
        if not os.path.exists(parts[1]):
            rep.error(f"{os.path.basename(path)}:{lineno}",
                      f"opens a file that does not exist: {parts[1]}")
    return models


def collect_definitions(paths: list[str]) -> tuple[set[str], set[str], set[str]]:
    """(named selections, aliases, colour definitions) defined anywhere."""
    names, aliases, colours = set(), set(), set()
    for p in paths:
        if not os.path.exists(p):
            continue
        for _, line in read_lines(p):
            m = re.match(r"^name\s+(?:frozen\s+)?(\w+)\b", line)
            if m:
                names.add(m.group(1))
            m = re.match(r"^alias\s+(?:\^)?(\w+)\b", line)
            if m:
                aliases.add(m.group(1))
            m = re.match(r"^colordef\s+(\w+)\b", line)
            if m:
                colours.add(m.group(1))
    return names, aliases, colours


def check_model_refs(path: str, models: dict, rep: Report) -> None:
    """Every #N referenced must have been opened."""
    opened = set(models)
    for lineno, line in read_lines(path):
        for ref in re.findall(r"#(\d+)(?:-(\d+))?", line):
            lo = int(ref[0])
            hi = int(ref[1]) if ref[1] else lo
            for m in range(lo, hi + 1):
                if m not in opened:
                    rep.error(f"{os.path.basename(path)}:{lineno}",
                              f"references model #{m}, but 00_open.cxc only "
                              f"opens #{min(opened)}-#{max(opened)}"
                              if opened else f"references model #{m}, nothing opened")


def check_names(path: str, names: set, aliases: set, colours: set,
                rep: Report) -> None:
    """Catch uses of selection names, aliases and colours that were never
    defined. Both are silent-ish failures mid-render."""
    known_cmds = {
        "open", "close", "set", "hide", "show", "color", "colordef", "name",
        "alias", "style", "size", "cartoon", "surface", "transparency",
        "lighting", "material", "graphics", "view", "turn", "move", "roll",
        "zoom", "wait", "coordset", "perframe", "movie", "windowsize", "exit",
        "select", "2dlabels", "label", "delete", "camera", "clip", "fly",
        "info", "log", "ui", "preset", "nucleotides", "crossfade", "vop",
    }
    for lineno, line in read_lines(path):
        head = line.split()[0].lstrip("~^")
        if head not in known_cmds and head not in aliases:
            rep.warn(f"{os.path.basename(path)}:{lineno}",
                     f"'{head}' is neither a known command nor a defined alias")
        # colour arguments
        for tok in re.findall(r"\bc[A-Z]\w*", line):
            if tok not in colours:
                rep.error(f"{os.path.basename(path)}:{lineno}",
                          f"uses colour '{tok}' which no colordef defines")
        # bare selection names: a token that is a known name is fine; one that
        # looks like a name but is not defined is the thing we want to catch.
        for tok in re.findall(r"(?<![#\w/:.\-])([a-z][A-Za-z]{4,})(?![\w(])", line):
            if tok in known_cmds or tok in aliases or tok in names:
                continue
            if tok in ARGUMENT_WORDS:
                continue
            rep.note(f"{os.path.basename(path)}:{lineno}: unrecognised word "
                     f"'{tok}' (harmless if it is a command argument)")


ARGUMENT_WORDS = {
    "true", "false", "none", "target", "frames", "range", "resolution",
    "probeRadius", "supersample", "format", "framerate", "quality", "record",
    "encode", "stop", "models", "atoms", "cartoons", "surfaces", "sphere",
    "ball", "stick", "tube", "ballScale", "stickRadius", "modeh", "sides",
    "rad", "width", "color", "bgColor", "silhouettes", "shadows", "soft",
    "multiShadow", "intensity", "fillIntensity", "ambientIntensity",
    "depthCue", "reflectivity", "dull", "shiny", "subdivision", "pad",
    "protein", "text", "xpos", "ypos", "size", "bold", "create", "change",
    "delete", "all", "directory", "transparentBackground", "style", "flat",
    "frozen", "spacing", "surface", "cartoon", "sharp", "wobble",
}


def check_waits(path: str, rep: Report) -> None:
    """
    Motion commands with an explicit frame count must be followed by a `wait`
    of at least that many frames before the next motion command, and every
    perframe must be stopped.
    """
    lines = read_lines(path)
    pending: list[tuple[int, str, int]] = []
    open_perframe: tuple[int, str] | None = None
    for lineno, line in lines:
        head = line.split()[0].lstrip("~")
        if line.startswith("~perframe"):
            open_perframe = None
            continue
        if head == "wait":
            parts = line.split()
            want = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            need = max((f for _, _, f in pending), default=0)
            if want < need:
                rep.error(f"{os.path.basename(path)}:{lineno}",
                          f"waits {want} frames but the preceding motion runs "
                          f"{need}; the movie will cut the move short")
            pending = []
            continue
        if head == PERFRAME_START:
            m = re.search(r"frames\s+(\d+)", line)
            f = int(m.group(1)) if m else 0
            pending.append((lineno, head, f))
            open_perframe = (lineno, line)
            continue
        if head in MOTION:
            m = re.search(r"frames\s+(\d+)", line)
            if m:
                pending.append((lineno, head, int(m.group(1))))
            else:
                # `turn y 1.5 60` -- trailing bare integer is the frame count
                parts = line.split()
                if head in ("turn", "move", "roll") and parts[-1].isdigit():
                    pending.append((lineno, head, int(parts[-1])))
    if pending:
        first = pending[0]
        rep.error(f"{os.path.basename(path)}:{first[0]}",
                  f"'{first[1]}' runs {first[2]} frames with no following "
                  f"`wait`; subsequent commands will overlap it")
    if open_perframe:
        rep.error(f"{os.path.basename(path)}:{open_perframe[0]}",
                  "perframe is never stopped with ~perframe; it will keep "
                  "firing through the following shots")


def check_trajectory_opens(models: dict, rep: Report) -> None:
    for num, info in models.items():
        base = os.path.basename(info["file"])
        is_traj = any(k in base for k in ("dissolve", "condense", "breathe"))
        if is_traj and not info["coordset"]:
            rep.error("00_open.cxc",
                      f"#{num} {base} is a trajectory but is opened without "
                      f"`coordset true`; ChimeraX will load it as N sibling "
                      f"models and `coordset #{num} ...` will play nothing")
        if not is_traj and info["coordset"]:
            rep.warn("00_open.cxc",
                     f"#{num} {base} is opened with `coordset true` but does "
                     f"not look like a trajectory")


def check_frame_counts(gen: str, metrics_path: str, rep: Report) -> None:
    """Alias frame counts in generated.cxc must match what prep actually wrote."""
    if not (os.path.exists(gen) and os.path.exists(metrics_path)):
        return
    with open(metrics_path) as fh:
        metrics = json.load(fh)
    traj = metrics.get("trajectories", {})
    want = {"playDissolve": ("dissolve", traj.get("dissolve", {}).get("frames")),
            "playCondense": ("condense", traj.get("condense", {}).get("frames")),
            "playBreathe": ("breathe", traj.get("breathe", {}).get("frames"))}
    text = open(gen).read()
    for alias, (key, n) in want.items():
        if n is None:
            continue
        m = re.search(rf"alias\s+{alias}\s+coordset\s+#\d+\s+1,(\d+)", text)
        if not m:
            rep.error("generated.cxc", f"alias {alias} is missing")
        elif int(m.group(1)) != n:
            rep.error("generated.cxc",
                      f"{alias} plays 1,{m.group(1)} but metrics.json says the "
                      f"{key} trajectory has {n} frames")
    for alias, (key, n) in want.items():
        if n is None:
            continue
        m = re.search(rf"alias\s+fade{key.capitalize()}\b.*?frames\s+(\d+)", text)
        if m and int(m.group(1)) != n:
            rep.error("generated.cxc",
                      f"fade{key.capitalize()} runs {m.group(1)} frames but the "
                      f"{key} trajectory has {n}")


def check_shot_lengths(shots: str, metrics_path: str, rep: Report) -> None:
    """The `wait` after each playback alias should equal the trajectory length."""
    if not (os.path.exists(shots) and os.path.exists(metrics_path)):
        return
    with open(metrics_path) as fh:
        traj = json.load(fh).get("trajectories", {})
    lines = read_lines(shots)
    alias_to_key = {"playDissolve": "dissolve", "playCondense": "condense",
                    "playBreathe": "breathe"}
    for i, (lineno, line) in enumerate(lines):
        key = alias_to_key.get(line.split()[0])
        if not key:
            continue
        n = traj.get(key, {}).get("frames")
        if n is None:
            continue
        for _, later in lines[i:i + 8]:
            if later.startswith("wait"):
                parts = later.split()
                got = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                if got != n:
                    rep.error(f"{os.path.basename(shots)}:{lineno}",
                              f"{line.split()[0]} plays {n} frames but the "
                              f"following wait is {got}")
                break


def check_quotes(path: str, rep: Report) -> None:
    for lineno, line in read_lines(path):
        if line.count('"') % 2:
            rep.error(f"{os.path.basename(path)}:{lineno}", "unbalanced quote")


def main() -> int:
    p = argparse.ArgumentParser(description="Static checks on the .cxc scripts.")
    p.add_argument("--out-dir", default="output")
    p.add_argument("--strict", action="store_true", help="warnings fail too")
    p.add_argument("--verbose", action="store_true", help="show notes")
    args = p.parse_args()

    rep = Report()
    open_cxc = os.path.join(args.out_dir, "00_open.cxc")
    gen_cxc = os.path.join(args.out_dir, "generated.cxc")
    metrics = os.path.join(args.out_dir, "metrics.json")
    hand = [f for f in ("00_setup.cxc", "10_shots.cxc", "render_shrink.cxc")
            if os.path.exists(f)]

    models = parse_open_models(open_cxc, rep)
    check_trajectory_opens(models, rep)

    all_cxc = [open_cxc, gen_cxc] + hand
    names, aliases, colours = collect_definitions(all_cxc)
    print(f"[definitions] {len(names)} selection names, {len(aliases)} aliases, "
          f"{len(colours)} colours, {len(models)} models opened")

    for f in hand:
        check_model_refs(f, models, rep)
        check_names(f, names, aliases, colours, rep)
        check_waits(f, rep)
        check_quotes(f, rep)

    check_frame_counts(gen_cxc, metrics, rep)
    check_shot_lengths("10_shots.cxc", metrics, rep)

    if args.verbose:
        for n in rep.notes:
            print(f"  note   {n}")
    for w in rep.warnings:
        print(f"  WARN   {w}")
    for e in rep.errors:
        print(f"  ERROR  {e}")

    bad = len(rep.errors) + (len(rep.warnings) if args.strict else 0)
    print(f"\n{len(rep.errors)} error(s), {len(rep.warnings)} warning(s), "
          f"{len(rep.notes)} note(s)")
    if bad == 0:
        print("cxc scripts look consistent -- safe to start the render")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
