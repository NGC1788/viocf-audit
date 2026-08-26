#!/usr/bin/env python3
"""Visual cue and crash-safe session logger for VioCF real-instrument recordings."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

from viocf.recording import RecordingItem, RecordingSession, cue_frame

TECHNIQUE_LABELS = {
    "sustain": "SUSTAIN · 길게 활",
    "staccato": "STACCATO · 짧게 끊기",
    "pizzicato": "PIZZICATO · 손가락 뜯기",
    "legato_slur": "LEGATO / SLUR · 이음줄",
}


def technique_label(value: str) -> str:
    return TECHNIQUE_LABELS.get(value, value.upper())


def condition_schedule(item: RecordingItem) -> str:
    if item.branch_time_s is None:
        return f"{item.dynamic_label}  ·  CC1 {item.cc1_final}"
    return (
        f"mf / CC1 {item.cc1_initial}  →  {item.branch_time_s:.2f}초에 "
        f"{item.dynamic_label} / CC1 {item.cc1_final}"
    )


class TkCueApp:
    def __init__(
        self,
        session: RecordingSession,
        *,
        countdown_s: float,
        duration_s: float,
        fullscreen: bool,
    ) -> None:
        import tkinter as tk

        self.tk = tk
        self.session = session
        self.countdown_s = countdown_s
        self.duration_s = duration_s
        self.fullscreen = fullscreen
        self.state = "ready"
        self.phase_started = 0.0
        self.last_cue_phase = ""

        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            raise RuntimeError("Tk display is unavailable") from exc
        self.root.title("VioCF 실악기 녹음 큐")
        self.root.configure(background="#101217")
        self.root.geometry("1280x800")
        self.root.attributes("-fullscreen", fullscreen)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.progress = tk.Label(
            self.root,
            bg="#101217",
            fg="#aab2c0",
            font=("Arial", 18),
        )
        self.progress.pack(pady=(24, 6))
        self.condition = tk.Label(
            self.root,
            bg="#101217",
            fg="#f5f7fa",
            font=("Arial", 27, "bold"),
            wraplength=1160,
            justify="center",
        )
        self.condition.pack(pady=14)
        self.dynamic = tk.Label(
            self.root,
            bg="#101217",
            fg="#54e38e",
            font=("Arial", 76, "bold"),
            wraplength=1160,
            justify="center",
        )
        self.dynamic.pack(expand=True, fill="both", padx=50, pady=12)
        self.filename = tk.Label(
            self.root,
            bg="#101217",
            fg="#d0d6df",
            font=("Courier", 13),
            wraplength=1160,
            justify="center",
        )
        self.filename.pack(pady=8)
        self.instructions = tk.Label(
            self.root,
            text="SPACE 시작   ·   A 채택   ·   R 재촬영   ·   S 건너뛰기   ·   Q 종료",
            bg="#1a1e27",
            fg="#d0d6df",
            font=("Arial", 17),
            pady=14,
        )
        self.instructions.pack(fill="x", side="bottom")

        self.root.bind("<space>", lambda _event: self.start())
        self.root.bind("a", lambda _event: self.accept())
        self.root.bind("A", lambda _event: self.accept())
        self.root.bind("r", lambda _event: self.redo())
        self.root.bind("R", lambda _event: self.redo())
        self.root.bind("s", lambda _event: self.skip())
        self.root.bind("S", lambda _event: self.skip())
        self.root.bind("q", lambda _event: self.close())
        self.root.bind("Q", lambda _event: self.close())
        self.root.bind("<F11>", lambda _event: self.toggle_fullscreen())
        self.root.bind("<Escape>", lambda _event: self.leave_fullscreen())
        self.render_ready()

    def set_background(self, color: str) -> None:
        self.root.configure(background=color)
        for widget in (self.progress, self.condition, self.dynamic, self.filename):
            widget.configure(background=color)

    def render_header(self, item: RecordingItem) -> None:
        current_number = self.session.completed + 1
        self.progress.configure(
            text=(
                f"{current_number} / {self.session.total}   ·   "
                f"채택 {self.session.accepted}   ·   건너뜀 {self.session.skipped}   ·   "
                f"시도 {self.session.attempt_for(item)}"
            )
        )
        self.condition.configure(
            text=(
                f"{item.violin_id}  |  {item.prompt_id}  |  "
                f"{technique_label(item.technique)}\n{condition_schedule(item)}"
            )
        )
        self.filename.configure(
            text=f"저장 파일: {item.raw_audio_path}\n참조 MIDI: {item.midi_path}"
        )

    def render_ready(self) -> None:
        self.state = "ready"
        self.last_cue_phase = ""
        item = self.session.current
        self.set_background("#101217")
        if item is None:
            self.render_done()
            return
        self.render_header(item)
        delayed_hint = "\n전환 목표를 미리 확인하세요" if item.delayed else ""
        self.dynamic.configure(
            text=f"준비\nSPACE를 누르면 시작{delayed_hint}",
            foreground="#54e38e",
        )

    def render_done(self) -> None:
        self.state = "done"
        self.set_background("#10251d")
        self.progress.configure(
            text=(
                f"완료 {self.session.total} / {self.session.total}   ·   "
                f"채택 {self.session.accepted}   ·   건너뜀 {self.session.skipped}"
            )
        )
        self.condition.configure(text="녹음 세션 완료")
        self.dynamic.configure(text="DONE", foreground="#54e38e")
        self.filename.configure(
            text=(
                f"복구 로그: {self.session.log_path}\n"
                f"상태표: {self.session.log_path.with_suffix('.csv')}"
            )
        )

    def start(self) -> None:
        if self.state != "ready" or self.session.done:
            return
        self.state = "countdown"
        self.phase_started = time.monotonic()
        self.tick()

    def tick(self) -> None:
        item = self.session.current
        if item is None:
            self.render_done()
            return
        now = time.monotonic()
        if self.state == "countdown":
            elapsed = now - self.phase_started
            remaining = self.countdown_s - elapsed
            if remaining > 0:
                self.set_background("#101217")
                self.dynamic.configure(
                    text=str(max(1, math.ceil(remaining))), foreground="#f5d76e"
                )
                self.root.after(20, self.tick)
                return
            self.state = "recording"
            self.phase_started = now
            self.last_cue_phase = ""

        if self.state != "recording":
            return
        elapsed = now - self.phase_started
        frame = cue_frame(item, elapsed, self.duration_s)
        if frame.phase == "done":
            self.state = "review"
            self.set_background("#171b23")
            self.dynamic.configure(
                text="TAKE 종료\nA 채택  ·  R 재촬영  ·  S 건너뛰기",
                foreground="#f5f7fa",
                font=("Arial", 42, "bold"),
            )
            return

        if frame.phase != self.last_cue_phase:
            self.last_cue_phase = frame.phase
        if frame.phase == "initial":
            self.set_background("#3a2b05")
            self.dynamic.configure(
                text=(
                    f"유지  {frame.dynamic_label}\n"
                    f"{frame.switch_in_s:.2f}초 뒤 → {item.dynamic_label}"
                ),
                foreground="#ffe083",
                font=("Arial", 61, "bold"),
            )
        else:
            self.set_background("#123c2d")
            prefix = "전환" if item.delayed else "GO"
            self.dynamic.configure(
                text=f"{prefix}\n{frame.dynamic_label}  ·  CC1 {frame.cc1}",
                foreground="#64f0a5",
                font=("Arial", 70, "bold"),
            )
        self.filename.configure(
            text=(
                f"REC {elapsed:05.2f} / {self.duration_s:.2f}초\n"
                f"저장 파일: {item.raw_audio_path}"
            )
        )
        self.root.after(20, self.tick)

    def accept(self) -> None:
        if self.state != "review":
            return
        self.session.mark("accept")
        self.dynamic.configure(font=("Arial", 76, "bold"))
        self.render_ready()

    def redo(self) -> None:
        if self.state not in {"countdown", "recording", "review"}:
            return
        self.session.mark("redo")
        self.dynamic.configure(font=("Arial", 76, "bold"))
        self.render_ready()

    def skip(self) -> None:
        if self.state not in {"ready", "review"}:
            return
        self.session.mark("skip")
        self.dynamic.configure(font=("Arial", 76, "bold"))
        self.render_ready()

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def leave_fullscreen(self) -> None:
        self.fullscreen = False
        self.root.attributes("-fullscreen", False)

    def close(self) -> None:
        self.session.write_snapshot()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def _terminal_take(item: RecordingItem, countdown_s: float, duration_s: float) -> None:
    countdown_end = time.monotonic() + countdown_s
    while time.monotonic() < countdown_end:
        remaining = countdown_end - time.monotonic()
        print(f"\r시작까지 {max(1, math.ceil(remaining))} ", end="", flush=True)
        time.sleep(min(0.05, max(0.0, remaining)))
    print("\rGO!            ")
    started = time.monotonic()
    while True:
        elapsed = time.monotonic() - started
        frame = cue_frame(item, elapsed, duration_s)
        if frame.phase == "done":
            break
        if frame.phase == "initial":
            cue = f"유지 {frame.dynamic_label}; {frame.switch_in_s:.2f}초 뒤 {item.dynamic_label}"
        else:
            cue = f"{'전환' if item.delayed else '유지'} {frame.dynamic_label} / CC1 {frame.cc1}"
        print(f"\rREC {elapsed:05.2f}/{duration_s:.2f}초  {cue:<42}", end="", flush=True)
        time.sleep(0.04)
    print("\rTAKE 종료" + " " * 70)


def run_terminal(session: RecordingSession, countdown_s: float, duration_s: float) -> None:
    while not session.done:
        item = session.current
        assert item is not None
        print("\n" + "=" * 72)
        print(
            f"{session.completed + 1}/{session.total} | {item.violin_id} | "
            f"시도 {session.attempt_for(item)}"
        )
        print(f"{item.prompt_id} | {technique_label(item.technique)}")
        print(condition_schedule(item))
        print(f"저장 파일: {item.raw_audio_path}")
        command = input("Enter=큐 시작, s=건너뛰기, q=종료 > ").strip().lower()
        if command == "q":
            break
        if command == "s":
            session.mark("skip")
            continue
        _terminal_take(item, countdown_s, duration_s)
        while True:
            decision = input("a=채택, r=재촬영, s=건너뛰기, q=종료 > ").strip().lower()
            if decision == "a":
                session.mark("accept")
                break
            if decision == "r":
                session.mark("redo")
                break
            if decision == "s":
                session.mark("skip")
                break
            if decision == "q":
                session.write_snapshot()
                return
    print(
        f"세션 상태: {session.completed}/{session.total}, "
        f"채택 {session.accepted}, 건너뜀 {session.skipped}\n로그: {session.log_path}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VioCF 실악기 녹음용 무음 시각 큐와 복구 가능한 세션 로그"
    )
    parser.add_argument("manifest", type=Path, help="*_real.csv 또는 *_delayed_real.csv")
    parser.add_argument(
        "--log",
        type=Path,
        help="세션 JSONL 경로 (기본: data/recording_sessions/<manifest>.jsonl)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="추가 결정적 셔플 seed; 생략하면 manifest recording_order를 그대로 사용",
    )
    parser.add_argument("--countdown", type=float, default=3.0, help="시작 카운트다운 초")
    parser.add_argument("--duration", type=float, default=10.0, help="한 take의 시각 큐 길이 초")
    parser.add_argument("--windowed", action="store_true", help="전체 화면 대신 창으로 실행")
    parser.add_argument("--terminal", action="store_true", help="Tk GUI 대신 터미널 큐 사용")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.countdown < 0:
        raise SystemExit("--countdown은 0 이상이어야 합니다")
    if args.duration <= 0:
        raise SystemExit("--duration은 0보다 커야 합니다")
    manifest = args.manifest.expanduser().resolve()
    project_root = Path(__file__).resolve().parents[1]
    log_path = (
        args.log.expanduser().resolve()
        if args.log is not None
        else project_root / "data" / "recording_sessions" / f"{manifest.stem}.jsonl"
    )
    session = RecordingSession.open(manifest, log_path, seed=args.seed)
    if session.repaired_log:
        print("주의: 비정상 종료로 잘린 마지막 로그 1개를 제거하고 복구했습니다.")
    print(f"세션: {session.session_id} | 진행 {session.completed}/{session.total}")
    print(f"로그: {session.log_path}")
    if session.done:
        print("이 세션은 이미 완료되었습니다.")
        return 0
    if args.terminal:
        run_terminal(session, args.countdown, args.duration)
        return 0
    try:
        app = TkCueApp(
            session,
            countdown_s=args.countdown,
            duration_s=args.duration,
            fullscreen=not args.windowed,
        )
    except (ImportError, RuntimeError) as exc:
        print(f"GUI를 시작하지 못했습니다: {exc}", file=sys.stderr)
        print("화면 없는 환경에서는 --terminal 옵션을 사용하세요.", file=sys.stderr)
        return 2
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
