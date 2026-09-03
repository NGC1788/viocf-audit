from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from .audio import convert_audio_file, segment_continuous_recording
from .calibration import apply_calibration_to_manifest, fit_technique_calibration
from .config import load_config, project_root_from_config
from .design import create_design, create_smoke_design
from .features import extract_manifest_features
from .metrics import run_metric_suite
from .midi import inspect_violet_midi
from .plots import make_figures
from .preflight import write_preflight
from .qc import qc_manifest
from .surrogate import train_response_surrogate
from .sweep import create_compute_sweep
from .violet import collect_violet_run


def _json_print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _config_and_root(config_path: str) -> tuple[object, Path]:
    config = load_config(config_path)
    return config, project_root_from_config(config)


def command_preflight(args: argparse.Namespace) -> None:
    _, root = _config_and_root(args.config)
    output = Path(args.output) if args.output else root / "results" / "preflight.json"
    report = write_preflight(root, output)
    _json_print(report)


def command_make_design(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    outputs = create_design(config, profile=args.profile)
    _json_print({key: str(value) for key, value in outputs.items()})


def command_make_sweep(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    outputs = create_compute_sweep(
        config,
        dense_replicates=args.dense_replicates,
        guidance_replicates=args.guidance_replicates,
        steps_replicates=args.steps_replicates,
        include_exploratory_techniques=args.include_exploratory_techniques,
    )
    _json_print({key: str(value) for key, value in outputs.items()})


def command_make_smoke(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    output = create_smoke_design(config)
    _json_print({"manifest": str(output), "clips": 2})


def command_inspect_midi(args: argparse.Namespace) -> None:
    paths: list[Path] = []
    for value in args.paths:
        path = Path(value)
        paths.extend(sorted(path.rglob("*.mid")) if path.is_dir() else [path])
    reports = [inspect_violet_midi(path) for path in paths]
    invalid = [report for report in reports if not bool(report["valid"])]
    _json_print({"files": len(reports), "invalid": invalid})
    if invalid:
        raise SystemExit(2)


def command_convert_real(args: argparse.Namespace) -> None:
    config, root = _config_and_root(args.config)
    manifest = pd.read_csv(args.manifest)
    converted = 0
    missing = []
    skipped = []
    for record in manifest.to_dict(orient="records"):
        source = root / str(record["raw_audio_path"])
        destination = root / str(record["audio_path"])
        if not source.exists():
            missing.append(str(source))
            continue
        if destination.exists() and not args.overwrite:
            skipped.append(str(destination))
            continue
        convert_audio_file(
            source,
            destination,
            target_rate=config.sample_rate,
            mono=True,
            subtype="PCM_24",
        )
        converted += 1
    _json_print({"converted": converted, "missing": missing, "skipped_existing": skipped})


def command_segment(args: argparse.Namespace) -> None:
    config, root = _config_and_root(args.config)
    cue_log = pd.read_csv(args.cue_log)
    required = {"clip_id", "source_wav", "start_s", "end_s"}
    missing_columns = sorted(required - set(cue_log.columns))
    if missing_columns:
        raise ValueError(f"Cue log is missing columns: {missing_columns}")
    completed = []
    for record in cue_log.to_dict(orient="records"):
        source = Path(str(record["source_wav"]))
        if not source.is_absolute():
            source = root / source
        destination = root / "data" / "real_48k" / f"{record['clip_id']}.wav"
        if destination.exists() and not args.overwrite:
            continue
        segment_continuous_recording(
            source,
            float(record["start_s"]),
            float(record["end_s"]),
            destination,
            target_rate=config.sample_rate,
        )
        completed.append(str(destination))
    _json_print({"segments_written": len(completed), "paths": completed})


# 음성 대조에서 주입한 누출이 **어디에 떨어져야 하는가**.
#
# 두 번 틀렸고 두 번 다 전수 실행이 잡아냈다. 기록해 둔다.
#
# 1차: "0.6 초 미리 당긴다" 고정값.
#   측정 창은 [onset+0.04, onset+0.23] 고정인데 분기 오프셋은 0.25~3.00 s 라,
#   분기가 멀면 당겨도 창 뒤에 떨어졌다. 960 중 576 이 '동일'로 나왔고
#   그 값이 정확히 '창에 닿지 않는 오프셋 3개 x 192' 였다.
#
# 2차: "오프셋과 무관하게 onset+0.10 에 떨어뜨린다".
#   오프셋 편향은 사라졌는데 이번엔 **오프셋마다 정확히 절반(96/192)** 이 '동일'
#   이었다. 192 = 6주법 x 32반복 이므로 96 = 3주법 — 줄이 풀리는 주법
#   (pizzicato/spiccato/staccato) 쪽이다.
#
#   그 주법들은 **음 시작 시점의 CC1 로 감쇠가 정해지고 이후 변화에 반응하지
#   않는다**(튕긴 줄은 손잡이를 돌려도 다시 안 커진다 — 일부러 그렇게 만든
#   물리다). onset+0.10 은 음이 시작된 **뒤**라서 그 주법들에는 애초에
#   주입이 되지 않았다. 파이프라인이 못 잡은 게 아니라 넣은 게 없었다.
#
# 그래서 **음 시작 앞**에 떨어뜨린다. 그러면
#   - 활 주법: 창 [onset+0.04, ...] 이 바뀐다 (평활 시정수 20 ms 라 충분히 안정)
#   - 감쇠 주법: 시작 시점 CC1 이 바뀌어 튕김 자체가 달라진다
# 양쪽 모두 창 안에 차이가 생긴다.
LEAK_LANDS_BEFORE_ONSET_S = 0.05


def _render_reference_one(job: tuple[dict, str, float, int, float, str]) -> dict:
    import soundfile as sf

    from .causal_reference import render

    record, arm, leak, rate, clip_seconds, output_dir = job
    samples = render(
        record["midi_path"], str(record["technique"]),
        sample_rate=rate, clip_seconds=clip_seconds,
        seed=int(record["seed"]), leak_seconds=leak,
    )
    path = Path(output_dir) / f"{record['clip_id']}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, rate)
    row = dict(record)
    row["audio_path"] = str(path)
    row["source"] = "model"
    row["reference_arm"] = arm
    return row


def command_render_reference(args: argparse.Namespace) -> None:
    from concurrent.futures import ProcessPoolExecutor

    config, root = _config_and_root(args.config)
    manifest = pd.read_csv(args.manifest)
    if args.limit and args.limit > 0:
        # noise_group 통째로 잘라야 짝이 깨지지 않는다.
        keep = manifest["noise_group"].drop_duplicates().head(
            max(1, args.limit // 3)
        )
        manifest = manifest.loc[manifest["noise_group"].isin(keep)]
    output_dir = str(root / "data" / "reference_audio" / args.arm)
    jobs = []
    for record in manifest.to_dict(orient="records"):
        if args.arm == "leaky":
            # 주입 시점 = note_onset - LEAK_LANDS_BEFORE_ONSET_S.
            # CC 사건은 note_onset + branch_offset 에 있으므로 그만큼 당긴다.
            offset = float(record.get("branch_offset_s") or 0.0)
            leak = max(0.0, offset + LEAK_LANDS_BEFORE_ONSET_S)
        else:
            leak = 0.0
        jobs.append(
            (record, args.arm, leak, config.sample_rate, config.clip_seconds, output_dir)
        )
    workers = args.workers or max(1, (os.cpu_count() or 2) - 2)
    rows: list[dict] = []
    if workers <= 1:
        rows = [_render_reference_one(job) for job in jobs]
    else:
        os.environ.setdefault("PYTHONPATH", str(Path(__file__).resolve().parents[1]))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for done, row in enumerate(
                pool.map(_render_reference_one, jobs, chunksize=8), start=1
            ):
                rows.append(row)
                if done % 500 == 0:
                    print(f"  기준 렌더 {done:,}/{len(jobs):,}", file=sys.stderr, flush=True)
    frame = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    leaks = sorted({round(job[2], 3) for job in jobs})
    _json_print({"arm": args.arm, "rows": len(frame), "leak_seconds": leaks,
                 "leak_lands_at_s": (
                     None if args.arm != "leaky"
                     else f"onset-{LEAK_LANDS_BEFORE_ONSET_S}"
                 ),
                 "output": str(Path(args.output).resolve())})


def command_make_config_robustness(args: argparse.Namespace) -> None:
    from .delayed_sweep import config_plan_size, create_config_robustness

    config, _ = _config_and_root(args.config)
    plan = config_plan_size(args.replicates)
    outputs = create_config_robustness(config, replicates=args.replicates)
    _json_print({"plan": plan, "manifests": {k: str(v) for k, v in outputs.items()}})


def command_make_delayed_sweep(args: argparse.Namespace) -> None:
    from .delayed_sweep import create_delayed_sweep, plan_size

    config, _ = _config_and_root(args.config)
    plan = plan_size(args.replicates)
    outputs = create_delayed_sweep(config, replicates=args.replicates)
    _json_print({"plan": plan, "manifests": {k: str(v) for k, v in outputs.items()}})


def command_qc(args: argparse.Namespace) -> None:
    config, root = _config_and_root(args.config)
    frame = qc_manifest(args.manifest, args.output, root, config.raw, workers=args.workers)
    _json_print(
        {
            "rows": len(frame),
            "passed": int(frame["qc_pass"].fillna(False).sum()),
            "failed": int((~frame["qc_pass"].fillna(False)).sum()),
            "output": str(Path(args.output).resolve()),
        }
    )


def command_features(args: argparse.Namespace) -> None:
    config, root = _config_and_root(args.config)
    frame = extract_manifest_features(
        args.manifest,
        args.output,
        root,
        config.raw,
        include_missing=args.include_missing,
        workers=args.workers,
    )
    errors = int(frame.get("feature_error", pd.Series(index=frame.index, dtype=object)).notna().sum())
    _json_print({"rows": len(frame), "errors": errors, "output": str(Path(args.output).resolve())})


def command_metrics(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    outputs = run_metric_suite(args.features, args.output_dir, config.raw)
    _json_print({key: str(value) for key, value in outputs.items()})


def command_calibrate(args: argparse.Namespace) -> None:
    fit_violins = tuple(value.strip() for value in args.fit_violins.split(",") if value.strip())
    outputs = fit_technique_calibration(
        args.features,
        args.output_dir,
        feature=args.feature,
        fit_violins=fit_violins,
        heldout_violin=args.heldout_violin,
    )
    _json_print({key: str(value) for key, value in outputs.items()})


def command_apply_calibration(args: argparse.Namespace) -> None:
    _, root = _config_and_root(args.config)
    frame = apply_calibration_to_manifest(
        args.manifest,
        args.mapping,
        root,
        args.output,
        calibration_id=args.calibration_id,
    )
    _json_print({"rows": len(frame), "output": str(Path(args.output).resolve())})


def command_collect_violet(args: argparse.Namespace) -> None:
    _, root = _config_and_root(args.config)
    frame = collect_violet_run(
        args.run_dir,
        args.manifest,
        root,
        args.output,
        copy_audio=not args.no_copy,
    )
    _json_print(
        {
            "expected": len(frame),
            "found": int(frame["violet_found"].sum()),
            "output": str(Path(args.output).resolve()),
        }
    )


def command_figures(args: argparse.Namespace) -> None:
    outputs = make_figures(args.features, args.metrics_dir, args.output_dir)
    _json_print({"figures": [str(path) for path in outputs]})


def command_train_surrogate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    targets = args.targets or config.raw["analysis"]["response_features"]
    outputs = train_response_surrogate(
        args.features,
        args.output_dir,
        targets=targets,
        random_seed=int(config.raw["analysis"]["random_seed"]),
        n_estimators=args.n_estimators,
        cv_splits=args.cv_splits,
        n_jobs=args.n_jobs,
    )
    _json_print({key: str(value) for key, value in outputs.items()})


def command_embed(args: argparse.Namespace) -> None:
    from .embeddings import embed_manifest

    config = load_config(args.config)
    frame = embed_manifest(
        args.manifest,
        args.output,
        project_root_from_config(config),
        model_id=args.model_id,
        layers=tuple(args.layers),
        device=args.device,
        limit=args.limit,
    )
    _json_print({"rows_written": len(frame), "output": str(args.output)})


def command_embed_metrics(args: argparse.Namespace) -> None:
    import pandas as pd

    from .embeddings import embedding_contrast_c2st, summarise

    frames = [pd.read_csv(path) for path in args.embeddings]
    data = pd.concat(frames, ignore_index=True, sort=False)
    result = embedding_contrast_c2st(data)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    summary = summarise([result])
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _json_print(summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="viocf",
        description="VioCF-Audit reproducible experiment toolkit",
    )
    parser.add_argument("--config", default="configs/experiment.yaml", help="Experiment YAML")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="Check server/GPU/disk/software")
    preflight.add_argument("--output")
    preflight.set_defaults(func=command_preflight)

    reference = subparsers.add_parser(
        "render-reference",
        help="인과적 기준 렌더러로 오디오를 만든다 (파이프라인 검증용)",
    )
    reference.add_argument("--config", default="configs/experiment.yaml")
    reference.add_argument("--manifest", required=True)
    reference.add_argument("--arm", choices=("causal", "leaky"), default="causal")
    reference.add_argument("--limit", type=int, default=0, help="0 이면 전부")
    reference.add_argument("--output", required=True)
    reference.add_argument("--workers", type=int, default=None)
    reference.set_defaults(func=command_render_reference)

    config_rob = subparsers.add_parser(
        "make-config-robustness",
        help="설정 강건성 설계 (w_cc 5점 x sampling_steps 4점)",
    )
    config_rob.add_argument("--config", default="configs/experiment.yaml")
    config_rob.add_argument("--replicates", type=int, default=32)
    config_rob.set_defaults(func=command_make_config_robustness)

    delayed_sweep = subparsers.add_parser(
        "make-delayed-sweep",
        help="지연 분기 확장 설계 (분기 오프셋 x 주법부류 x w_cc)",
    )
    delayed_sweep.add_argument("--config", default="configs/experiment.yaml")
    delayed_sweep.add_argument("--replicates", type=int, default=32)
    delayed_sweep.set_defaults(func=command_make_delayed_sweep)

    design = subparsers.add_parser("make-design", help="Generate MIDI and factorial manifests")
    design.add_argument("--profile", choices=["pilot", "full", "expanded"], default="pilot")
    design.set_defaults(func=command_make_design)

    sweep = subparsers.add_parser(
        "make-sweep", help="Generate dense CC1, guidance, and sampler-step sweeps"
    )
    sweep.add_argument("--dense-replicates", type=int, default=8)
    sweep.add_argument("--guidance-replicates", type=int, default=4)
    sweep.add_argument("--steps-replicates", type=int, default=8)
    sweep.add_argument(
        "--include-exploratory-techniques",
        action="store_true",
        help="Also sweep techniques without a real-instrument baseline",
    )
    sweep.set_defaults(func=command_make_sweep)

    embed = subparsers.add_parser(
        "embed", help="Extract MERT embeddings for a manifest (GPU)"
    )
    embed.add_argument("--manifest", required=True)
    embed.add_argument("--output", required=True)
    embed.add_argument("--model-id", default="m-a-p/MERT-v1-95M")
    embed.add_argument("--layers", type=int, nargs="+", default=[3, 4, 5, 6])
    embed.add_argument("--device", default=None)
    embed.add_argument("--limit", type=int, default=None)
    embed.set_defaults(func=command_embed)

    embed_metrics = subparsers.add_parser(
        "embed-metrics",
        help="Classifier two-sample test on MERT contrast vectors",
    )
    embed_metrics.add_argument("--embeddings", nargs="+", required=True)
    embed_metrics.add_argument("--output", required=True)
    embed_metrics.set_defaults(func=command_embed_metrics)

    smoke = subparsers.add_parser("make-smoke", help="Create a two-clip same-noise smoke test")
    smoke.set_defaults(func=command_make_smoke)

    midi = subparsers.add_parser("inspect-midi", help="Validate keyswitch/CC1 MIDI files")
    midi.add_argument("paths", nargs="+")
    midi.set_defaults(func=command_inspect_midi)

    convert = subparsers.add_parser("convert-real", help="Convert raw recordings to 48 kHz PCM24 mono")
    convert.add_argument("--manifest", required=True)
    convert.add_argument("--overwrite", action="store_true")
    convert.set_defaults(func=command_convert_real)

    segment = subparsers.add_parser("segment", help="Split a continuous recording using a cue log")
    segment.add_argument("--cue-log", required=True)
    segment.add_argument("--overwrite", action="store_true")
    segment.set_defaults(func=command_segment)

    qc = subparsers.add_parser("qc", help="Run non-destructive audio QC")
    qc.add_argument("--manifest", required=True)
    qc.add_argument("--output", required=True)
    qc.add_argument(
        "--workers", type=int, default=None,
        help="병렬 워커 수 (기본: 코어수-2). 1 이면 직렬. 결과는 워커 수와 무관하게 동일하다.",
    )
    qc.set_defaults(func=command_qc)

    features = subparsers.add_parser("features", help="Extract interpretable audio features")
    features.add_argument("--manifest", required=True)
    features.add_argument("--output", required=True)
    features.add_argument("--include-missing", action="store_true")
    features.add_argument(
        "--workers", type=int, default=None,
        help="병렬 워커 수 (기본: 코어수-2). 1 이면 직렬. 결과는 워커 수와 무관하게 동일하다.",
    )
    features.set_defaults(func=command_features)

    metrics = subparsers.add_parser("metrics", help="Compute alignment, leakage, and compositionality")
    metrics.add_argument("--features", nargs="+", required=True)
    metrics.add_argument("--output-dir", required=True)
    metrics.set_defaults(func=command_metrics)

    calibrate = subparsers.add_parser("calibrate", help="Fit technique-aware monotonic CC1 calibration")
    calibrate.add_argument("--features", nargs="+", required=True)
    calibrate.add_argument("--output-dir", required=True)
    calibrate.add_argument("--feature", default="rms_dbfs")
    calibrate.add_argument("--fit-violins", default="V1,V2")
    calibrate.add_argument("--heldout-violin", default="V3")
    calibrate.set_defaults(func=command_calibrate)

    apply_cal = subparsers.add_parser("apply-calibration", help="Create corrected-CC1 MIDI manifest")
    apply_cal.add_argument("--manifest", required=True)
    apply_cal.add_argument("--mapping", required=True)
    apply_cal.add_argument("--output", required=True)
    apply_cal.add_argument("--calibration-id", default="isotonic-v1")
    apply_cal.set_defaults(func=command_apply_calibration)

    collect = subparsers.add_parser("collect-violet", help="Verify and collect an official VIOLET run")
    collect.add_argument("--run-dir", required=True)
    collect.add_argument("--manifest", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--no-copy", action="store_true")
    collect.set_defaults(func=command_collect_violet)

    figures = subparsers.add_parser("figures", help="Create presentation-ready audit figures")
    figures.add_argument("--features", nargs="+", required=True)
    figures.add_argument("--metrics-dir", required=True)
    figures.add_argument("--output-dir", required=True)
    figures.set_defaults(func=command_figures)

    surrogate = subparsers.add_parser(
        "train-surrogate", help="Fit prompt-grouped ExtraTrees response surfaces"
    )
    surrogate.add_argument("--features", nargs="+", required=True)
    surrogate.add_argument("--output-dir", required=True)
    surrogate.add_argument("--targets", nargs="+")
    surrogate.add_argument("--n-estimators", type=int, default=400)
    surrogate.add_argument("--cv-splits", type=int, default=5)
    surrogate.add_argument("--n-jobs", type=int, default=-1)
    surrogate.set_defaults(func=command_train_surrogate)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
