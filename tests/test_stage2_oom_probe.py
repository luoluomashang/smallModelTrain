from __future__ import annotations

import argparse
import io
import subprocess
from pathlib import Path

import pytest


def test_run_oom_probes_records_failed_probe_and_stdout_only_cuda_oom(
    tmp_path: Path,
    monkeypatch,
):
    from small_model_train import stage2_oom_probe

    class FakePopen:
        calls = 0

        def __init__(self, command, **kwargs):
            FakePopen.calls += 1
            self.command = command
            self.stdout = io.StringIO("RuntimeError: CUDA out of memory\n")
            self.stderr = io.StringIO("")
            self.returncode = 1

        def poll(self):
            return self.returncode

        def wait(self):
            return self.returncode

    monkeypatch.setattr(stage2_oom_probe.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(stage2_oom_probe, "collect_gpu_processes", lambda: [])

    results = stage2_oom_probe.run_oom_probes(
        model_dir="model",
        cards="cards.jsonl",
        sft_dataset="sft.jsonl",
        config="config.yaml",
        log_dir=tmp_path / "logs",
    )

    assert len(results) == len(stage2_oom_probe.PROBES)
    assert all(result["status"] == "failed" for result in results)
    assert results[0]["probe"] == stage2_oom_probe.PROBES[0]
    assert results[0]["error_type"] == "cuda_oom"
    assert Path(results[0]["stdout_log"]).read_text(encoding="utf-8") == (
        "RuntimeError: CUDA out of memory\n"
    )


def test_render_probe_report_includes_probe_results():
    from small_model_train.stage2_oom_probe import PROBES, render_probe_report

    report = render_probe_report(
        [
            {
                "probe": PROBES[0],
                "status": "passed",
                "exit_code": 0,
                "error_type": "none",
                "suggestion": "无",
            },
            {
                "probe": PROBES[1],
                "status": "failed",
                "exit_code": 1,
                "error_type": "cuda_oom",
                "suggestion": "降低 cutoff_len",
            },
        ]
    )

    assert "## Probe Results" in report
    assert f"- {PROBES[0]}: passed (exit=0, error=none)" in report
    assert f"- {PROBES[1]}: failed (exit=1, error=cuda_oom)" in report
    assert "Probe 2 失败：基座 4-bit 加载不稳" in report


def test_run_oom_probe_dry_run_does_not_execute_subprocess(
    tmp_path: Path,
    monkeypatch,
):
    from scripts import run_oom_probe

    def fail_run(*args, **kwargs):
        raise AssertionError("dry-run must not execute probes")

    monkeypatch.setattr(run_oom_probe, "run_oom_probes", fail_run)
    report = tmp_path / "oom_probe_report.md"

    exit_code = run_oom_probe.main(["--report", str(report), "--dry-run"])

    assert exit_code == 0
    assert "# OOM Probe Report" in report.read_text(encoding="utf-8")
    assert "## Probe Results" not in report.read_text(encoding="utf-8")


def test_run_oom_probe_returns_one_when_any_probe_fails(tmp_path: Path, monkeypatch):
    from scripts import run_oom_probe

    def fake_run_oom_probes(**_kwargs):
        return [
            {
                "probe": run_oom_probe.PROBES[0],
                "status": "failed",
                "exit_code": 1,
                "error_type": "cuda_oom",
                "suggestion": "降低 cutoff_len",
            }
        ]

    monkeypatch.setattr(run_oom_probe, "run_oom_probes", fake_run_oom_probes)
    report = tmp_path / "oom_probe_report.md"

    exit_code = run_oom_probe.main(["--report", str(report)])

    assert exit_code == 1
    assert "## Probe Results" in report.read_text(encoding="utf-8")
    assert "cuda_oom" in report.read_text(encoding="utf-8")


def test_one_step_probe_snapshot_uses_supplied_sft_dataset_name(tmp_path: Path, monkeypatch):
    from scripts import stage2_oom_probe_worker

    config = tmp_path / "sft.yaml"
    config.write_text(
        "model_name_or_path: old_model\n"
        "output_dir: old_output\n"
        "dataset: old_dataset\n"
        "dataset_dir: old_dataset_dir\n"
        "cutoff_len: 8192\n",
        encoding="utf-8",
    )
    sft_dataset = tmp_path / "data_sft" / "sft_chapter_v1.jsonl"
    sft_dataset.parent.mkdir()
    sft_dataset.write_text('{"text": "sample"}\n', encoding="utf-8")

    def fake_run(command, **kwargs):
        assert command[0:2] == ["llamafactory-cli", "train"]
        assert kwargs["env"]["WANDB_DISABLED"] == "true"
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(stage2_oom_probe_worker.subprocess, "run", fake_run)
    args = argparse.Namespace(
        probe=5,
        model_dir="model",
        cards="cards.jsonl",
        sft_dataset=str(sft_dataset),
        config=str(config),
        log_dir=str(tmp_path / "logs"),
    )

    exit_code = stage2_oom_probe_worker._probe_one_step_training(args)

    snapshot = tmp_path / "logs" / "probe_5" / "training_config_snapshot.yaml"
    snapshot_text = snapshot.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "dataset: sft_chapter_v1\n" in snapshot_text
    assert f"dataset_dir: {sft_dataset.parent}\n" in snapshot_text
    assert str(sft_dataset) not in snapshot_text
    assert "old_dataset" not in snapshot_text
