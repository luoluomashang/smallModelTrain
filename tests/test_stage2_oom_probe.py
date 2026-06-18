from __future__ import annotations

import io
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
