from pathlib import Path

from small_model_train.stage2_config import (
    build_llamafactory_command,
    make_training_snapshot,
    read_flat_yaml,
    write_flat_yaml,
)


def test_read_and_write_flat_yaml_round_trip(tmp_path: Path):
    path = tmp_path / "config.yaml"
    write_flat_yaml(
        path,
        {
            "bf16": True,
            "cutoff_len": 8192,
            "learning_rate": 3.0e-5,
            "template": "qwen3",
        },
    )

    result = read_flat_yaml(path)

    assert result["bf16"] is True
    assert result["cutoff_len"] == 8192
    assert result["learning_rate"] == 3.0e-5
    assert result["template"] == "qwen3"


def test_read_flat_yaml_skips_noise_and_parses_empty_values(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "\n".join(
            [
                "# generated snapshot",
                "not yaml",
                "adapter_name_or_path: null",
                "stage: 'sft'",
                'template: "qwen3"',
                "do_train: false",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = read_flat_yaml(path)

    assert result == {
        "adapter_name_or_path": None,
        "stage": "sft",
        "template": "qwen3",
        "do_train": False,
    }


def test_make_training_snapshot_overrides_model_output_and_smoke_values(
    tmp_path: Path,
):
    source = tmp_path / "source.yaml"
    output = tmp_path / "outputs" / "training_config_snapshot.yaml"
    write_flat_yaml(
        source,
        {"model_name_or_path": "remote", "output_dir": "old", "cutoff_len": 8192},
    )

    snapshot = make_training_snapshot(
        source_config=source,
        output_config=output,
        model_dir=r"E:\models\Qwen3-4B-Instruct-2507",
        output_dir="outputs/sft_smoke",
        overrides={"max_samples": 100, "num_train_epochs": 1},
    )

    assert snapshot["model_name_or_path"] == r"E:\models\Qwen3-4B-Instruct-2507"
    assert snapshot["output_dir"] == "outputs/sft_smoke"
    assert snapshot["max_samples"] == 100
    assert output.exists()


def test_build_llamafactory_command_uses_config_path():
    command = build_llamafactory_command(
        "outputs/sft_smoke/training_config_snapshot.yaml"
    )

    assert command == [
        "llamafactory-cli",
        "train",
        "outputs/sft_smoke/training_config_snapshot.yaml",
    ]
