from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from download_dataset import DatasetDownloadError, download_dataset  # noqa: E402


class FakeDataset:
    def __init__(self, location: Path) -> None:
        self.location = str(location)


class FakeVersion:
    def __init__(self, *, create_valid_dataset: bool) -> None:
        self.create_valid_dataset = create_valid_dataset
        self.download_calls: list[dict[str, Any]] = []

    def download(self, *, model_format: str, location: str, overwrite: bool) -> FakeDataset:
        location_path = Path(location)
        self.download_calls.append(
            {
                "model_format": model_format,
                "location": location_path,
                "location_existed_before_download": location_path.exists(),
                "overwrite": overwrite,
            }
        )
        location_path.mkdir(parents=True)
        if self.create_valid_dataset:
            (location_path / "data.yaml").write_text(
                "train: train/images\nval: valid/images\nnames: [A]\n",
                encoding="utf-8",
            )
            (location_path / "train" / "images").mkdir(parents=True)
            (location_path / "train" / "labels").mkdir(parents=True)
            (location_path / "valid" / "images").mkdir(parents=True)
            (location_path / "valid" / "labels").mkdir(parents=True)
            (location_path / "train" / "images" / "a.jpg").write_bytes(b"fake image")
            (location_path / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        return FakeDataset(location_path)


class FakeProject:
    def __init__(self, version: FakeVersion) -> None:
        self.fake_version = version

    def version(self, version_number: int) -> FakeVersion:
        assert version_number == 2
        return self.fake_version


class FakeWorkspace:
    def __init__(self, version: FakeVersion) -> None:
        self.fake_version = version

    def project(self, project_name: str) -> FakeProject:
        assert project_name == "sibi-bieme"
        return FakeProject(self.fake_version)


class FakeRoboflow:
    def __init__(self, api_key: str, version: FakeVersion) -> None:
        self.api_key = api_key
        self.fake_version = version

    def workspace(self, workspace_name: str) -> FakeWorkspace:
        assert workspace_name == "sibi-detection-nftzq"
        return FakeWorkspace(self.fake_version)


def roboflow_factory(version: FakeVersion):
    return lambda api_key: FakeRoboflow(api_key, version)


def test_download_dataset_stages_into_path_that_does_not_exist_before_sdk_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ROBOFLOW_API_KEY", "secret-value")
    version = FakeVersion(create_valid_dataset=True)
    output_dir = tmp_path / "SiBiSee"

    result = download_dataset(output_dir, roboflow_factory=roboflow_factory(version))

    captured = capsys.readouterr()
    metadata = json.loads((output_dir / "source_metadata.json").read_text(encoding="utf-8"))

    assert not captured.out
    assert "secret-value" not in captured.out
    assert version.download_calls[0]["model_format"] == "yolov8"
    assert version.download_calls[0]["overwrite"] is False
    assert version.download_calls[0]["location_existed_before_download"] is False
    assert output_dir.exists()
    assert result.dataset_dir == output_dir.resolve()
    assert result.data_yaml.exists()
    assert result.image_count == 1
    assert result.label_count == 1
    assert metadata["file_count"] > 0
    assert "secret-value" not in (output_dir / "source_metadata.json").read_text(encoding="utf-8")
    assert "dataset_dir" not in metadata


def test_download_dataset_rejects_empty_sdk_result_without_success_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ROBOFLOW_API_KEY", "secret-value")
    version = FakeVersion(create_valid_dataset=False)
    output_dir = tmp_path / "SiBiSee"

    with pytest.raises(DatasetDownloadError, match="data.yaml tidak ditemukan"):
        download_dataset(output_dir, roboflow_factory=roboflow_factory(version))

    assert version.download_calls
    assert not output_dir.exists()
    assert not (output_dir / "source_metadata.json").exists()
    assert not (output_dir / "source_manifest.json").exists()


def test_download_dataset_refuses_non_empty_output_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ROBOFLOW_API_KEY", "secret-value")
    version = FakeVersion(create_valid_dataset=True)
    output_dir = tmp_path / "SiBiSee"
    output_dir.mkdir()
    (output_dir / "source_metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(DatasetDownloadError, match="Output directory tidak kosong"):
        download_dataset(output_dir, roboflow_factory=roboflow_factory(version))

    assert version.download_calls == []
    assert (output_dir / "source_metadata.json").exists()
