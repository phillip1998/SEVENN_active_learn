import io
import os
import time
import zipfile

from fastapi.testclient import TestClient

import app


def test_submit_job_and_download_results(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_g16 = bin_dir / "g16"
    fake_g16.write_text(
        "#!/bin/bash\necho done\n",
        encoding="utf-8",
    )
    fake_g16.chmod(0o755)

    monkeypatch.setattr(app, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(app, "EXECUTOR", "local")
    monkeypatch.setattr(app, "RUN_COMMAND", ["bash", "run.sh"])
    monkeypatch.setattr(app, "GAUSSIAN_BIN", str(fake_g16))
    monkeypatch.setattr(app, "GAUSSIAN_DIR", str(bin_dir))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")

    client = TestClient(app.app)
    response = client.post(
        "/jobs",
        data={
            "node": "node07",
            "commands_text": "g16 mol1.gjf >>mol1.log\ng16 mol2.gjf >>mol2.log",
        },
        files=[
            ("files", ("mol1.gjf", b"%chk=mol1.chk\n\n0 1\n", "text/plain")),
            ("files", ("mol2.gjf", b"%chk=mol2.chk\n\n0 1\n", "text/plain")),
        ],
    )

    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert response.json()["node"] == "node07"

    for _ in range(50):
        status_response = client.get(f"/jobs/{job_id}")
        status = status_response.json()["status"]
        if status in {"completed", "failed"}:
            break
        time.sleep(0.02)

    assert status_response.json()["status"] == "completed"
    assert status_response.json()["log_files"] == ["mol1.log", "mol2.log"]

    results_response = client.get(f"/jobs/{job_id}/results")
    assert results_response.status_code == 200

    with zipfile.ZipFile(io.BytesIO(results_response.content)) as archive:
        assert archive.namelist() == ["mol1.log", "mol2.log"]


def test_rejects_non_gjf_files(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(app, "EXECUTOR", "local")

    client = TestClient(app.app)
    response = client.post(
        "/jobs",
        files=[("files", ("bad.txt", b"nope", "text/plain"))],
    )

    assert response.status_code == 400


def test_rejects_non_gaussian_commands(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(app, "EXECUTOR", "local")

    client = TestClient(app.app)
    response = client.post(
        "/jobs",
        data={"commands_text": "rm -rf /"},
        files=[("files", ("mol.gjf", b"%chk=mol.chk\n\n0 1\n", "text/plain"))],
    )

    assert response.status_code == 400
