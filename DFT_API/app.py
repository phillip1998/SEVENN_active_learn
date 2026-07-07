from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CLUSTER_JOBS_DIR = "/home/ufslab/API_workspace"
GAUSSIAN_BIN = os.getenv("DFT_GAUSSIAN_BIN", "/opt/gaussian/g16/g16")
GAUSSIAN_DIR = str(Path(GAUSSIAN_BIN).parent)
JOBS_DIR = Path(os.getenv("DFT_JOBS_DIR", DEFAULT_CLUSTER_JOBS_DIR)).resolve()
EXECUTOR = os.getenv("DFT_EXECUTOR", "docker_qsub")
RUN_COMMAND = shlex.split(os.getenv("DFT_RUN_COMMAND", "bash run.sh"))
QSUB_IMAGE = os.getenv("DFT_QSUB_IMAGE", "nfs-shared-storage-image")
QSUB_NETWORK_MODE = os.getenv("DFT_QSUB_NETWORK_MODE", "host")
QSUB_WORKDIR = os.getenv("DFT_QSUB_WORKDIR", "/work")
QSUB_COMMAND_TEMPLATE = os.getenv("DFT_QSUB_COMMAND_TEMPLATE", "qsub {submit_script}")
HOST_JOBS_DIR = os.getenv("DFT_HOST_JOBS_DIR", DEFAULT_CLUSTER_JOBS_DIR)
SCHEDULER_JOBS_DIR = os.getenv("DFT_SCHEDULER_JOBS_DIR", HOST_JOBS_DIR)
JOB_DIR_MODE = int(os.getenv("DFT_JOB_DIR_MODE", "777"), 8)
QSUB_EXTRA_VOLUMES = os.getenv(
    "DFT_QSUB_EXTRA_VOLUMES",
    '{"/etc/passwd": {"bind": "/etc/passwd", "mode": "ro"}, "/etc/group": {"bind": "/etc/group", "mode": "ro"}}',
)
QSUB_ENVIRONMENT = os.getenv(
    "DFT_QSUB_ENVIRONMENT",
    '{"SGE_QMASTER": "UFSLAB", "SGE_QMASTER_PORT": "6444", "SGE_EXECD_PORT": "6445"}',
)
QSUB_USER = os.getenv("DFT_QSUB_USER")

SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
NODE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
JOB_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
GAUSSIAN_COMMAND_RE = re.compile(
    r"^(?:g16|[A-Za-z0-9._/-]*/g16)\s+([A-Za-z0-9._/-]+\.gjf)\s*(>>?|2>>?)\s*([A-Za-z0-9._/-]+\.log)\s*$"
)

app = FastAPI(
    title="DFT Calculation API",
    description="Upload Gaussian .gjf files, submit qsub jobs, and download finished .log files.",
    version="0.1.0",
)


class JobFile(BaseModel):
    name: str
    size_bytes: int


class JobStatus(BaseModel):
    job_id: str
    status: str
    created_at: str
    updated_at: str
    input_files: list[JobFile]
    log_files: list[str] = Field(default_factory=list)
    return_code: int | None = None
    scheduler_job_id: str | None = None
    node: str | None = None
    gaussian_job: str | None = None
    scheduler_workdir: str | None = None
    commands: list[str] = Field(default_factory=list)
    message: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_filename(name: str) -> str:
    basename = Path(name).name.strip()
    cleaned = SAFE_NAME_RE.sub("_", basename)
    if not cleaned or cleaned in {".", ".."}:
        raise HTTPException(status_code=400, detail=f"Invalid filename: {name!r}")
    if not cleaned.lower().endswith(".gjf"):
        raise HTTPException(status_code=400, detail=f"Only .gjf files are accepted: {name!r}")
    return cleaned


def validate_node(node: str) -> str:
    cleaned = node.strip()
    if not NODE_RE.fullmatch(cleaned):
        raise HTTPException(status_code=400, detail="Node must contain only letters, numbers, dots, underscores, or hyphens.")
    return cleaned


def validate_job_name(job_name: str) -> str:
    cleaned = job_name.strip()
    if not JOB_NAME_RE.fullmatch(cleaned):
        raise HTTPException(status_code=400, detail="Job name must contain only letters, numbers, dots, underscores, or hyphens.")
    return cleaned


def normalize_commands(commands: list[str] | None, commands_text: str | None, input_files: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    if commands:
        lines.extend(commands)
    if commands_text:
        lines.extend(commands_text.splitlines())

    cleaned = [line.strip() for line in lines if line.strip()]
    if not cleaned:
        return [f"{GAUSSIAN_BIN} {item['name']} >>{Path(item['name']).with_suffix('.log').name}" for item in input_files]

    normalized: list[str] = []
    for line in cleaned:
        if "\x00" in line or "\n" in line or "\r" in line:
            raise HTTPException(status_code=400, detail="Command must be a single line.")
        match = GAUSSIAN_COMMAND_RE.fullmatch(line)
        if not match:
            raise HTTPException(
                status_code=400,
                detail=f"Only Gaussian commands like 'g16 input.gjf >>input.log' are accepted: {line}",
            )
        normalized.append(f"{GAUSSIAN_BIN} {match.group(1)} {match.group(2)}{match.group(3)}")
    return normalized


def command_log_files(commands: list[str]) -> list[str]:
    log_files: list[str] = []
    for command in commands:
        match = GAUSSIAN_COMMAND_RE.fullmatch(command.strip())
        if match:
            log_files.append(Path(match.group(3)).name)
    return sorted(log_files)


def write_gaussian_run_script(
    path: Path,
    *,
    node: str,
    job_name: str,
    gaussian_job: str,
    scheduler_workdir: str,
    commands: list[str],
) -> None:
    script = "\n".join(
        [
            "#!/bin/bash",
            "",
            "",
            "#$ -pe mpi_36 36",
            f"#$ -N {job_name}",
            "#$ -S /bin/bash",
            f"#$ -q all.q@{node}",
            "#$ -V",
            f"#$ -wd {scheduler_workdir}",
            "",
            "",
            'echo "Got $NSLOTS slots."',
            "cat $TMPDIR/machines",
            "",
            "export gr=/opt/gaussian",
            "export g16root=/opt/gaussian",
            f"export GAUSS_EXEDIR={shlex.quote(GAUSSIAN_DIR)}",
            f"export PATH={shlex.quote(GAUSSIAN_DIR)}:$PATH",
            "export GAUSS_SCRDIR=/scratch",
            f"export JOB={shlex.quote(gaussian_job)}",
            "",
            *commands,
            "",
        ]
    )
    run_script = path / "run.sh"
    run_script.write_text(script, encoding="utf-8", newline="\n")
    os.chmod(run_script, 0o755)


def job_dir(job_id: str) -> Path:
    if not re.fullmatch(r"[A-Fa-f0-9-]{36}", job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    path = (JOBS_DIR / job_id).resolve()
    if JOBS_DIR not in path.parents:
        raise HTTPException(status_code=404, detail="Job not found")
    return path


def metadata_path(path: Path) -> Path:
    return path / "job.json"


def read_metadata(path: Path) -> dict[str, Any]:
    meta_file = metadata_path(path)
    if not meta_file.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    with meta_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_metadata(path: Path, data: dict[str, Any]) -> None:
    data["updated_at"] = utc_now()
    tmp_path = metadata_path(path).with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(metadata_path(path))


def list_log_files(path: Path) -> list[str]:
    return sorted(p.name for p in path.glob("*.log") if p.is_file())


def expected_log_files(data: dict[str, Any]) -> list[str]:
    commands = data.get("commands") or []
    if commands:
        return command_log_files(commands)
    return sorted(Path(item["name"]).with_suffix(".log").name for item in data["input_files"])


def logs_are_ready(path: Path, data: dict[str, Any]) -> bool:
    existing = set(list_log_files(path))
    expected = set(expected_log_files(data))
    return bool(expected) and expected.issubset(existing)


def current_status(path: Path) -> JobStatus:
    data = read_metadata(path)
    data["log_files"] = list_log_files(path)
    if data["status"] in {"submitted", "running"} and logs_are_ready(path, data):
        data["status"] = "completed"
        data["message"] = "All expected .log files are available."
        write_metadata(path, data)
    return JobStatus(**data)


def host_job_dir(path: Path, job_id: str) -> str:
    if HOST_JOBS_DIR:
        return str(Path(HOST_JOBS_DIR) / job_id)
    return str(path)


def scheduler_job_dir(path: Path, job_id: str) -> str:
    if SCHEDULER_JOBS_DIR:
        return str(Path(SCHEDULER_JOBS_DIR) / job_id)
    return host_job_dir(path, job_id)


def parse_extra_volumes() -> dict[str, dict[str, str]]:
    try:
        volumes = json.loads(QSUB_EXTRA_VOLUMES)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid DFT_QSUB_EXTRA_VOLUMES JSON: {exc}") from exc
    if not isinstance(volumes, dict):
        raise RuntimeError("DFT_QSUB_EXTRA_VOLUMES must be a JSON object.")
    return volumes


def parse_qsub_environment() -> dict[str, str]:
    try:
        environment = json.loads(QSUB_ENVIRONMENT)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid DFT_QSUB_ENVIRONMENT JSON: {exc}") from exc
    if not isinstance(environment, dict):
        raise RuntimeError("DFT_QSUB_ENVIRONMENT must be a JSON object.")
    return {str(key): str(value) for key, value in environment.items()}


def submit_with_docker_qsub(path: Path, data: dict[str, Any]) -> tuple[int, str, str, str | None]:
    try:
        import docker
    except ImportError as exc:
        raise RuntimeError("Python package 'docker' is required for DFT_EXECUTOR=docker_qsub.") from exc

    if not QSUB_USER:
        raise RuntimeError('DFT_QSUB_USER is required for docker_qsub, for example: DFT_QSUB_USER="$(id -u):$(id -g)"')

    job_id = data["job_id"]
    volumes = parse_extra_volumes()
    environment = parse_qsub_environment()
    host_workdir = host_job_dir(path, job_id)
    scheduler_workdir = scheduler_job_dir(path, job_id)
    volumes[host_workdir] = {"bind": scheduler_workdir, "mode": "rw"}

    script_path = f"{QSUB_WORKDIR}/run.sh"
    host_script_path = f"{host_workdir}/run.sh"
    scheduler_script_path = f"{scheduler_workdir}/run.sh"
    command = QSUB_COMMAND_TEMPLATE.format(
        script=script_path,
        workdir=QSUB_WORKDIR,
        submit_script=scheduler_script_path,
        submit_workdir=scheduler_workdir,
        host_script=host_script_path,
        host_workdir=host_workdir,
        scheduler_script=scheduler_script_path,
        scheduler_workdir=scheduler_workdir,
        job_id=job_id,
    )

    client = docker.from_env()
    try:
        output = client.containers.run(
            QSUB_IMAGE,
            command=command,
            network_mode=QSUB_NETWORK_MODE,
            volumes=volumes,
            working_dir=QSUB_WORKDIR,
            environment=environment,
            user=QSUB_USER,
            remove=True,
            stdout=True,
            stderr=True,
        )
    except docker.errors.ContainerError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr)
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else str(exc.stdout)
        return int(exc.exit_status), stdout, stderr, None

    stdout = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else str(output)
    scheduler_job_id = parse_scheduler_job_id(stdout)
    return 0, stdout, "", scheduler_job_id


def parse_scheduler_job_id(stdout: str) -> str | None:
    # SGE/Grid Engine usually prints: Your job 12345 ("name") has been submitted.
    for token in stdout.replace("\n", " ").split():
        if re.match(r"^\d+(\D.*)?$", token):
            return token.strip()
    return None


def run_submission(path: Path, data: dict[str, Any]) -> tuple[int, str, str, str | None]:
    if EXECUTOR == "local":
        completed = subprocess.run(
            RUN_COMMAND,
            cwd=path,
            text=True,
            capture_output=True,
            check=False,
        )
        return completed.returncode, completed.stdout, completed.stderr, None
    if EXECUTOR == "docker_qsub":
        return submit_with_docker_qsub(path, data)
    raise RuntimeError(f"Unknown DFT_EXECUTOR: {EXECUTOR}")


def submit_job_to_executor(path: Path) -> None:
    data = read_metadata(path)
    try:
        data["status"] = "running" if EXECUTOR == "local" else "submitting"
        data["message"] = "Calculation submission started."
        write_metadata(path, data)

        return_code, stdout, stderr, scheduler_job_id = run_submission(path, data)
        (path / "runner.stdout").write_text(stdout, encoding="utf-8", errors="replace")
        (path / "runner.stderr").write_text(stderr, encoding="utf-8", errors="replace")

        data = read_metadata(path)
        data["return_code"] = return_code
        data["scheduler_job_id"] = scheduler_job_id
        if return_code == 0:
            if EXECUTOR == "local":
                data["status"] = "completed" if logs_are_ready(path, data) else "failed"
                data["message"] = "Calculation completed." if data["status"] == "completed" else "Calculation finished but expected logs are missing."
            else:
                data["status"] = "submitted"
                data["message"] = "Calculation submitted to scheduler."
        else:
            data["status"] = "failed"
            data["message"] = f"Calculation submission exited with code {return_code}."
        write_metadata(path, data)
    except Exception as exc:
        data = read_metadata(path)
        data["status"] = "failed"
        data["message"] = f"Server failed to submit calculation: {exc}"
        write_metadata(path, data)


@app.on_event("startup")
def ensure_job_root() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/jobs", response_model=JobStatus, status_code=202)
async def submit_job(
    files: list[UploadFile] = File(...),
    node: str = Form("node03"),
    commands: list[str] | None = Form(None),
    commands_text: str | None = Form(None),
    job_name: str = Form("Gau_jyp"),
    gaussian_job: str = Form("test0706"),
) -> JobStatus:
    if not files:
        raise HTTPException(status_code=400, detail="At least one .gjf file is required.")

    job_id = str(uuid.uuid4())
    path = job_dir(job_id)
    path.mkdir(parents=True, exist_ok=False)
    os.chmod(path, JOB_DIR_MODE)

    input_files: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    try:
        node = validate_node(node)
        job_name = validate_job_name(job_name)
        gaussian_job = validate_job_name(gaussian_job)
        for upload in files:
            filename = safe_filename(upload.filename or "")
            if filename in seen_names:
                raise HTTPException(status_code=400, detail=f"Duplicate filename: {filename}")
            seen_names.add(filename)

            destination = path / filename
            size = 0
            with destination.open("wb") as f:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    f.write(chunk)
            input_files.append({"name": filename, "size_bytes": size})

        command_lines = normalize_commands(commands, commands_text, input_files)
        scheduler_workdir = scheduler_job_dir(path, job_id)
        write_gaussian_run_script(
            path,
            node=node,
            job_name=job_name,
            gaussian_job=gaussian_job,
            scheduler_workdir=scheduler_workdir,
            commands=command_lines,
        )

        now = utc_now()
        metadata = {
            "job_id": job_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "input_files": input_files,
            "log_files": [],
            "return_code": None,
            "scheduler_job_id": None,
            "node": node,
            "gaussian_job": gaussian_job,
            "scheduler_workdir": scheduler_workdir,
            "commands": command_lines,
            "message": "Job queued.",
        }
        write_metadata(path, metadata)

        thread = threading.Thread(target=submit_job_to_executor, args=(path,), daemon=True)
        thread.start()
        return current_status(path)
    except Exception:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        raise


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    return current_status(job_dir(job_id))


@app.get("/jobs/{job_id}/status", response_model=JobStatus)
def get_job_status(job_id: str) -> JobStatus:
    return current_status(job_dir(job_id))


@app.get("/jobs/{job_id}/results")
def download_results(job_id: str) -> FileResponse:
    path = job_dir(job_id)
    status = current_status(path)
    if status.status != "completed":
        raise HTTPException(status_code=409, detail=f"Job is still {status.status}.")
    if not status.log_files:
        raise HTTPException(status_code=404, detail="No .log result files are available.")

    zip_path = path / "results.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for log_name in status.log_files:
            archive.write(path / log_name, arcname=log_name)

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{job_id}_logs.zip",
    )


@app.get("/jobs/{job_id}/logs/{log_name}")
def download_log(job_id: str, log_name: str) -> FileResponse:
    path = job_dir(job_id)
    filename = Path(log_name).name
    if filename != log_name or not filename.lower().endswith(".log"):
        raise HTTPException(status_code=400, detail="Invalid log filename.")

    log_path = path / filename
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log file not found.")

    return FileResponse(log_path, media_type="text/plain", filename=filename)
