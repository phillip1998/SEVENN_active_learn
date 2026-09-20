from __future__ import annotations

import json
import math
import mimetypes
import os
import shutil
import subprocess
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .solution_sampling import validate_sampling
from .active_learning import sample_active_learning_clusters
from .cluster_sampling import sample_dft_clusters
from .gaussian import convert_xyz_to_gjf_batch, write_gaussian_manifest
from .lammps import prepare_sevennet_lammps_from_gro, completed_md_data
from .model_paths import resolve_checkpoint
from .sevennet_finetune import prepare_sevennet_finetune


TERMINAL_DFT_STATUSES = {"completed", "failed"}
DEFAULT_MD_RUN_COMMAND = """mpirun --allow-run-as-root -np 3 bash -c '
  ulimit -s unlimited
  export PYTORCH_NO_CUDA_MEMORY_CACHING=1
  export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
  lmp -in in.sevennet.lmp
'"""


@dataclass(frozen=True)
class DftApiConfig:
    base_url: str = "http://localhost:8000"
    nodes: tuple[str, ...] = ("node01",)
    job_name: str = "Gau_AL"
    gaussian_job_prefix: str = "active_learning"
    route: str = "#p wb97xd/def2svp nosymm force scf=tight"
    charge: int = 0
    multiplicity: int = 1
    mem: str = "16GB"
    nprocshared: int = 16
    submit: bool = True


@dataclass(frozen=True)
class SamplingConfig:
    mode: str = "film"
    solute_resnames: tuple[str, ...] = ()
    solvent_resnames: tuple[str, ...] = ()
    solution_solute_weights: dict[int, float] = field(default_factory=lambda: {0: 0.25, 1: 0.75})

    def __post_init__(self):
        validate_sampling(self.mode, self.solute_resnames, self.solvent_resnames, self.solution_solute_weights)

    initial_total: int = 24
    min_total: int = 4
    max_total: int = 96
    target_loop_hours: float = 12.0
    cluster_sizes: tuple[int, ...] = (2, 3, 4)
    initial_weights: dict[int, float] = field(default_factory=lambda: {2: 0.85, 3: 0.15, 4: 0.0})
    mature_weights: dict[int, float] = field(default_factory=lambda: {2: 0.25, 3: 0.45, 4: 0.30})
    weight_ramp_steps: int = 6
    frame_stride: int = 1
    max_frames: int | None = None
    random_frame_samples: int | None = 8
    candidate_seeds_per_frame: int | None = None
    neighbor_pool: int = 8
    contact_cutoff_angstrom: float = 5.0
    close_contact_alert_angstrom: float = 1.2
    hard_reject_distance_angstrom: float = 0.55
    random_seed: int = 17


@dataclass(frozen=True)
class MdConfig:
    gro_path: str
    start_mode: str = "previous"
    xtc_path: str | None = None
    model_path: str = "7net-omni"
    modal: str = "omol25_low"
    pair_style: str = "e3gnn/parallel"
    parallel_model_count: int | None = 4
    enable_flash: bool = True
    elements: tuple[str, ...] | None = None
    temperature_k: float = 300.0
    timestep_ps: float = 0.001
    run_steps: int = 1000
    thermo_interval: int = 100
    dump_interval: int = 100
    ensemble: str = "nvt"
    seed: int = 12345
    d3: bool = False
    run_command: str | None = DEFAULT_MD_RUN_COMMAND
    expected_dump: str = "dump.sevennet.lammpstrj"
    export_whole_xtc: bool = False
    analysis_topology_path: str | None = None


@dataclass(frozen=True)
class FinetuneConfig:
    pretrained: str = "7net-omni"
    modal: str = "omol25_low"
    external_log_dirs: tuple[str, ...] = ()
    epoch: int = 20
    batch_size: int = 1
    learning_rate: float = 1.0e-5
    force_loss_weight: float = 10.0
    data_divide_ratio: float = 0.2
    best_metric: str = "Force_RMSE"
    huber_delta: float = 0.1
    train_shift_scale: bool = True
    train_denominator: bool = False
    require_normal_termination: bool = False
    run_command: str | None = None


@dataclass(frozen=True)
class LoopConfig:
    work_dir: str = "active_learning_loop"
    poll_seconds: int = 3600
    max_iterations: int | None = None
    convergence_patience: int = 4
    execute_md: bool = False
    execute_finetune: bool = False
    dft: DftApiConfig = field(default_factory=DftApiConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    md: MdConfig = field(default_factory=lambda: MdConfig(gro_path="YCOL160.gro", xtc_path="YCOL160.xtc"))
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)


@dataclass
class DftBatchState:
    job_id: str
    iteration: int
    node: str
    submitted_at: str
    status: str = "queued"
    gjf_files: list[str] = field(default_factory=list)
    log_files: list[str] = field(default_factory=list)
    results_dir: str | None = None
    downloaded: bool = False
    message: str | None = None


@dataclass
class LoopState:
    iteration: int = 0
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())
    batches: list[DftBatchState] = field(default_factory=list)
    completed_log_dirs: list[str] = field(default_factory=list)
    completed_dft_logs: int = 0
    last_budget_check_at: str | None = None
    last_budget_completed_logs: int = 0
    no_improvement_steps: int = 0
    stopped_reason: str | None = None


class DftApiClient:
    def __init__(self, base_url: str, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_status(self, job_id: str) -> dict[str, Any]:
        return self._json("GET", f"/jobs/{job_id}/status")

    def submit_job(
        self,
        *,
        node: str,
        job_name: str,
        gaussian_job: str,
        gjf_paths: Sequence[Path],
    ) -> dict[str, Any]:
        fields = {
            "node": node,
            "job_name": job_name,
            "gaussian_job": gaussian_job,
        }
        files = [("files", path.name, path.read_bytes(), _guess_mime(path)) for path in gjf_paths]
        body, content_type = _encode_multipart(fields, files)
        return self._json("POST", "/jobs", data=body, content_type=content_type)

    def download_results(self, job_id: str, output_zip: Path) -> None:
        request = urllib.request.Request(f"{self.base_url}/jobs/{job_id}/results", method="GET")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            output_zip.parent.mkdir(parents=True, exist_ok=True)
            output_zip.write_bytes(response.read())

    def _json(
        self,
        method: str,
        path: str,
        data: bytes | None = None,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        headers = {}
        if content_type:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DFT API {method} {path} failed: HTTP {exc.code}: {detail}") from exc


def run_active_learning_loop(
    config_path: str | Path,
    *,
    once: bool = False,
    dry_run: bool = False,
    progress: bool = True,
) -> LoopState:
    """Run or resume the active-learning controller.

    The TOML config is reloaded at the beginning of every iteration, so changing
    loop parameters while the controller sleeps is reflected in the next step.
    """

    config_path = Path(config_path)
    config = load_loop_config(config_path)
    work_dir = Path(config.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    state_path = work_dir / "state.json"
    state = load_loop_state(state_path)

    while True:
        config = load_loop_config(config_path)
        work_dir = Path(config.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        state_path = work_dir / "state.json"
        state.updated_at = _utc_now()

        if config.max_iterations is not None and state.iteration >= config.max_iterations:
            state.stopped_reason = f"Reached max_iterations={config.max_iterations}"
            save_loop_state(state_path, state)
            return state

        _log(progress, f"[loop] iteration {state.iteration}: polling DFT jobs")
        update_dft_jobs(config, state, work_dir, dry_run=dry_run, progress=progress)
        write_loop_summary(work_dir, state)

        if state.no_improvement_steps >= config.convergence_patience:
            state.stopped_reason = (
                f"No configured improvement signal for {state.no_improvement_steps} step(s); "
                "check MD/finetune summaries before continuing."
            )
            save_loop_state(state_path, state)
            return state

        _log(progress, f"[loop] iteration {state.iteration}: preparing MD, sampling, DFT submission")
        run_one_iteration(config, state, work_dir, dry_run=dry_run, progress=progress)
        state.iteration += 1
        state.updated_at = _utc_now()
        save_loop_state(state_path, state)
        write_loop_summary(work_dir, state)

        if once:
            return state
        _log(progress, f"[loop] sleeping for {config.poll_seconds} seconds")
        time.sleep(max(1, config.poll_seconds))


def run_one_iteration(
    config: LoopConfig,
    state: LoopState,
    work_dir: Path,
    *,
    dry_run: bool,
    progress: bool,
) -> None:
    if config.md.export_whole_xtc:
        from .trajectory_export import load_analysis_topology
        if not config.md.analysis_topology_path:
            raise ValueError("md.export_whole_xtc requires md.analysis_topology_path (matching bonded topology/TPR)")
        analysis_topology = load_analysis_topology(config.md.analysis_topology_path, config.md.gro_path)
        if hasattr(analysis_topology, "trajectory"):
            analysis_topology.trajectory.close()
    continuation_data = None
    if config.md.start_mode not in ("initial", "previous"):
        raise ValueError("md.start_mode must be 'initial' or 'previous'")
    if config.md.start_mode == "previous" and state.iteration > 0:
        continuation_data = completed_md_data(
            work_dir / f"iter_{state.iteration - 1:04d}" / "md", config.md.gro_path)
    iteration_dir = work_dir / f"iter_{state.iteration:04d}"
    iteration_dir.mkdir(parents=True, exist_ok=True)


    counts = allocate_cluster_counts(
        total=estimate_next_submission_count(config, state),
        cluster_sizes=config.sampling.cluster_sizes,
        weights=cluster_weights_for_iteration(config.sampling, state.iteration, state.no_improvement_steps),
    )
    sampling_options = dict(sampling_mode=config.sampling.mode,
        solute_resnames=config.sampling.solute_resnames,
        solvent_resnames=config.sampling.solvent_resnames,
        solution_solute_weights=config.sampling.solution_solute_weights)
    samples_dir = iteration_dir / "samples"
    sampled = False
    if state.iteration == 0 and config.md.xtc_path:
        sampler_kind = "initial_xtc"
        for cluster_size, n_samples in counts.items():
            if n_samples <= 0:
                continue
            selected_samples = sample_dft_clusters(
                **sampling_options,
                gro_path=config.md.gro_path,
                xtc_path=config.md.xtc_path,
                cluster_size=cluster_size,
                n_samples=n_samples,
                output_dir=samples_dir / f"{cluster_size}mol",
                frame_stride=config.sampling.frame_stride,
                max_frames=config.sampling.max_frames,
                random_frame_samples=config.sampling.random_frame_samples,
                neighbor_pool=config.sampling.neighbor_pool,
                contact_cutoff_nm=config.sampling.contact_cutoff_angstrom / 10.0,
                min_distance_reject_nm=config.sampling.hard_reject_distance_angstrom / 10.0,
                random_seed=config.sampling.random_seed + state.iteration + cluster_size,
            )
            sampled = sampled or bool(selected_samples)
    else:
        sampler_kind = "md_dump"
        dump_path = work_dir / f"iter_{state.iteration - 1:04d}" / "md" / config.md.expected_dump
        if not dump_path.is_file():
            _log(progress, f"[loop] previous MD dump not found at {dump_path}; DFT sampling is skipped this step")
        else:
            for cluster_size, n_samples in counts.items():
                _log(progress, f"[loop] sampling {n_samples} clusters of size {cluster_size} from previous MD dump")
                if n_samples <= 0:
                    continue
                selected_samples = sample_active_learning_clusters(
                    **sampling_options,
                    dump_path=dump_path,
                    reference_gro=config.md.gro_path,
                    cluster_size=cluster_size,
                    n_samples=n_samples,
                    output_dir=samples_dir / f"{cluster_size}mol",
                    frame_stride=config.sampling.frame_stride,
                    max_frames=config.sampling.max_frames,
                    random_frame_samples=config.sampling.random_frame_samples,
                    candidate_seeds_per_frame=config.sampling.candidate_seeds_per_frame,
                    neighbor_pool=config.sampling.neighbor_pool,
                    contact_cutoff_angstrom=config.sampling.contact_cutoff_angstrom,
                    close_contact_alert_angstrom=config.sampling.close_contact_alert_angstrom,
                    hard_reject_distance_angstrom=config.sampling.hard_reject_distance_angstrom,
                    random_seed=config.sampling.random_seed + state.iteration + cluster_size,
                    progress=progress,
                )
                sampled = sampled or bool(selected_samples)

    _write_iteration_manifest(iteration_dir, sampler_kind, counts, config, state)
    if not sampled:
        _log(progress, "[loop] no samples selected; skipping DFT submission")
    if sampled:
        gjf_dir = iteration_dir / "gjf"
        gaussian_inputs = convert_xyz_to_gjf_batch(
            samples_dir,
            output_dir=gjf_dir,
            route=config.dft.route,
            charge=config.dft.charge,
            multiplicity=config.dft.multiplicity,
            mem=config.dft.mem,
            nprocshared=config.dft.nprocshared,
        )
        write_gaussian_manifest(iteration_dir / "gaussian_manifest.tsv", gaussian_inputs)

        gjf_files = sorted(gjf_dir.rglob("*.gjf"))
        if not gjf_files:
            _log(progress, "[loop] no GJF files were generated; skipping DFT submission")
        elif dry_run or not config.dft.submit:
            _log(progress, f"[loop] prepared {len(gjf_files)} GJF files; DFT submit disabled")
        else:
            submit_gjf_batches(config, state, gjf_files, progress=progress)

    completed_log_dirs = [Path(path) for path in state.completed_log_dirs if Path(path).is_dir()]
    external_log_paths = _existing_external_log_paths(config.finetune.external_log_dirs)
    finetune_sources = [*completed_log_dirs, *external_log_paths]
    if finetune_sources:
        finetune_dir = iteration_dir / "finetune"
        logs_root = iteration_dir / "dft_logs_for_finetune"
        collected_logs = _collect_logs(finetune_sources, logs_root)
        finetune_pretrained = _select_finetune_pretrained(
            config=config,
            work_dir=work_dir,
            current_iteration=state.iteration,
        )
        _write_finetune_source_manifest(
            finetune_dir=finetune_dir,
            pretrained=finetune_pretrained,
            completed_log_dirs=completed_log_dirs,
            external_log_paths=external_log_paths,
            collected_logs=collected_logs,
        )
        _log(progress, f"[loop] preparing SevenNet fine-tune set from {logs_root}")
        _log(progress, f"[loop] fine-tune checkpoint: {finetune_pretrained}")
        prepare_sevennet_finetune(
            output_dir=finetune_dir,
            logs_path=logs_root,
            pretrained=finetune_pretrained,
            modal=config.finetune.modal,
            epoch=config.finetune.epoch,
            batch_size=config.finetune.batch_size,
            learning_rate=config.finetune.learning_rate,
            force_loss_weight=config.finetune.force_loss_weight,
            data_divide_ratio=config.finetune.data_divide_ratio,
            best_metric=config.finetune.best_metric,
            huber_delta=config.finetune.huber_delta,
            train_shift_scale=config.finetune.train_shift_scale,
            train_denominator=config.finetune.train_denominator,
            require_normal_termination=config.finetune.require_normal_termination,
            sevennet_label=f"gaussian_iter_{state.iteration:04d}",
            split_manifest_path=work_dir / "split_manifest.json",
            historical_split_dirs=tuple(work_dir / f"iter_{i:04d}" / "finetune" for i in range(state.iteration)),
        )
        if config.execute_finetune and config.finetune.run_command and not dry_run:
            _run_command(config.finetune.run_command, cwd=finetune_dir, progress=progress)
    else:
        _log(progress, "[loop] no completed DFT logs yet; skipping fine-tune preparation")

    md_dir = iteration_dir / "md"
    model_path, pair_style, parallel_model_count = _prepare_md_model(
        config=config,
        iteration_dir=iteration_dir,
        md_dir=md_dir,
        dry_run=dry_run,
        progress=progress,
    )
    prepare_sevennet_lammps_from_gro(
        gro_path=config.md.gro_path,
        output_dir=md_dir,
        continuation_data_path=continuation_data,
        model_path=model_path,
        elements=config.md.elements,
        pair_style=pair_style,  # type: ignore[arg-type]
        parallel_model_count=parallel_model_count,
        temperature_k=config.md.temperature_k,
        timestep_ps=config.md.timestep_ps,
        run_steps=config.md.run_steps,
        thermo_interval=config.md.thermo_interval,
        dump_interval=config.md.dump_interval,
        ensemble=config.md.ensemble,  # type: ignore[arg-type]
        seed=config.md.seed + state.iteration,
        d3=config.md.d3,
    )
    _write_md_run_script(md_dir, config.md.run_command)
    if config.execute_md and config.md.run_command and not dry_run:
        if config.md.export_whole_xtc:
            for artifact in ("whole.xtc", "whole.gro", "whole_export.json", "whole_export_error.json"):
                (md_dir / artifact).unlink(missing_ok=True)
        for artifact in ("md.complete", "final.data"):
            (md_dir / artifact).unlink(missing_ok=True)
        try:
            _run_command(config.md.run_command, cwd=md_dir, progress=progress)
            completed_md_data(md_dir, config.md.gro_path)
        except Exception:
            (md_dir / "md.complete").unlink(missing_ok=True)
            raise
        if config.md.export_whole_xtc:
            from .trajectory_export import export_whole_trajectory
            try:
                report = export_whole_trajectory(md_dir / config.md.expected_dump,
                    config.md.gro_path, config.md.analysis_topology_path, md_dir,
                    timestep_ps=config.md.timestep_ps)
                _log(progress, f"[analysis] wrote {report['frames']} whole-molecule frames to {md_dir / 'whole.xtc'}")
            except Exception as exc:
                # Analysis output must not invalidate completed MD or resubmit DFT on retry.
                (md_dir / "whole_export_error.json").write_text(json.dumps({
                    "error": str(exc), "dump": str(md_dir / config.md.expected_dump),
                    "iteration": state.iteration,
                }, indent=2) + "\n", encoding="utf-8")
                import warnings
                warnings.warn(f"Analysis XTC export failed; MD remains complete: {exc}", stacklevel=2)


def update_dft_jobs(
    config: LoopConfig,
    state: LoopState,
    work_dir: Path,
    *,
    dry_run: bool,
    progress: bool,
) -> None:
    if dry_run:
        return
    client = DftApiClient(config.dft.base_url)
    for batch in state.batches:
        if batch.status in TERMINAL_DFT_STATUSES and batch.downloaded:
            continue
        try:
            status = client.get_status(batch.job_id)
        except Exception as exc:
            batch.message = str(exc)
            continue
        batch.status = str(status.get("status", batch.status))
        batch.log_files = list(status.get("log_files", batch.log_files) or [])
        batch.message = status.get("message")
        if batch.status == "completed" and not batch.downloaded:
            results_dir = work_dir / f"iter_{batch.iteration:04d}" / "dft_results" / batch.job_id
            zip_path = results_dir / "results.zip"
            client.download_results(batch.job_id, zip_path)
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(results_dir)
            batch.results_dir = str(results_dir)
            batch.downloaded = True
            if str(results_dir) not in state.completed_log_dirs:
                state.completed_log_dirs.append(str(results_dir))
            _log(progress, f"[loop] downloaded DFT results for {batch.job_id} -> {results_dir}")
    state.completed_dft_logs = sum(len(list(Path(path).glob("*.log"))) for path in state.completed_log_dirs if Path(path).is_dir())


def submit_gjf_batches(
    config: LoopConfig,
    state: LoopState,
    gjf_files: Sequence[Path],
    *,
    progress: bool,
) -> None:
    client = DftApiClient(config.dft.base_url)
    nodes = list(config.dft.nodes)
    if not nodes:
        raise ValueError("At least one DFT node is required")
    chunks = split_round_robin(gjf_files, len(nodes))
    for node, chunk in zip(nodes, chunks):
        if not chunk:
            continue
        gaussian_job = f"{config.dft.gaussian_job_prefix}_{state.iteration:04d}_{node}"
        _log(progress, f"[loop] submitting {len(chunk)} GJF files to {node}")
        response = client.submit_job(
            node=node,
            job_name=config.dft.job_name,
            gaussian_job=gaussian_job,
            gjf_paths=chunk,
        )
        state.batches.append(
            DftBatchState(
                job_id=response["job_id"],
                iteration=state.iteration,
                node=node,
                submitted_at=_utc_now(),
                status=response.get("status", "queued"),
                gjf_files=[str(path) for path in chunk],
                log_files=list(response.get("log_files", []) or []),
                message=response.get("message"),
            )
        )


def estimate_next_submission_count(config: LoopConfig, state: LoopState) -> int:
    sampling = config.sampling
    now = _parse_time(_utc_now())
    if state.last_budget_check_at is None:
        state.last_budget_check_at = _utc_now()
        state.last_budget_completed_logs = state.completed_dft_logs
        return sampling.initial_total

    previous_time = _parse_time(state.last_budget_check_at)
    elapsed_hours = max((now - previous_time) / 3600.0, 1e-6)
    completed_delta = max(0, state.completed_dft_logs - state.last_budget_completed_logs)
    state.last_budget_check_at = _utc_now()
    state.last_budget_completed_logs = state.completed_dft_logs
    if completed_delta <= 0:
        pending = sum(1 for batch in state.batches if batch.status not in TERMINAL_DFT_STATUSES)
        conservative = max(sampling.min_total, math.floor(sampling.initial_total / (pending + 1)))
        return min(sampling.max_total, conservative)
    projected = round(completed_delta * sampling.target_loop_hours / elapsed_hours)
    return min(sampling.max_total, max(sampling.min_total, projected))


def cluster_weights_for_iteration(
    sampling: SamplingConfig,
    iteration: int,
    no_improvement_steps: int = 0,
) -> dict[int, float]:
    ramp = min(1.0, max(0.0, (iteration + 0.5 * no_improvement_steps) / max(1, sampling.weight_ramp_steps)))
    weights: dict[int, float] = {}
    for size in sampling.cluster_sizes:
        start = sampling.initial_weights.get(size, 0.0)
        end = sampling.mature_weights.get(size, 0.0)
        weights[size] = (1.0 - ramp) * start + ramp * end
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0.0:
        return {size: 1.0 / len(sampling.cluster_sizes) for size in sampling.cluster_sizes}
    return {size: max(0.0, value) / total for size, value in weights.items()}


def allocate_cluster_counts(total: int, cluster_sizes: Sequence[int], weights: dict[int, float]) -> dict[int, int]:
    raw = {size: max(0.0, weights.get(size, 0.0)) * total for size in cluster_sizes}
    counts = {size: int(math.floor(value)) for size, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(cluster_sizes, key=lambda size: raw[size] - counts[size], reverse=True)
    for size in order[:remaining]:
        counts[size] += 1
    return counts


def load_loop_config(path: str | Path) -> LoopConfig:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    dft = _section(data, "dft")
    sampling = _section(data, "sampling")
    md = _section(data, "md")
    finetune = _section(data, "finetune")
    return LoopConfig(
        work_dir=str(data.get("work_dir", "active_learning_loop")),
        poll_seconds=int(data.get("poll_seconds", 3600)),
        max_iterations=_optional_int(data.get("max_iterations")),
        convergence_patience=int(data.get("convergence_patience", 4)),
        execute_md=bool(data.get("execute_md", False)),
        execute_finetune=bool(data.get("execute_finetune", False)),
        dft=DftApiConfig(
            base_url=str(dft.get("base_url", "http://localhost:8000")),
            nodes=tuple(str(item) for item in dft.get("nodes", ["node01"])),
            job_name=str(dft.get("job_name", "Gau_AL")),
            gaussian_job_prefix=str(dft.get("gaussian_job_prefix", "active_learning")),
            route=str(dft.get("route", "#p wb97xd/def2svp nosymm force scf=tight")),
            charge=int(dft.get("charge", 0)),
            multiplicity=int(dft.get("multiplicity", 1)),
            mem=str(dft.get("mem", "16GB")),
            nprocshared=int(dft.get("nprocshared", 16)),
            submit=bool(dft.get("submit", True)),
        ),
        sampling=SamplingConfig(
            mode=str(sampling.get("mode", "film")),
            solute_resnames=tuple(sampling.get("solute_resnames", [])),
            solvent_resnames=tuple(sampling.get("solvent_resnames", [])),
            solution_solute_weights=_int_key_dict(sampling.get("solution_solute_weights", {"0": 0.25, "1": 0.75})),
            initial_total=int(sampling.get("initial_total", 24)),
            min_total=int(sampling.get("min_total", 4)),
            max_total=int(sampling.get("max_total", 96)),
            target_loop_hours=float(sampling.get("target_loop_hours", 12.0)),
            cluster_sizes=tuple(int(item) for item in sampling.get("cluster_sizes", [2, 3, 4])),
            initial_weights=_int_key_dict(sampling.get("initial_weights", {"2": 0.85, "3": 0.15, "4": 0.0})),
            mature_weights=_int_key_dict(sampling.get("mature_weights", {"2": 0.25, "3": 0.45, "4": 0.30})),
            weight_ramp_steps=int(sampling.get("weight_ramp_steps", 6)),
            frame_stride=int(sampling.get("frame_stride", 1)),
            max_frames=_optional_int(sampling.get("max_frames")),
            random_frame_samples=_optional_int(sampling.get("random_frame_samples", 8)),
            candidate_seeds_per_frame=_optional_int(sampling.get("candidate_seeds_per_frame")),
            neighbor_pool=int(sampling.get("neighbor_pool", 8)),
            contact_cutoff_angstrom=float(sampling.get("contact_cutoff_angstrom", 5.0)),
            close_contact_alert_angstrom=float(sampling.get("close_contact_alert_angstrom", 1.2)),
            hard_reject_distance_angstrom=float(sampling.get("hard_reject_distance_angstrom", 0.55)),
            random_seed=int(sampling.get("random_seed", 17)),
        ),
        md=MdConfig(
            gro_path=str(md.get("gro_path", "YCOL160.gro")),
            start_mode=str(md.get("start_mode", "previous")),
            xtc_path=None if md.get("xtc_path") is None else str(md.get("xtc_path")),
            model_path=str(md.get("model_path", "7net-omni")),
            modal=str(md.get("modal", "omol25_low")),
            pair_style=str(md.get("pair_style", "e3gnn/parallel")),
            parallel_model_count=_optional_int(md.get("parallel_model_count", 4)),
            enable_flash=bool(md.get("enable_flash", True)),
            elements=None if md.get("elements") is None else tuple(str(item) for item in md.get("elements", [])),
            temperature_k=float(md.get("temperature_k", 300.0)),
            timestep_ps=float(md.get("timestep_ps", 0.001)),
            run_steps=int(md.get("run_steps", 1000)),
            thermo_interval=int(md.get("thermo_interval", 100)),
            dump_interval=int(md.get("dump_interval", 100)),
            ensemble=str(md.get("ensemble", "nvt")),
            seed=int(md.get("seed", 12345)),
            d3=bool(md.get("d3", False)),
            run_command=str(md.get("run_command", DEFAULT_MD_RUN_COMMAND)),
            expected_dump=str(md.get("expected_dump", "dump.sevennet.lammpstrj")),
            export_whole_xtc=bool(md.get("export_whole_xtc", False)),
            analysis_topology_path=str(md["analysis_topology_path"]) if md.get("analysis_topology_path") else None,
        ),
        finetune=FinetuneConfig(
            pretrained=str(finetune.get("pretrained", "7net-omni")),
            modal=str(finetune.get("modal", "omol25_low")),
            external_log_dirs=tuple(str(item) for item in finetune.get("external_log_dirs", [])),
            epoch=int(finetune.get("epoch", 20)),
            batch_size=int(finetune.get("batch_size", 1)),
            learning_rate=float(finetune.get("learning_rate", 1.0e-5)),
            force_loss_weight=float(finetune.get("force_loss_weight", 10.0)),
            data_divide_ratio=float(finetune.get("data_divide_ratio", 0.2)),
            best_metric=str(finetune.get("best_metric", "Force_RMSE")),
            huber_delta=float(finetune.get("huber_delta", 0.1)),
            train_shift_scale=bool(finetune.get("train_shift_scale", True)),
            train_denominator=bool(finetune.get("train_denominator", False)),
            require_normal_termination=bool(finetune.get("require_normal_termination", False)),
            run_command=None if finetune.get("run_command") is None else str(finetune.get("run_command")),
        ),
    )


def load_loop_state(path: Path) -> LoopState:
    if not path.is_file():
        return LoopState()
    data = json.loads(path.read_text(encoding="utf-8"))
    batches = [DftBatchState(**item) for item in data.pop("batches", [])]
    state = LoopState(**data)
    state.batches = batches
    return state


def save_loop_state(path: Path, state: LoopState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(state)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def write_loop_summary(work_dir: Path, state: LoopState) -> None:
    lines = [
        "# Active Learning Loop Summary",
        "",
        f"- iteration: {state.iteration}",
        f"- completed_dft_logs: {state.completed_dft_logs}",
        f"- pending_batches: {sum(1 for batch in state.batches if batch.status not in TERMINAL_DFT_STATUSES)}",
        f"- no_improvement_steps: {state.no_improvement_steps}",
        f"- stopped_reason: {state.stopped_reason or ''}",
        "",
        "## DFT batches",
        "",
        "| iteration | job_id | node | status | logs | downloaded |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for batch in state.batches:
        lines.append(
            f"| {batch.iteration} | {batch.job_id} | {batch.node} | {batch.status} | "
            f"{len(batch.log_files)} | {batch.downloaded} |"
        )
    (work_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def split_round_robin(items: Sequence[Path], n_chunks: int) -> list[list[Path]]:
    chunks = [[] for _ in range(n_chunks)]
    for index, item in enumerate(items):
        chunks[index % n_chunks].append(item)
    return chunks


def _prepare_md_model(
    *,
    config: LoopConfig,
    iteration_dir: Path,
    md_dir: Path,
    dry_run: bool,
    progress: bool,
) -> tuple[str, str, int | None]:
    md_dir.mkdir(parents=True, exist_ok=True)
    pair_style = config.md.pair_style
    model_path = Path(_select_model_path(config, iteration_dir))
    if pair_style != "e3gnn/parallel":
        return str(_absolute_path(model_path)), pair_style, config.md.parallel_model_count

    if model_path.is_dir():
        absolute_model_path = _absolute_path(model_path)
        return str(absolute_model_path), pair_style, _parallel_model_count(absolute_model_path, config.md.parallel_model_count)

    deployed_dir = _absolute_path(md_dir / "deployed_parallel")
    deploy_command = _sevennet_parallel_deploy_command(
        checkpoint=resolve_checkpoint(str(model_path)),
        modal=config.md.modal,
        output_dir=deployed_dir,
        enable_flash=config.md.enable_flash,
    )
    deploy_script = md_dir / "deploy_model.sh"
    deploy_script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n" + deploy_command + "\n",
        encoding="utf-8",
        newline="\n",
    )

    if not deployed_dir.is_dir() and config.execute_md and not dry_run:
        _run_command(deploy_command, cwd=md_dir, progress=progress)

    return str(deployed_dir), pair_style, _parallel_model_count(deployed_dir, config.md.parallel_model_count)


def _write_md_run_script(md_dir: Path, command: str | None) -> None:
    if not command:
        return
    (md_dir / "run_md.sh").write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\n\ncd -- "$(dirname -- "$0")"\nrm -f -- md.complete final.data\ntrap "rm -f -- md.complete" ERR\n' + command.rstrip() + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _absolute_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _select_model_path(config: LoopConfig, iteration_dir: Path) -> str:
    # Retain the last successfully deployed model when no new labels are ready.
    current = int(iteration_dir.name.removeprefix("iter_"))
    for iteration in range(current, -1, -1):
        finetune_dir = iteration_dir.parent / f"iter_{iteration:04d}" / "finetune"
        if config.md.pair_style == "e3gnn/parallel":
            model = finetune_dir / "deployed_parallel"
            if model.is_dir() and any(model.glob("deployed_parallel_*.pt")):
                return str(model)
        else:
            model = finetune_dir / "deployed_serial.pt"
            if model.is_file():
                return str(model)
    return config.md.model_path


def _select_finetune_pretrained(
    *,
    config: LoopConfig,
    work_dir: Path,
    current_iteration: int,
) -> str:
    checkpoint = _latest_previous_finetune_checkpoint(work_dir, current_iteration)
    if checkpoint is not None:
        return str(_absolute_path(checkpoint))
    return resolve_checkpoint(config.finetune.pretrained)


def _latest_previous_finetune_checkpoint(work_dir: Path, current_iteration: int) -> Path | None:
    for iteration in range(current_iteration - 1, -1, -1):
        checkpoint = work_dir / f"iter_{iteration:04d}" / "finetune" / "checkpoint_best.pth"
        if checkpoint.is_file():
            return checkpoint
    return None


def _sevennet_parallel_deploy_command(
    *,
    checkpoint: str,
    modal: str,
    output_dir: Path,
    enable_flash: bool,
) -> str:
    parts = ["sevenn", "get_model", _shell_quote(checkpoint), "--get_parallel"]
    if enable_flash:
        parts.append("--enable_flash")
    if modal:
        parts.extend(["--modal", _shell_quote(modal)])
    parts.extend(["-o", _shell_quote(output_dir.name)])
    return " ".join(parts)


def _parallel_model_count(model_dir: Path, fallback: int | None) -> int | None:
    if model_dir.is_dir():
        count = len(list(model_dir.glob("deployed_parallel_*.pt")))
        if count > 0:
            return count
    return fallback


def _shell_quote(value: str) -> str:
    if not value:
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:-")
    if all(char in safe for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _collect_logs(sources: Sequence[Path], dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    collected: list[Path] = []
    for source in sources:
        source_logs = _log_files_under(source)
        for log_path in source_logs:
            if source.is_file():
                relative_stem = log_path.stem
            else:
                relative_stem = "_".join(log_path.relative_to(source).with_suffix("").parts)
            target = _unique_log_target(dest, source, relative_stem, log_path.suffix)
            if not target.exists():
                shutil.copyfile(log_path, target)
            collected.append(target)
    return collected


def _existing_external_log_paths(paths: Sequence[str]) -> list[Path]:
    existing: list[Path] = []
    for item in paths:
        path = Path(item).expanduser()
        if path.exists():
            existing.append(path)
    return existing


def _log_files_under(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in {".log", ".out"} else []
    logs: list[Path] = []
    for pattern in ("*.log", "*.out"):
        logs.extend(path.rglob(pattern))
    return sorted(set(logs))


def _unique_log_target(dest: Path, source: Path, relative_stem: str, suffix: str) -> Path:
    source_name = source.stem if source.is_file() else source.name
    target = dest / f"{source_name}_{relative_stem}{suffix}"
    if not target.exists():
        return target
    index = 2
    while True:
        candidate = dest / f"{source_name}_{relative_stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _write_finetune_source_manifest(
    *,
    finetune_dir: Path,
    pretrained: str,
    completed_log_dirs: Sequence[Path],
    external_log_paths: Sequence[Path],
    collected_logs: Sequence[Path],
) -> None:
    finetune_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": _utc_now(),
        "pretrained": pretrained,
        "completed_log_dirs": [str(path) for path in completed_log_dirs],
        "external_log_paths": [str(path) for path in external_log_paths],
        "n_collected_logs": len(collected_logs),
        "collected_logs": [str(path) for path in collected_logs],
    }
    (finetune_dir / "finetune_sources.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_iteration_manifest(
    iteration_dir: Path,
    sampler_kind: str,
    counts: dict[int, int],
    config: LoopConfig,
    state: LoopState,
) -> None:
    weights = cluster_weights_for_iteration(config.sampling, state.iteration, state.no_improvement_steps)
    manifest = {
        "iteration": state.iteration,
        "created_at": _utc_now(),
        "sampler": sampler_kind,
        "sampling_mode": config.sampling.mode,
        "solute_resnames": config.sampling.solute_resnames,
        "solvent_resnames": config.sampling.solvent_resnames,
        "solution_solute_weights": config.sampling.solution_solute_weights,
        "cluster_counts": counts,
        "cluster_weights": weights,
        "dft_nodes": list(config.dft.nodes),
    }
    (iteration_dir / "iteration_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _run_command(command: str, cwd: Path, progress: bool) -> None:
    _log(progress, f"[loop] running command in {cwd}: {command}")
    subprocess.run(command, cwd=cwd, shell=True, check=True)


def _encode_multipart(
    fields: dict[str, str],
    files: Sequence[tuple[str, str, bytes, str]],
) -> tuple[bytes, str]:
    boundary = f"----mlpmdloop{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        chunks.append(str(value).encode())
        chunks.append(b"\r\n")
    for field_name, filename, content, content_type in files:
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(
            (
                f'Content-Disposition: form-data; name="{field_name}"; '
                f'filename="{Path(filename).name}"\r\n'
            ).encode()
        )
        chunks.append(f"Content-Type: {content_type}\r\n\r\n".encode())
        chunks.append(content)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _guess_mime(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _int_key_dict(value: dict[Any, Any]) -> dict[int, float]:
    return {int(key): float(item) for key, item in value.items()}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def _log(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)
