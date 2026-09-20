from __future__ import annotations

import json
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import QProcess, QProcessEnvironment, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QPlainTextEdit,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


EXAMPLE_CONFIG = ROOT / "active_learning_loop.example.toml"


class ActiveLearningGui(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MLP-MD Active Learning Loop")
        self.resize(1180, 780)
        self.config_path: Path | None = None
        self.process: QProcess | None = None

        central = QWidget()
        root = QVBoxLayout(central)
        root.addLayout(self._build_top_bar())
        root.addWidget(self._build_tabs(), stretch=2)
        root.addLayout(self._build_run_bar())
        root.addWidget(self._build_log_box(), stretch=1)
        self.setCentralWidget(central)

        if EXAMPLE_CONFIG.is_file():
            self.load_config(EXAMPLE_CONFIG)

    def _build_top_bar(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self.config_label = QLabel("Config: not saved")
        open_btn = QPushButton("Open config")
        open_btn.clicked.connect(self.open_config)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_config)
        save_as_btn = QPushButton("Save as")
        save_as_btn.clicked.connect(self.save_config_as)
        layout.addWidget(self.config_label, stretch=1)
        layout.addWidget(open_btn)
        layout.addWidget(save_btn)
        layout.addWidget(save_as_btn)
        return layout

    def _build_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.addTab(self._loop_tab(), "Loop")
        tabs.addTab(self._dft_tab(), "DFT")
        tabs.addTab(self._sampling_tab(), "Sampling")
        tabs.addTab(self._md_tab(), "MD")
        tabs.addTab(self._finetune_tab(), "Fine-tune")
        tabs.addTab(self._utilities_tab(), "Utilities")
        tabs.addTab(self._status_tab(), "State")
        return tabs

    def _loop_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.work_dir = PathEdit(select_dir=True)
        self.poll_seconds = int_spin(1, 1_000_000, 3600)
        self.max_iterations = OptionalIntEdit(20)
        self.convergence_patience = int_spin(1, 10_000, 4)
        self.execute_md = QCheckBox("Run LAMMPS/MD command during loop")
        self.execute_finetune = QCheckBox("Run SevenNet fine-tune command during loop")
        form.addRow("Work directory", self.work_dir)
        form.addRow("Poll seconds", self.poll_seconds)
        form.addRow("Max iterations", self.max_iterations)
        form.addRow("Convergence patience", self.convergence_patience)
        form.addRow("Execute MD", self.execute_md)
        form.addRow("Execute fine-tune", self.execute_finetune)
        return page

    def _dft_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.dft_base_url = QLineEdit()
        self.dft_nodes = QLineEdit()
        self.dft_job_name = QLineEdit()
        self.dft_job_prefix = QLineEdit()
        self.dft_route = QLineEdit()
        self.dft_charge = int_spin(-100, 100, 0)
        self.dft_multiplicity = int_spin(1, 100, 1)
        self.dft_mem = QLineEdit()
        self.dft_nprocshared = int_spin(1, 4096, 16)
        self.dft_submit = QCheckBox("Submit generated GJF files to DFT API")
        form.addRow("API base URL", self.dft_base_url)
        form.addRow("Nodes (comma-separated)", self.dft_nodes)
        form.addRow("Job name", self.dft_job_name)
        form.addRow("Gaussian job prefix", self.dft_job_prefix)
        form.addRow("Route", self.dft_route)
        form.addRow("Charge", self.dft_charge)
        form.addRow("Multiplicity", self.dft_multiplicity)
        form.addRow("Memory", self.dft_mem)
        form.addRow("NProcShared", self.dft_nprocshared)
        form.addRow("Submit", self.dft_submit)
        return page

    def _sampling_tab(self) -> QWidget:
        page = QWidget()
        grid = QGridLayout(page)
        basic = QGroupBox("Counts")
        form = QFormLayout(basic)
        self.sampling_mode = QComboBox()
        self.sampling_mode.addItems(["film", "solution"])
        self.solute_resnames = QLineEdit()
        self.solvent_resnames = QLineEdit()
        self.solution_solute_weights = QLineEdit()
        form.addRow("Sampling mode", self.sampling_mode)
        form.addRow("Solute resnames", self.solute_resnames)
        form.addRow("Solvent resnames", self.solvent_resnames)
        form.addRow("Solute count weights", self.solution_solute_weights)
        self.initial_total = int_spin(1, 1_000_000, 24)
        self.min_total = int_spin(0, 1_000_000, 4)
        self.max_total = int_spin(1, 1_000_000, 96)
        self.target_loop_hours = float_spin(0.01, 100_000.0, 12.0, 2)
        self.cluster_sizes = QLineEdit()
        form.addRow("Initial total", self.initial_total)
        form.addRow("Min total", self.min_total)
        form.addRow("Max total", self.max_total)
        form.addRow("Target loop hours", self.target_loop_hours)
        form.addRow("Cluster sizes", self.cluster_sizes)

        weights = QGroupBox("Weights")
        weight_form = QFormLayout(weights)
        self.initial_weights = QLineEdit()
        self.mature_weights = QLineEdit()
        self.weight_ramp_steps = int_spin(1, 1_000_000, 6)
        weight_form.addRow("Initial weights", self.initial_weights)
        weight_form.addRow("Mature weights", self.mature_weights)
        weight_form.addRow("Ramp steps", self.weight_ramp_steps)

        filters = QGroupBox("Sampling filters")
        filter_form = QFormLayout(filters)
        self.frame_stride = int_spin(1, 1_000_000, 1)
        self.max_frames = OptionalIntEdit(100)
        self.random_frame_samples = OptionalIntEdit(8)
        self.candidate_seeds_per_frame = OptionalIntEdit(None)
        self.neighbor_pool = int_spin(1, 1_000_000, 8)
        self.contact_cutoff = float_spin(0.0, 1000.0, 5.0, 3)
        self.close_contact_alert = float_spin(0.0, 1000.0, 1.2, 3)
        self.hard_reject_distance = float_spin(0.0, 1000.0, 0.55, 3)
        self.random_seed = int_spin(0, 2_147_483_647, 17)
        filter_form.addRow("Frame stride", self.frame_stride)
        filter_form.addRow("Max frames", self.max_frames)
        filter_form.addRow("Random frame samples", self.random_frame_samples)
        filter_form.addRow("Candidate seeds/frame", self.candidate_seeds_per_frame)
        filter_form.addRow("Neighbor pool", self.neighbor_pool)
        filter_form.addRow("Contact cutoff A", self.contact_cutoff)
        filter_form.addRow("Close contact alert A", self.close_contact_alert)
        filter_form.addRow("Hard reject distance A", self.hard_reject_distance)
        filter_form.addRow("Random seed", self.random_seed)

        grid.addWidget(basic, 0, 0)
        grid.addWidget(weights, 0, 1)
        grid.addWidget(filters, 1, 0, 1, 2)
        return page

    def _md_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.md_gro_path = PathEdit()
        self.md_xtc_path = PathEdit()
        self.md_start_mode = QComboBox()
        self.md_start_mode.addItems(["previous", "initial"])
        self.md_export_whole_xtc = QCheckBox("Export whole-molecule XTC after MD")
        self.md_analysis_topology = PathEdit()
        self.md_model_path = PathEdit()
        self.md_modal = QLineEdit()
        self.md_pair_style = QComboBox()
        self.md_pair_style.addItems(["e3gnn/parallel", "e3gnn"])
        self.md_parallel_model_count = OptionalIntEdit(4)
        self.md_enable_flash = QCheckBox("Enable SevenNet flash option")
        self.md_elements = QLineEdit()
        self.md_temperature = float_spin(0.0, 100_000.0, 300.0, 3)
        self.md_timestep = float_spin(0.0, 1000.0, 0.001, 6)
        self.md_run_steps = int_spin(1, 2_147_483_647, 1000)
        self.md_thermo_interval = int_spin(1, 2_147_483_647, 100)
        self.md_dump_interval = int_spin(1, 2_147_483_647, 100)
        self.md_ensemble = QComboBox()
        self.md_ensemble.addItems(["nvt", "nve"])
        self.md_seed = int_spin(0, 2_147_483_647, 12345)
        self.md_d3 = QCheckBox("Enable D3")
        self.md_expected_dump = QLineEdit()
        self.md_run_command = QPlainTextEdit()
        self.md_run_command.setMinimumHeight(96)
        form.addRow("GRO path", self.md_gro_path)
        form.addRow("XTC path", self.md_xtc_path)
        form.addRow("MD start mode", self.md_start_mode)
        form.addRow("Analysis XTC", self.md_export_whole_xtc)
        form.addRow("Analysis topology (TPR)", self.md_analysis_topology)
        form.addRow("Model path", self.md_model_path)
        form.addRow("Modal", self.md_modal)
        form.addRow("Pair style", self.md_pair_style)
        form.addRow("Parallel model count", self.md_parallel_model_count)
        form.addRow("Flash", self.md_enable_flash)
        form.addRow("Elements", self.md_elements)
        form.addRow("Temperature K", self.md_temperature)
        form.addRow("Timestep ps", self.md_timestep)
        form.addRow("Run steps", self.md_run_steps)
        form.addRow("Thermo interval", self.md_thermo_interval)
        form.addRow("Dump interval", self.md_dump_interval)
        form.addRow("Ensemble", self.md_ensemble)
        form.addRow("Seed", self.md_seed)
        form.addRow("D3", self.md_d3)
        form.addRow("Expected dump", self.md_expected_dump)
        form.addRow("Run command", self.md_run_command)
        return page

    def _finetune_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.ft_pretrained = PathEdit()
        self.ft_modal = QLineEdit()
        self.ft_external_log_dirs = QPlainTextEdit()
        self.ft_external_log_dirs.setPlaceholderText("One file or directory per line")
        self.ft_external_log_dirs.setMinimumHeight(70)
        self.ft_epoch = int_spin(1, 1_000_000, 20)
        self.ft_batch_size = int_spin(1, 1_000_000, 1)
        self.ft_learning_rate = float_spin(0.0, 1.0, 1.0e-5, 8)
        self.ft_force_loss_weight = float_spin(0.0, 100_000.0, 10.0, 3)
        self.ft_data_divide_ratio = float_spin(0.0, 1.0, 0.2, 3)
        self.ft_best_metric = QLineEdit()
        self.ft_huber_delta = float_spin(0.0, 1000.0, 0.1, 3)
        self.ft_train_shift_scale = QCheckBox("Train shift/scale")
        self.ft_train_denominator = QCheckBox("Train denominator")
        self.ft_require_normal_termination = QCheckBox("Require Gaussian normal termination")
        self.ft_run_command = QLineEdit()
        form.addRow("Pretrained", self.ft_pretrained)
        form.addRow("Modal", self.ft_modal)
        form.addRow("External log dirs", self.ft_external_log_dirs)
        form.addRow("Epoch", self.ft_epoch)
        form.addRow("Batch size", self.ft_batch_size)
        form.addRow("Learning rate", self.ft_learning_rate)
        form.addRow("Force loss weight", self.ft_force_loss_weight)
        form.addRow("Data divide ratio", self.ft_data_divide_ratio)
        form.addRow("Best metric", self.ft_best_metric)
        form.addRow("Huber delta", self.ft_huber_delta)
        form.addRow("Train shift/scale", self.ft_train_shift_scale)
        form.addRow("Train denominator", self.ft_train_denominator)
        form.addRow("Require normal termination", self.ft_require_normal_termination)
        form.addRow("Run command", self.ft_run_command)
        return page

    def _status_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        refresh_btn = QPushButton("Refresh state/summary")
        refresh_btn.clicked.connect(self.refresh_state)
        self.state_view = QPlainTextEdit()
        self.state_view.setReadOnly(True)
        layout.addWidget(refresh_btn)
        layout.addWidget(self.state_view)
        return page

    def _utilities_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        top = QHBoxLayout()
        self.utility_selector = QComboBox()
        self.utility_selector.addItems(
            [
                "Sample clusters from GRO/XTC",
                "Sample active-learning clusters from LAMMPS dump",
                "Convert XYZ to Gaussian GJF",
                "Convert Gaussian logs to dataset",
                "Prepare SevenNet fine-tune directory",
                "Prepare LAMMPS input",
            ]
        )
        self.utility_selector.currentIndexChanged.connect(self._change_utility_page)
        self.util_run_btn = QPushButton("Run selected utility")
        self.util_run_btn.clicked.connect(self.run_utility)
        top.addWidget(QLabel("Utility"))
        top.addWidget(self.utility_selector, stretch=1)
        top.addWidget(self.util_run_btn)

        self.utility_stack = QStackedWidget()
        self.utility_stack.addWidget(self._util_sample_clusters_page())
        self.utility_stack.addWidget(self._util_sample_active_page())
        self.utility_stack.addWidget(self._util_xyz_to_gjf_page())
        self.utility_stack.addWidget(self._util_logs_to_dataset_page())
        self.utility_stack.addWidget(self._util_prepare_finetune_page())
        self.utility_stack.addWidget(self._util_prepare_lammps_page())

        layout.addLayout(top)
        layout.addWidget(self.utility_stack)
        return page

    def _util_sample_clusters_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.us_gro = PathEdit()
        self.us_xtc = PathEdit()
        self.us_cluster_sizes = QLineEdit("2")
        self.us_n_samples = int_spin(1, 1_000_000, 20)
        self.us_output_dir = PathEdit(select_dir=True)
        self.us_output_dir.setText("sampled_clusters")
        self.us_frame_stride = int_spin(1, 1_000_000, 1)
        self.us_max_frames = OptionalIntEdit(None)
        self.us_random_frame_samples = OptionalIntEdit(8)
        self.us_neighbor_pool = int_spin(1, 1_000_000, 8)
        self.us_contact_cutoff_nm = float_spin(0.0, 1000.0, 0.50, 3)
        form.addRow("GRO", self.us_gro)
        form.addRow("XTC", self.us_xtc)
        form.addRow("Cluster sizes", self.us_cluster_sizes)
        form.addRow("Samples per size", self.us_n_samples)
        form.addRow("Output dir", self.us_output_dir)
        form.addRow("Frame stride", self.us_frame_stride)
        form.addRow("Max frames", self.us_max_frames)
        form.addRow("Random frame samples", self.us_random_frame_samples)
        form.addRow("Neighbor pool", self.us_neighbor_pool)
        form.addRow("Contact cutoff nm", self.us_contact_cutoff_nm)
        return page

    def _util_sample_active_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.ua_dump = PathEdit()
        self.ua_reference_gro = PathEdit()
        self.ua_cluster_sizes = QLineEdit("2")
        self.ua_n_samples = int_spin(1, 1_000_000, 20)
        self.ua_output_dir = PathEdit(select_dir=True)
        self.ua_output_dir.setText("active_learning_samples")
        self.ua_frame_stride = int_spin(1, 1_000_000, 1)
        self.ua_max_frames = OptionalIntEdit(None)
        self.ua_random_frame_samples = OptionalIntEdit(None)
        self.ua_candidate_seeds = OptionalIntEdit(None)
        self.ua_quiet = QCheckBox("Hide progress messages")
        self.ua_neighbor_pool = int_spin(1, 1_000_000, 8)
        self.ua_contact_cutoff = float_spin(0.0, 1000.0, 5.0, 3)
        self.ua_close_contact = float_spin(0.0, 1000.0, 1.2, 3)
        self.ua_hard_reject = float_spin(0.0, 1000.0, 0.55, 3)
        self.ua_random_seed = int_spin(0, 2_147_483_647, 17)
        form.addRow("LAMMPS dump", self.ua_dump)
        form.addRow("Reference GRO", self.ua_reference_gro)
        form.addRow("Cluster sizes", self.ua_cluster_sizes)
        form.addRow("Samples per size", self.ua_n_samples)
        form.addRow("Output dir", self.ua_output_dir)
        form.addRow("Frame stride", self.ua_frame_stride)
        form.addRow("Max frames", self.ua_max_frames)
        form.addRow("Random frame samples", self.ua_random_frame_samples)
        form.addRow("Candidate seeds/frame", self.ua_candidate_seeds)
        form.addRow("Quiet", self.ua_quiet)
        form.addRow("Neighbor pool", self.ua_neighbor_pool)
        form.addRow("Contact cutoff A", self.ua_contact_cutoff)
        form.addRow("Close contact alert A", self.ua_close_contact)
        form.addRow("Hard reject distance A", self.ua_hard_reject)
        form.addRow("Random seed", self.ua_random_seed)
        return page

    def _util_xyz_to_gjf_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.ug_input = PathEdit()
        self.ug_output_dir = PathEdit(select_dir=True)
        self.ug_output_dir.setText("gaussian_inputs")
        self.ug_route = QLineEdit("#p wb97xd/def2svp nosymm force scf=tight")
        self.ug_charge = int_spin(-100, 100, 0)
        self.ug_multiplicity = int_spin(1, 100, 1)
        self.ug_mem = QLineEdit("16GB")
        self.ug_nprocshared = int_spin(1, 4096, 16)
        self.ug_no_chk = QCheckBox("Do not write %chk")
        self.ug_no_recursive = QCheckBox("Do not search recursively")
        form.addRow("XYZ input", self.ug_input)
        form.addRow("Output dir", self.ug_output_dir)
        form.addRow("Route", self.ug_route)
        form.addRow("Charge", self.ug_charge)
        form.addRow("Multiplicity", self.ug_multiplicity)
        form.addRow("Memory", self.ug_mem)
        form.addRow("NProcShared", self.ug_nprocshared)
        form.addRow("No chk", self.ug_no_chk)
        form.addRow("No recursive", self.ug_no_recursive)
        return page

    def _util_logs_to_dataset_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.ul_input = PathEdit()
        self.ul_output_dir = PathEdit(select_dir=True)
        self.ul_output_dir.setText("training_dataset")
        self.ul_formats = QLineEdit("jsonl, extxyz")
        self.ul_no_recursive = QCheckBox("Do not search recursively")
        self.ul_require_normal = QCheckBox("Require normal termination")
        self.ul_sevennet_label = QLineEdit("gaussian")
        form.addRow("Gaussian log/out input", self.ul_input)
        form.addRow("Output dir", self.ul_output_dir)
        form.addRow("Formats", self.ul_formats)
        form.addRow("No recursive", self.ul_no_recursive)
        form.addRow("Require normal termination", self.ul_require_normal)
        form.addRow("SevenNet label", self.ul_sevennet_label)
        return page

    def _util_prepare_finetune_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.uf_source_kind = QComboBox()
        self.uf_source_kind.addItems(["Gaussian logs", "Existing extxyz dataset"])
        self.uf_source = PathEdit()
        self.uf_output_dir = PathEdit(select_dir=True)
        self.uf_output_dir.setText("sevennet_finetune")
        self.uf_pretrained = PathEdit()
        self.uf_pretrained.setText("7net-omni")
        self.uf_modal = QLineEdit("omol25_low")
        self.uf_epoch = int_spin(1, 1_000_000, 20)
        self.uf_batch_size = int_spin(1, 1_000_000, 1)
        self.uf_lr = float_spin(0.0, 1.0, 1.0e-5, 8)
        self.uf_force_loss_weight = float_spin(0.0, 100_000.0, 10.0, 3)
        self.uf_data_divide_ratio = float_spin(0.0, 1.0, 0.2, 3)
        self.uf_best_metric = QLineEdit("Force_RMSE")
        self.uf_huber_delta = float_spin(0.0, 1000.0, 0.1, 3)
        self.uf_train_shift_scale = QCheckBox("Train shift/scale")
        self.uf_train_shift_scale.setChecked(True)
        self.uf_train_denominator = QCheckBox("Train denominator")
        self.uf_require_normal = QCheckBox("Require normal termination")
        self.uf_sevennet_label = QLineEdit("gaussian_finetune")
        form.addRow("Source type", self.uf_source_kind)
        form.addRow("Source path", self.uf_source)
        form.addRow("Output dir", self.uf_output_dir)
        form.addRow("Pretrained", self.uf_pretrained)
        form.addRow("Modal", self.uf_modal)
        form.addRow("Epoch", self.uf_epoch)
        form.addRow("Batch size", self.uf_batch_size)
        form.addRow("Learning rate", self.uf_lr)
        form.addRow("Force loss weight", self.uf_force_loss_weight)
        form.addRow("Data divide ratio", self.uf_data_divide_ratio)
        form.addRow("Best metric", self.uf_best_metric)
        form.addRow("Huber delta", self.uf_huber_delta)
        form.addRow("Train shift/scale", self.uf_train_shift_scale)
        form.addRow("Train denominator", self.uf_train_denominator)
        form.addRow("Require normal termination", self.uf_require_normal)
        form.addRow("SevenNet label", self.uf_sevennet_label)
        return page

    def _util_prepare_lammps_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.up_gro = PathEdit()
        self.up_model = PathEdit()
        self.up_output_dir = PathEdit(select_dir=True)
        self.up_output_dir.setText("lammps_sevennet")
        self.up_elements = QLineEdit()
        self.up_pair_style = QComboBox()
        self.up_pair_style.addItems(["e3gnn", "e3gnn/parallel", "mliap"])
        self.up_parallel_model_count = OptionalIntEdit(None)
        self.up_temperature = float_spin(0.0, 100_000.0, 300.0, 3)
        self.up_timestep = float_spin(0.0, 1000.0, 0.001, 6)
        self.up_run_steps = int_spin(1, 2_147_483_647, 1000)
        self.up_thermo_interval = int_spin(1, 2_147_483_647, 100)
        self.up_dump_interval = int_spin(1, 2_147_483_647, 100)
        self.up_ensemble = QComboBox()
        self.up_ensemble.addItems(["nvt", "nve", "minimize"])
        self.up_seed = int_spin(0, 2_147_483_647, 12345)
        self.up_d3 = QCheckBox("Use SevenNet+D3")
        form.addRow("GRO", self.up_gro)
        form.addRow("Model", self.up_model)
        form.addRow("Output dir", self.up_output_dir)
        form.addRow("Elements", self.up_elements)
        form.addRow("Pair style", self.up_pair_style)
        form.addRow("Parallel model count", self.up_parallel_model_count)
        form.addRow("Temperature K", self.up_temperature)
        form.addRow("Timestep ps", self.up_timestep)
        form.addRow("Run steps", self.up_run_steps)
        form.addRow("Thermo interval", self.up_thermo_interval)
        form.addRow("Dump interval", self.up_dump_interval)
        form.addRow("Ensemble", self.up_ensemble)
        form.addRow("Seed", self.up_seed)
        form.addRow("D3", self.up_d3)
        return page

    def _build_run_bar(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self.once = QCheckBox("Run one iteration and exit")
        self.once.setChecked(True)
        self.dry_run = QCheckBox("Dry run")
        self.dry_run.setChecked(True)
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.run_loop)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_loop)
        layout.addWidget(self.once)
        layout.addWidget(self.dry_run)
        layout.addStretch(1)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.stop_btn)
        return layout

    def _build_log_box(self) -> QPlainTextEdit:
        log = QPlainTextEdit()
        log.setReadOnly(True)
        log.setPlaceholderText("Process output appears here.")
        log.setFont(QFont("Consolas", 10))
        self.log = log
        return log

    def open_config(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open TOML config",
            str(ROOT),
            "TOML files (*.toml);;All files (*)",
        )
        if filename:
            self.load_config(Path(filename))

    def load_config(self, path: Path) -> None:
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            self._apply_config(data)
        except Exception as exc:
            QMessageBox.critical(self, "Could not load config", str(exc))
            return
        self.config_path = path
        self.config_label.setText(f"Config: {path}")
        self.log.appendPlainText(f"Loaded config: {path}")
        self.refresh_state()

    def save_config(self) -> bool:
        if self.config_path is None:
            return self.save_config_as()
        return self._write_config(self.config_path)

    def save_config_as(self) -> bool:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save TOML config",
            str(ROOT / "active_learning_loop.gui.toml"),
            "TOML files (*.toml);;All files (*)",
        )
        if not filename:
            return False
        path = Path(filename)
        if path.suffix.lower() != ".toml":
            path = path.with_suffix(".toml")
        if self._write_config(path):
            self.config_path = path
            self.config_label.setText(f"Config: {path}")
            return True
        return False

    def _write_config(self, path: Path) -> bool:
        try:
            path.write_text(to_toml(self._collect_config()), encoding="utf-8", newline="\n")
        except Exception as exc:
            QMessageBox.critical(self, "Could not save config", str(exc))
            return False
        self.log.appendPlainText(f"Saved config: {path}")
        return True

    def run_loop(self) -> None:
        if not self.save_config():
            return
        assert self.config_path is not None

        script = ROOT / "scripts" / "run_active_learning_loop.py"
        args = [str(script), "--conf", str(self.config_path)]
        if self.once.isChecked():
            args.append("--once")
        if self.dry_run.isChecked():
            args.append("--dry-run")

        self._start_process(args)

    def run_utility(self) -> None:
        try:
            args = self._utility_args()
        except ValueError as exc:
            QMessageBox.warning(self, "Missing or invalid utility input", str(exc))
            return
        self._start_process(args)

    def _start_process(self, args: list[str]) -> None:
        if self.process is not None and self.process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, "Process is running", "Stop the current process before starting another one.")
            return
        python = Path(sys.executable)
        self.process = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(env)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.finished.connect(self._process_finished)

        self.log.appendPlainText("")
        self.log.appendPlainText(f"> {python} {' '.join(args)}")
        self.run_btn.setEnabled(False)
        if hasattr(self, "util_run_btn"):
            self.util_run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.process.start(str(python), args)

    def _utility_args(self) -> list[str]:
        index = self.utility_selector.currentIndex()
        if index == 0:
            args = [script_path("sample_clusters.py"), "--gro", required(self.us_gro.text(), "GRO")]
            append_option(args, "--xtc", self.us_xtc.text())
            args += ["--cluster-sizes", *parse_list(self.us_cluster_sizes.text())]
            args += ["--n-samples", str(self.us_n_samples.value()), "--output-dir", required(self.us_output_dir.text(), "Output dir")]
            args += ["--frame-stride", str(self.us_frame_stride.value())]
            append_optional_int(args, "--max-frames", self.us_max_frames)
            append_optional_int(args, "--random-frame-samples", self.us_random_frame_samples)
            args += ["--neighbor-pool", str(self.us_neighbor_pool.value())]
            args += ["--contact-cutoff-nm", str(self.us_contact_cutoff_nm.value())]
            return args
        if index == 1:
            args = [
                script_path("sample_active_learning.py"),
                "--dump",
                required(self.ua_dump.text(), "LAMMPS dump"),
                "--reference-gro",
                required(self.ua_reference_gro.text(), "Reference GRO"),
                "--cluster-sizes",
                *parse_list(self.ua_cluster_sizes.text()),
                "--n-samples",
                str(self.ua_n_samples.value()),
                "--output-dir",
                required(self.ua_output_dir.text(), "Output dir"),
                "--frame-stride",
                str(self.ua_frame_stride.value()),
            ]
            append_optional_int(args, "--max-frames", self.ua_max_frames)
            append_optional_int(args, "--random-frame-samples", self.ua_random_frame_samples)
            append_optional_int(args, "--candidate-seeds-per-frame", self.ua_candidate_seeds)
            append_flag(args, "--quiet", self.ua_quiet)
            args += ["--neighbor-pool", str(self.ua_neighbor_pool.value())]
            args += ["--contact-cutoff-angstrom", str(self.ua_contact_cutoff.value())]
            args += ["--close-contact-alert-angstrom", str(self.ua_close_contact.value())]
            args += ["--hard-reject-distance-angstrom", str(self.ua_hard_reject.value())]
            args += ["--random-seed", str(self.ua_random_seed.value())]
            return args
        if index == 2:
            args = [
                script_path("xyz_to_gjf.py"),
                "--input",
                required(self.ug_input.text(), "XYZ input"),
                "--output-dir",
                required(self.ug_output_dir.text(), "Output dir"),
                "--route",
                required(self.ug_route.text(), "Route"),
                "--charge",
                str(self.ug_charge.value()),
                "--multiplicity",
                str(self.ug_multiplicity.value()),
                "--mem",
                self.ug_mem.text().strip(),
                "--nprocshared",
                str(self.ug_nprocshared.value()),
            ]
            append_flag(args, "--no-chk", self.ug_no_chk)
            append_flag(args, "--no-recursive", self.ug_no_recursive)
            return args
        if index == 3:
            formats = parse_list(self.ul_formats.text())
            if not formats:
                raise ValueError("Formats must include jsonl and/or extxyz")
            args = [
                script_path("logs_to_dataset.py"),
                "--input",
                required(self.ul_input.text(), "Gaussian log/out input"),
                "--output-dir",
                required(self.ul_output_dir.text(), "Output dir"),
                "--formats",
                *formats,
                "--sevennet-label",
                required(self.ul_sevennet_label.text(), "SevenNet label"),
            ]
            append_flag(args, "--no-recursive", self.ul_no_recursive)
            append_flag(args, "--require-normal-termination", self.ul_require_normal)
            return args
        if index == 4:
            source_flag = "--logs" if self.uf_source_kind.currentIndex() == 0 else "--dataset-extxyz"
            args = [
                script_path("prepare_sevennet_finetune.py"),
                source_flag,
                required(self.uf_source.text(), "Source path"),
                "--output-dir",
                required(self.uf_output_dir.text(), "Output dir"),
                "--pretrained",
                required(self.uf_pretrained.text(), "Pretrained"),
                "--epoch",
                str(self.uf_epoch.value()),
                "--batch-size",
                str(self.uf_batch_size.value()),
                "--lr",
                str(self.uf_lr.value()),
                "--force-loss-weight",
                str(self.uf_force_loss_weight.value()),
                "--data-divide-ratio",
                str(self.uf_data_divide_ratio.value()),
                "--best-metric",
                required(self.uf_best_metric.text(), "Best metric"),
                "--huber-delta",
                str(self.uf_huber_delta.value()),
                "--sevennet-label",
                required(self.uf_sevennet_label.text(), "SevenNet label"),
            ]
            append_option(args, "--modal", self.uf_modal.text())
            if not self.uf_train_shift_scale.isChecked():
                args.append("--no-train-shift-scale")
            append_flag(args, "--train-denominator", self.uf_train_denominator)
            append_flag(args, "--require-normal-termination", self.uf_require_normal)
            return args
        args = [
            script_path("prepare_lammps.py"),
            "--gro",
            required(self.up_gro.text(), "GRO"),
            "--model",
            required(self.up_model.text(), "Model"),
            "--output-dir",
            required(self.up_output_dir.text(), "Output dir"),
            "--pair-style",
            self.up_pair_style.currentText(),
            "--temperature",
            str(self.up_temperature.value()),
            "--timestep",
            str(self.up_timestep.value()),
            "--run-steps",
            str(self.up_run_steps.value()),
            "--thermo-interval",
            str(self.up_thermo_interval.value()),
            "--dump-interval",
            str(self.up_dump_interval.value()),
            "--ensemble",
            self.up_ensemble.currentText(),
            "--seed",
            str(self.up_seed.value()),
        ]
        elements = parse_list(self.up_elements.text())
        if elements:
            args += ["--elements", *elements]
        append_optional_int(args, "--parallel-model-count", self.up_parallel_model_count)
        append_flag(args, "--d3", self.up_d3)
        return args

    def _change_utility_page(self, index: int) -> None:
        self.utility_stack.setCurrentIndex(index)

    def stop_loop(self) -> None:
        if self.process is None:
            return
        self.log.appendPlainText("Stopping process...")
        self.process.terminate()
        if not self.process.waitForFinished(3000):
            self.process.kill()

    def _read_stdout(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        self.log.appendPlainText(text.rstrip())

    def _read_stderr(self) -> None:
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardError()).decode(errors="replace")
        self.log.appendPlainText(text.rstrip())

    def _process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self.log.appendPlainText(f"Process finished: exit_code={exit_code}, status={exit_status.name}")
        self.run_btn.setEnabled(True)
        if hasattr(self, "util_run_btn"):
            self.util_run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.refresh_state()

    def refresh_state(self) -> None:
        work_dir_text = self.work_dir.text().strip()
        if not work_dir_text:
            self.state_view.setPlainText("No work directory configured.")
            return
        work_dir = Path(work_dir_text)
        if not work_dir.is_absolute():
            work_dir = ROOT / work_dir
        state_path = work_dir / "state.json"
        summary_path = work_dir / "summary.md"
        parts: list[str] = []
        if state_path.is_file():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                parts.append(json.dumps(state, indent=2, ensure_ascii=False))
            except Exception as exc:
                parts.append(f"Could not read {state_path}: {exc}")
        else:
            parts.append(f"No state file yet: {state_path}")
        if summary_path.is_file():
            parts.append("\n--- summary.md ---\n")
            parts.append(summary_path.read_text(encoding="utf-8", errors="replace"))
        self.state_view.setPlainText("\n".join(parts))

    def _apply_config(self, data: dict[str, Any]) -> None:
        dft = section(data, "dft")
        sampling = section(data, "sampling")
        md = section(data, "md")
        ft = section(data, "finetune")

        self.work_dir.setText(str(data.get("work_dir", "active_learning_loop_run")))
        self.poll_seconds.setValue(int(data.get("poll_seconds", 3600)))
        self.max_iterations.set_value(data.get("max_iterations", 20))
        self.convergence_patience.setValue(int(data.get("convergence_patience", 4)))
        self.execute_md.setChecked(bool(data.get("execute_md", False)))
        self.execute_finetune.setChecked(bool(data.get("execute_finetune", False)))

        self.dft_base_url.setText(str(dft.get("base_url", "http://localhost:8000")))
        self.dft_nodes.setText(", ".join(str(item) for item in dft.get("nodes", ["node01"])))
        self.dft_job_name.setText(str(dft.get("job_name", "Gau_AL")))
        self.dft_job_prefix.setText(str(dft.get("gaussian_job_prefix", "active_learning")))
        self.dft_route.setText(str(dft.get("route", "#p wb97xd/def2svp nosymm force scf=tight")))
        self.dft_charge.setValue(int(dft.get("charge", 0)))
        self.dft_multiplicity.setValue(int(dft.get("multiplicity", 1)))
        self.dft_mem.setText(str(dft.get("mem", "16GB")))
        self.dft_nprocshared.setValue(int(dft.get("nprocshared", 16)))
        self.dft_submit.setChecked(bool(dft.get("submit", True)))

        self.sampling_mode.setCurrentText(str(sampling.get("mode", "film")))
        self.solute_resnames.setText(join_list(sampling.get("solute_resnames", [])))
        self.solvent_resnames.setText(join_list(sampling.get("solvent_resnames", [])))
        self.solution_solute_weights.setText(join_weights(sampling.get("solution_solute_weights", {"0": 0.25, "1": 0.75})))
        self.initial_total.setValue(int(sampling.get("initial_total", 24)))
        self.min_total.setValue(int(sampling.get("min_total", 4)))
        self.max_total.setValue(int(sampling.get("max_total", 96)))
        self.target_loop_hours.setValue(float(sampling.get("target_loop_hours", 12.0)))
        self.cluster_sizes.setText(join_list(sampling.get("cluster_sizes", [2, 3, 4])))
        self.initial_weights.setText(join_weights(sampling.get("initial_weights", {"2": 0.85, "3": 0.15, "4": 0.0})))
        self.mature_weights.setText(join_weights(sampling.get("mature_weights", {"2": 0.25, "3": 0.45, "4": 0.30})))
        self.weight_ramp_steps.setValue(int(sampling.get("weight_ramp_steps", 6)))
        self.frame_stride.setValue(int(sampling.get("frame_stride", 1)))
        self.max_frames.set_value(sampling.get("max_frames", 100))
        self.random_frame_samples.set_value(sampling.get("random_frame_samples", 8))
        self.candidate_seeds_per_frame.set_value(sampling.get("candidate_seeds_per_frame"))
        self.neighbor_pool.setValue(int(sampling.get("neighbor_pool", 8)))
        self.contact_cutoff.setValue(float(sampling.get("contact_cutoff_angstrom", 5.0)))
        self.close_contact_alert.setValue(float(sampling.get("close_contact_alert_angstrom", 1.2)))
        self.hard_reject_distance.setValue(float(sampling.get("hard_reject_distance_angstrom", 0.55)))
        self.random_seed.setValue(int(sampling.get("random_seed", 17)))

        self.md_export_whole_xtc.setChecked(bool(md.get("export_whole_xtc", False)))
        self.md_analysis_topology.setText(str(md.get("analysis_topology_path", "")))
        self.md_start_mode.setCurrentText(str(md.get("start_mode", "previous")))
        self.md_gro_path.setText(str(md.get("gro_path", "YCOL160.gro")))
        self.md_xtc_path.setText("" if md.get("xtc_path") is None else str(md.get("xtc_path")))
        self.md_model_path.setText(str(md.get("model_path", "7net-omni")))
        self.md_modal.setText(str(md.get("modal", "omol25_low")))
        self.md_pair_style.setCurrentText(str(md.get("pair_style", "e3gnn/parallel")))
        self.md_parallel_model_count.set_value(md.get("parallel_model_count", 4))
        self.md_enable_flash.setChecked(bool(md.get("enable_flash", True)))
        self.md_elements.setText(join_list(md.get("elements", [])))
        self.md_temperature.setValue(float(md.get("temperature_k", 300.0)))
        self.md_timestep.setValue(float(md.get("timestep_ps", 0.001)))
        self.md_run_steps.setValue(int(md.get("run_steps", 1000)))
        self.md_thermo_interval.setValue(int(md.get("thermo_interval", 100)))
        self.md_dump_interval.setValue(int(md.get("dump_interval", 100)))
        self.md_ensemble.setCurrentText(str(md.get("ensemble", "nvt")))
        self.md_seed.setValue(int(md.get("seed", 12345)))
        self.md_d3.setChecked(bool(md.get("d3", False)))
        self.md_run_command.setPlainText(str(md.get("run_command", "")))
        self.md_expected_dump.setText(str(md.get("expected_dump", "dump.sevennet.lammpstrj")))

        self.ft_pretrained.setText(str(ft.get("pretrained", "7net-omni")))
        self.ft_modal.setText(str(ft.get("modal", "omol25_low")))
        self.ft_external_log_dirs.setPlainText("\n".join(str(item) for item in ft.get("external_log_dirs", [])))
        self.ft_epoch.setValue(int(ft.get("epoch", 20)))
        self.ft_batch_size.setValue(int(ft.get("batch_size", 1)))
        self.ft_learning_rate.setValue(float(ft.get("learning_rate", 1.0e-5)))
        self.ft_force_loss_weight.setValue(float(ft.get("force_loss_weight", 10.0)))
        self.ft_data_divide_ratio.setValue(float(ft.get("data_divide_ratio", 0.2)))
        self.ft_best_metric.setText(str(ft.get("best_metric", "Force_RMSE")))
        self.ft_huber_delta.setValue(float(ft.get("huber_delta", 0.1)))
        self.ft_train_shift_scale.setChecked(bool(ft.get("train_shift_scale", True)))
        self.ft_train_denominator.setChecked(bool(ft.get("train_denominator", False)))
        self.ft_require_normal_termination.setChecked(bool(ft.get("require_normal_termination", False)))
        self.ft_run_command.setText("" if ft.get("run_command") is None else str(ft.get("run_command")))

    def _collect_config(self) -> dict[str, Any]:
        md_elements = parse_list(self.md_elements.text())
        return {
            "work_dir": self.work_dir.text().strip(),
            "poll_seconds": self.poll_seconds.value(),
            "max_iterations": self.max_iterations.value_or_none(),
            "convergence_patience": self.convergence_patience.value(),
            "execute_md": self.execute_md.isChecked(),
            "execute_finetune": self.execute_finetune.isChecked(),
            "dft": {
                "base_url": self.dft_base_url.text().strip(),
                "nodes": parse_list(self.dft_nodes.text()),
                "job_name": self.dft_job_name.text().strip(),
                "gaussian_job_prefix": self.dft_job_prefix.text().strip(),
                "route": self.dft_route.text().strip(),
                "charge": self.dft_charge.value(),
                "multiplicity": self.dft_multiplicity.value(),
                "mem": self.dft_mem.text().strip(),
                "nprocshared": self.dft_nprocshared.value(),
                "submit": self.dft_submit.isChecked(),
            },
            "sampling": {
                "mode": self.sampling_mode.currentText(),
                "solute_resnames": parse_list(self.solute_resnames.text()),
                "solvent_resnames": parse_list(self.solvent_resnames.text()),
                "solution_solute_weights": parse_weights(self.solution_solute_weights.text()),
                "initial_total": self.initial_total.value(),
                "min_total": self.min_total.value(),
                "max_total": self.max_total.value(),
                "target_loop_hours": self.target_loop_hours.value(),
                "cluster_sizes": [int(item) for item in parse_list(self.cluster_sizes.text())],
                "initial_weights": parse_weights(self.initial_weights.text()),
                "mature_weights": parse_weights(self.mature_weights.text()),
                "weight_ramp_steps": self.weight_ramp_steps.value(),
                "frame_stride": self.frame_stride.value(),
                "max_frames": self.max_frames.value_or_none(),
                "random_frame_samples": self.random_frame_samples.value_or_none(),
                "candidate_seeds_per_frame": self.candidate_seeds_per_frame.value_or_none(),
                "neighbor_pool": self.neighbor_pool.value(),
                "contact_cutoff_angstrom": self.contact_cutoff.value(),
                "close_contact_alert_angstrom": self.close_contact_alert.value(),
                "hard_reject_distance_angstrom": self.hard_reject_distance.value(),
                "random_seed": self.random_seed.value(),
            },
            "md": {
                "start_mode": self.md_start_mode.currentText(),
                "export_whole_xtc": self.md_export_whole_xtc.isChecked(),
                "analysis_topology_path": self.md_analysis_topology.text(),
                "gro_path": self.md_gro_path.text().strip(),
                "xtc_path": none_if_empty(self.md_xtc_path.text()),
                "model_path": self.md_model_path.text().strip(),
                "modal": self.md_modal.text().strip(),
                "pair_style": self.md_pair_style.currentText(),
                "parallel_model_count": self.md_parallel_model_count.value_or_none(),
                "enable_flash": self.md_enable_flash.isChecked(),
                "elements": None if not md_elements else md_elements,
                "temperature_k": self.md_temperature.value(),
                "timestep_ps": self.md_timestep.value(),
                "run_steps": self.md_run_steps.value(),
                "thermo_interval": self.md_thermo_interval.value(),
                "dump_interval": self.md_dump_interval.value(),
                "ensemble": self.md_ensemble.currentText(),
                "seed": self.md_seed.value(),
                "d3": self.md_d3.isChecked(),
                "run_command": self.md_run_command.toPlainText().strip(),
                "expected_dump": self.md_expected_dump.text().strip(),
            },
            "finetune": {
                "pretrained": self.ft_pretrained.text().strip(),
                "modal": self.ft_modal.text().strip(),
                "external_log_dirs": [
                    line.strip()
                    for line in self.ft_external_log_dirs.toPlainText().splitlines()
                    if line.strip()
                ],
                "epoch": self.ft_epoch.value(),
                "batch_size": self.ft_batch_size.value(),
                "learning_rate": self.ft_learning_rate.value(),
                "force_loss_weight": self.ft_force_loss_weight.value(),
                "data_divide_ratio": self.ft_data_divide_ratio.value(),
                "best_metric": self.ft_best_metric.text().strip(),
                "huber_delta": self.ft_huber_delta.value(),
                "train_shift_scale": self.ft_train_shift_scale.isChecked(),
                "train_denominator": self.ft_train_denominator.isChecked(),
                "require_normal_termination": self.ft_require_normal_termination.isChecked(),
                "run_command": none_if_empty(self.ft_run_command.text()),
            },
        }


class PathEdit(QWidget):
    def __init__(self, *, select_dir: bool = False) -> None:
        super().__init__()
        self.select_dir = select_dir
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        browse = QPushButton("Browse")
        browse.clicked.connect(self.browse)
        layout.addWidget(self.edit, stretch=1)
        layout.addWidget(browse)

    def text(self) -> str:
        return self.edit.text()

    def setText(self, text: str) -> None:
        self.edit.setText(text)

    def browse(self) -> None:
        start = self.edit.text().strip() or str(ROOT)
        if self.select_dir:
            filename = QFileDialog.getExistingDirectory(self, "Select directory", start)
        else:
            filename, _ = QFileDialog.getOpenFileName(self, "Select file or directory", start, "All files (*)")
            if not filename:
                filename = QFileDialog.getExistingDirectory(self, "Select directory", start)
        if filename:
            try:
                path = Path(filename)
                self.edit.setText(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))
            except ValueError:
                self.edit.setText(filename)


class OptionalIntEdit(QWidget):
    def __init__(self, default: int | None) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.enabled = QCheckBox("Set")
        self.spin = int_spin(0, 2_147_483_647, 0 if default is None else default)
        self.enabled.toggled.connect(self.spin.setEnabled)
        layout.addWidget(self.enabled)
        layout.addWidget(self.spin, stretch=1)
        self.set_value(default)

    def set_value(self, value: Any) -> None:
        is_set = value is not None
        self.enabled.setChecked(is_set)
        self.spin.setEnabled(is_set)
        if is_set:
            self.spin.setValue(int(value))

    def value_or_none(self) -> int | None:
        return self.spin.value() if self.enabled.isChecked() else None


def int_spin(minimum: int, maximum: int, value: int) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(value)
    return spin


def float_spin(minimum: float, maximum: float, value: float, decimals: int) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setValue(value)
    spin.setSingleStep(10 ** -decimals)
    return spin


def section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


def parse_list(text: str) -> list[str]:
    return [item.strip() for item in text.replace("\n", ",").split(",") if item.strip()]


def join_list(items: Any) -> str:
    if items is None:
        return ""
    return ", ".join(str(item) for item in items)


def parse_weights(text: str) -> dict[int, float]:
    weights: dict[int, float] = {}
    for item in parse_list(text):
        if "=" not in item:
            raise ValueError(f"Weight entry must look like size=value: {item}")
        key, value = item.split("=", 1)
        weights[int(key.strip())] = float(value.strip())
    return weights


def join_weights(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    return ", ".join(f"{key}={value}" for key, value in data.items())


def none_if_empty(text: str) -> str | None:
    stripped = text.strip()
    return stripped or None


def to_toml(data: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Active-learning loop configuration generated by scripts/gui_active_learning_loop.py.",
        "",
    ]
    for key in ("work_dir", "poll_seconds", "max_iterations", "convergence_patience", "execute_md", "execute_finetune"):
        if data[key] is None:
            continue
        lines.append(f"{key} = {toml_value(data[key])}")
    for section_name in ("dft", "sampling", "md", "finetune"):
        lines.append("")
        lines.append(f"[{section_name}]")
        for key, value in data[section_name].items():
            if value is None:
                continue
            lines.append(f"{key} = {toml_value(value)}")
    return "\n".join(lines) + "\n"


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{toml_key(key)} = {toml_value(item)}" for key, item in value.items()) + " }"
    if isinstance(value, str) and "\n" in value:
        escaped = value.replace("'''", "\\'\\'\\'")
        return "'''" + escaped + "'''"
    return json.dumps(str(value), ensure_ascii=False)


def toml_key(key: Any) -> str:
    text = str(key)
    return text if text.replace("_", "").isalnum() and not text[0].isdigit() else json.dumps(text)


def script_path(name: str) -> str:
    return str(ROOT / "scripts" / name)


def required(value: str, label: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{label} is required")
    return stripped


def append_option(args: list[str], flag: str, value: str) -> None:
    stripped = value.strip()
    if stripped:
        args.extend([flag, stripped])


def append_optional_int(args: list[str], flag: str, widget: OptionalIntEdit) -> None:
    value = widget.value_or_none()
    if value is not None:
        args.extend([flag, str(value)])


def append_flag(args: list[str], flag: str, widget: QCheckBox) -> None:
    if widget.isChecked():
        args.append(flag)


def main() -> None:
    os.chdir(ROOT)
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    window = ActiveLearningGui()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
