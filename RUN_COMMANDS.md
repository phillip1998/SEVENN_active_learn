# MLP-MD Loop 실행 명령어 모음

## 분석용 whole.xtc / whole.gro 출력

MD 종료 후 경계에서 끊긴 분자를 결합 정보로 복원하고, 분자의 기하 중심을 box 안으로
이동한 분석용 XTC를 추가 생성할 수 있습니다. 기존 force 포함 dump와 다음 MD용
`final.data`는 그대로 사용합니다. XTC 지원 LAMMPS 빌드는 필요하지 않으며 MDAnalysis로 변환합니다.

```toml
[md]
export_whole_xtc = true
analysis_topology_path = "/data/mixture.tpr"
```

TPR은 `gro_path`와 원자 수/순서/이름 및 residue 구성이 같고 분자 내부 결합이 있어야 합니다.
virtual site가 추가된 TPR 등 원자 수가 다른 파일은 사용할 수 없습니다.
GRO만으로 결합을 자동 추정하지 않습니다. 분자 한 개가 residue 한 개라는 현재 프로젝트 전제를
검사하며, 결합이 불완전하거나 여러 residue에 걸친 분자는 오류로 처리합니다.
TPR이 아직 준비되지 않았으면 기본값인 `export_whole_xtc = false`로 두세요.

각 iteration의 `md/`에 다음 결과가 생성됩니다.

- `whole.xtc`: 모든 dump frame을 분자 단위로 PBC 처리한 trajectory
- `whole.gro`: 같은 원자 순서의 첫 번째 보정 frame (reference 구조)
- `whole_export.json`: frame/atom 수, 시간 범위 및 처리 방식

기존 dump 파일만 따로 변환하려면:

```bash
python scripts/export_whole_trajectory.py \
  --dump active_learning_run01/iter_0000/md/dump.sevennet.lammpstrj \
  --reference-gro /data/mixture.gro \
  --topology /data/mixture.tpr \
  --timestep-ps 0.001 \
  --output-dir active_learning_run01/iter_0000/md
```

`--timestep-ps`는 dump 간격이 아닌 LAMMPS 한 step의 시간입니다. 예를 들어 1 fs timestep,
100-step dump 간격이면 `0.001`을 지정하며 XTC frame 간격은 0.1 ps가 됩니다.
좌표는 LAMMPS `units metal` 기준으로 읽고 XTC에서는 nm로 저장합니다.
시간은 각 iteration의 local step에 기반하며 iteration별 파일의 시간을 자동 연결하지 않습니다.

분자 중심을 box 안에 배치하기 때문에 일부 원자가 box 밖에 있어도 정상입니다.
이 결과는 분자 내부가 끊기지 않는 시각화/구조 분석용이며, 분자가 box를 건널 때 중심 위치는
다시 wrapping됩니다. 확산/MSD 분석용 시간축 no-jump trajectory는 아닙니다.
고정 결합 topology를 사용하므로 반응에 의해 결합이 바뀌는 trajectory의 연결성을 자동 판별하지 않습니다.

자동 loop는 설정한 topology를 MD 전에 검사합니다. MD 후 변환 과정에서 오류가 나면
`whole_export_error.json`과 경고를 남기고 완료된 MD와 loop 진행은 유지합니다.
원인을 고친 뒤 위 독립 변환 명령으로 재시도할 수 있습니다.
`execute_md = false`로 MD를 수동 실행한 경우에도 독립 변환 명령을 사용하세요.

## Iteration 간 고정 분할과 MD 이어 실행

현재 loop는 `work_dir/split_manifest.json`에 구조별 train/validation 소속을 저장합니다.
기존 구조는 소속을 유지하고 새 구조만 추가 배정합니다. 파일명이나 복사 경로가 바뀌어도
원소와 원자 간 거리(소수점 6자리)로 구조를 식별하며, 같은 기하 구조의 중복은 제거합니다.
원자 순서, 평행이동, 회전에 영향받지 않는 거리 기반 ID를 사용하지만, 유사 구조 및
같은 trajectory에서 나온 상관된 구조 전체를 묶는 기능은 아닙니다.

- `[finetune].data_divide_ratio`는 새 manifest를 만들 때의 목표 비율입니다. 이후에는
  manifest에 기록된 비율과 seed를 유지해야 하며, 기존 소속을 바꿔 비율을 맞추지 않습니다.
- 각 `finetune/split_summary.json`에 train/valid 개수와 중복 제거 개수를 기록합니다.
- validation에는 새 데이터가 추가될 수 있으므로 평가군 전체가 완전히 고정되는 것은 아닙니다.
  고정되는 것은 기존 구조의 소속입니다.
- 기존 실행을 이어갈 때 manifest가 없으면 과거 `train.extxyz`/`valid.extxyz`를 읽어
  소속을 복원합니다. 과거 train과 valid 양쪽에 등장한 구조가 있으면 leakage 오류로 중단합니다.
  새 work_dir와 오염되지 않은 checkpoint로 다시 시작해야 합니다.
- 외부에서 학습된 checkpoint의 학습 데이터 이력은 자동 확인할 수 없습니다.
  manifest와 과거 split 파일을 보존하세요. 같은 work_dir에 loop를 중복 실행하지 마세요.

독립 fine-tuning 준비 명령으로 여러 batch를 만들 때도 같은 manifest를 지정합니다.

```bash
python scripts/prepare_sevennet_finetune.py \
  --logs gaussian_logs_total_batch02 --output-dir sevennet_finetune_batch02 \
  --pretrained /models/checkpoint_best.pth \
  --split-manifest /data/al_run/split_manifest.json
```

MD는 `[md]`에서 시작 방식을 선택합니다. 기본값은 `previous`입니다.

```toml
[md]
start_mode = "previous"
```

- iteration 0: 원본 GRO에서 시작하고 초기 속도를 생성합니다.
- iteration 1 이후: 직전 `md/final.data`의 좌표, 셀, 속도, 원자 ID/type을 사용합니다.
  초기 속도를 다시 생성하지 않으며, 새로운 SevenNet 모델을 적용합니다.
  이번 iteration에 새 모델이 없으면 가장 최근 배포된 fine-tuning 모델을 사용합니다.
- `final.data`는 `run`/`minimize`가 끝난 뒤 `write_data final.data nocoeff`로 저장합니다.
  이후 `md.complete` 완료 표시를 씁니다. 마지막 dump 간격과 무관하게 마지막 상태를 저장합니다.
- `md_context.json`에 기준 GRO 및 원소 매핑을 기록합니다. 이전 상태가 없거나 완료 표시가 없거나,
  원자 수/ID/type, 속도, 셀 검증에 실패하면 원본 GRO로 돌아가지 않고 중단합니다.
- binary restart가 아니므로 thermostat 내부 상태는 초기화되고 timestep 번호도 매 iteration
  다시 시작합니다. 새 potential에서 이어가는 MD 구간이며 단일 trajectory의 완전한 restart는 아닙니다.
- `execute_md = false`로 준비만 했다면 생성된 `md/run_md.sh`를 성공적으로 실행한 뒤
  다음 iteration을 실행해야 합니다. dry-run은 실제 실행과 별도 work_dir에서 한 번씩 확인하세요.
- 예전 코드로 만든 MD에는 `final.data`와 완료 표시가 없으므로 바로 이어받을 수 없습니다.
  새 실행을 시작하거나 새 코드가 생성한 MD 입력으로 이전 구간의 최종 상태를 준비해야 합니다.

기존처럼 매번 원본 GRO에서 시작하려면 명시적으로 `start_mode = "initial"`을 사용합니다.

## Film / solution 샘플링 선택 (서버 CLI)

기본값은 `film`이며 기존 샘플링을 유지합니다. `solution`은 GRO의 residue 이름으로
용질/용매를 구분하고 조성별로 샘플을 선택합니다. 초기 GRO/XTC와 이후 MD dump에 모두 적용됩니다.

자동 loop 설정의 기존 `[sampling]`에 아래 항목을 추가하거나 수정합니다.
`A`, `B`는 실제 GRO의 대소문자까지 일치하는 residue 이름으로 바꾸세요.

```toml
[sampling]
mode = "solution"  # film 또는 solution
solute_resnames = ["A"]
solvent_resnames = ["B"]
solution_solute_weights = { "0" = 0.25, "1" = 0.75 }
```

가중치의 키는 **cluster 안의 용질 분자 수**이며, cluster 크기별로 따로 적용됩니다.
dimer 20개면 용매–용매 5개와 용질–용매 15개, trimer 20개면 용매 3개짜리 5개와
용질 1개+용매 2개짜리 15개를 요청합니다. 용질–용질도 포함하려면 `"2" = 0.1`처럼
가중치를 추가하세요. cluster 크기보다 큰 용질 개수는 제외하고 나머지 가중치를 정규화합니다.
정수 할당은 최대 나머지 방식이라 적은 샘플 수에서는 일부 조성의 할당량이 0일 수 있습니다.

```bash
python scripts/run_active_learning_loop.py --conf active_learning_loop.server.toml
```

초기 GRO/XTC에서 직접 추출:

```bash
python scripts/sample_clusters.py \
  --gro mixture.gro --xtc mixture.xtc \
  --sampling-mode solution \
  --solute-resnames A --solvent-resnames B \
  --solution-solute-weights '0:0.25,1:0.75' \
  --cluster-sizes 2 3 --n-samples 20 \
  --output-dir solution_initial
```

`--xtc`를 생략하면 GRO snapshot만 사용합니다. MD dump에서 직접 추출:

```bash
python scripts/sample_active_learning.py \
  --dump dump.sevennet.lammpstrj --reference-gro mixture.gro \
  --sampling-mode solution \
  --solute-resnames A --solvent-resnames B \
  --solution-solute-weights '0:0.25,1:0.75' \
  --cluster-sizes 2 3 --n-samples 20 \
  --output-dir solution_active
```

기존 방식은 `--sampling-mode film` 또는 옵션 생략으로 사용합니다.

solution의 선택 규칙:

- 혼합 cluster에서는 모든 용질을 seed로 검사합니다. MD의 `candidate_seeds_per_frame`은
  용매만 있는 cluster의 용매 seed에만 적용됩니다.
- seed와 heavy-atom 접촉 cutoff 이내에 있는 분자로 cluster를 구성합니다.
  `neighbor_pool`은 용질/용매 각각의 가장 가까운 접촉 이웃 수를 제한합니다.
  모든 구성 분자가 seed에 접촉하는 구조를 선택하며 긴 사슬 모양의 접촉 전체를 열거하지는 않습니다.
- 초기 샘플링은 pi-stacking 점수를 제외하고 contact, compactness와 다양성을 사용합니다.
  MD 후보 점수는 용질 개수가 같은 후보군 안에서 정규화합니다. committee uncertainty는 아닙니다.
- 각 크기별 폴더의 `sampling_report.json`에 조성별 요청/후보/선택/부족 수와
  sample별 residue 이름, 조성, 원자 수를 기록합니다.
- 후보가 부족하면 경고와 보고서를 남기며 다른 조성으로 자동 재배정하지 않습니다.
  모든 후보가 없으면 빈 메타데이터를 저장하고 loop의 DFT 제출을 건너뜁니다.
- 누락된 residue 이름, 존재하지 않는 이름, 용질/용매 중복 지정은 오류로 처리합니다.
- 원자 간 거리 계산 비용이 있으므로 큰 용매계는 적은 frame 수로 먼저 확인하세요.
  초기 샘플러는 모든 용매 seed를 검사합니다.

한 residue가 한 분자라는 전제, 원자 이름 기반 원소 추정, 직교 셀 PBC,
Gaussian의 공통 charge/multiplicity 설정은 기존과 같습니다. 이 옵션은 virtual site 처리,
triclinic 셀 또는 classical 용매와의 hybrid MD를 추가하지 않습니다.
다른 설정으로 재실행할 때는 이전 XYZ가 섞이지 않도록 새 output 폴더를 사용하세요.
dry-run은 실제 실행과 별도 `work_dir`에서 수행하세요.

이 문서는 지금까지 만든 파이프라인을 처음부터 따라 실행하기 위한 명령어 모음입니다.

기본 작업 폴더는 아래라고 가정합니다.

```text
C:\Users\phill\Documents\MLP MD loop
```

Windows에서는 `PowerShell`을 열고, Linux 서버에서는 `bash` 터미널을 사용하면 됩니다.

## GUI로 active learning 루프 실행

Windows에서 GUI로 설정을 열고 실행하려면 PowerShell에서 아래 순서대로 실행합니다.

```powershell
cd "C:\Users\phill\Documents\MLP MD loop"
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\gui_active_learning_loop.py
```

GUI는 기본 설정 예시인 `active_learning_loop.example.toml`을 자동으로 불러옵니다. 필요한 입력 파일, DFT API 주소, 샘플링 조건, MD 실행 명령, SevenNet fine-tune 명령을 탭에서 수정한 뒤 `Save as`로 설정 파일을 저장하고 `Run loop`를 누르면 됩니다.

GUI에서 저장한 설정 파일을 터미널에서 직접 실행하려면 아래처럼 실행합니다.

```powershell
python scripts\run_active_learning_loop.py --conf active_learning_loop.gui.toml
```

한 번만 시험 실행하려면 `--once`를 붙입니다.

```powershell
python scripts\run_active_learning_loop.py --conf active_learning_loop.gui.toml --once
```

## 0. Windows에서 작업 폴더로 이동

PowerShell을 열고 아래 명령어를 실행합니다.

```powershell
cd "C:\Users\phill\Documents\MLP MD loop"
```

현재 폴더에 파일이 있는지 확인합니다.

```powershell
dir
```

최소한 아래 파일들이 있으면 됩니다.

```text
YCOL160.gro
YCOL160.xtc
scripts
mlp_md_loop
requirements.txt
```

## 1. Python 가상환경 만들기

처음 한 번만 실행하면 됩니다.

```powershell
python -m venv .venv
```

가상환경을 켭니다.

```powershell
.\.venv\Scripts\Activate.ps1
```

만약 실행 정책 오류가 나오면, 이 명령어를 한 번 실행한 뒤 다시 켭니다.

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

필요한 Python 패키지를 설치합니다.

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

가상환경이 잘 켜졌는지 확인합니다.

```powershell
python -c "import MDAnalysis, ase; print('OK')"
```

참고: 가상환경을 켜기 어렵다면, 모든 명령어에서 `python` 대신 아래처럼 직접 실행해도 됩니다.

```powershell
.\.venv\Scripts\python.exe scripts\sample_clusters.py --help
```

## 2. GROMACS morphology에서 DFT 후보 XYZ 뽑기

입력 파일:

```text
YCOL160.gro
YCOL160.xtc
```

### 2.1 빠른 테스트용 dimer 5개

```powershell
python scripts\sample_clusters.py --gro YCOL160.gro --xtc YCOL160.xtc --cluster-sizes 2 --n-samples 5 --output-dir sampled_clusters_test --frame-stride 100 --max-frames 5 --neighbor-pool 8
```

결과 확인:

```powershell
dir sampled_clusters_test\2mol
```

### 2.2 실제 batch 예시: dimer 50개

```powershell
python scripts\sample_clusters.py --gro YCOL160.gro --xtc YCOL160.xtc --cluster-sizes 2 --n-samples 50 --output-dir sampled_clusters_ycol_xtc_batch01 --frame-stride 50 --max-frames 20 --neighbor-pool 8
```

### 2.3 실제 batch 예시: trimer 20개

```powershell
python scripts\sample_clusters.py --gro YCOL160.gro --xtc YCOL160.xtc --cluster-sizes 3 --n-samples 20 --output-dir sampled_clusters_ycol_xtc_batch01 --frame-stride 50 --max-frames 20 --neighbor-pool 8
```

결과 폴더:

```text
sampled_clusters_ycol_xtc_batch01\2mol
sampled_clusters_ycol_xtc_batch01\3mol
```

## 3. XYZ를 Gaussian GJF로 변환

아래 명령어는 `sampled_clusters_ycol_xtc_batch01` 안의 모든 `.xyz`를 `.gjf`로 변환합니다.

```powershell
python scripts\xyz_to_gjf.py --input sampled_clusters_ycol_xtc_batch01 --output-dir gaussian_inputs_ycol_xtc_batch01 --mem 16GB --nprocshared 16
```

결과 확인:

```powershell
dir gaussian_inputs_ycol_xtc_batch01
type gaussian_inputs_ycol_xtc_batch01\gaussian_inputs.tsv
```

Gaussian input route 기본값:

```text
#p wb97xd/def2svp nosymm force scf=tight
```

생성된 `.gjf`를 Gaussian 서버에서 계산합니다. 계산이 끝나면 `.log` 파일들을 한 폴더에 모읍니다.

예시 폴더 이름:

```text
gaussian_logs_batch01
```

## 4. Gaussian LOG를 SevenNet 학습 데이터로 변환

Gaussian 계산 결과 `.log` 또는 `.out` 파일들이 `gaussian_logs_batch01` 폴더에 있다고 가정합니다.

```powershell
python scripts\logs_to_dataset.py --input gaussian_logs_batch01 --output-dir training_dataset_batch01 --require-normal-termination
```

결과 파일:

```text
training_dataset_batch01\dataset.extxyz
training_dataset_batch01\dataset.jsonl
training_dataset_batch01\dataset_manifest.tsv
training_dataset_batch01\structure_list
```

계산이 정상 종료되지 않은 log도 일단 파싱하고 싶으면 `--require-normal-termination`을 빼고 실행합니다.

```powershell
python scripts\logs_to_dataset.py --input gaussian_logs_batch01 --output-dir training_dataset_batch01
```

## 5. SevenNet pretrained 모델 fine-tuning 준비

Gaussian log에서 바로 fine-tuning 폴더를 만들 수 있습니다.

```powershell
python scripts\prepare_sevennet_finetune.py --logs gaussian_logs_batch01 --output-dir sevennet_finetune_batch01 --epoch 20 --batch-size 1 --lr 1e-5 --force-loss-weight 10 --data-divide-ratio 0.2 --best-metric Force_RMSE --huber-delta 0.1
```

SevenNet-Omni를 pretrained 모델로 쓰려면 아래처럼 실행합니다. `7net-omni`는 multi-modal 모델이라 `--modal`이 필요합니다. 기본 추천값은 `mpa`입니다.

```powershell
python scripts\prepare_sevennet_finetune.py --logs gaussian_logs_batch01 --output-dir sevennet_finetune_omni_batch01 --pretrained 7net-omni --modal omol25_low --epoch 20 --batch-size 1 --lr 1e-5 --force-loss-weight 10 --data-divide-ratio 0.2 --best-metric Force_RMSE --huber-delta 0.1
```

다른 task를 쓰고 싶으면 `--modal omat24`, `--modal matpes_pbe`, `--modal omol25_low`처럼 바꿉니다. 단, 해당 task가 checkpoint에 있어야 합니다.

이미 만든 `dataset.extxyz`를 사용하려면 아래처럼 실행합니다.

```powershell
python scripts\prepare_sevennet_finetune.py --dataset-extxyz training_dataset_batch01\dataset.extxyz --output-dir sevennet_finetune_batch01 --epoch 20 --batch-size 1 --lr 1e-5 --force-loss-weight 10 --data-divide-ratio 0.2 --best-metric Force_RMSE --huber-delta 0.1
```

생성되는 주요 파일:

```text
sevennet_finetune_batch01\fine_tuning_set.extxyz
sevennet_finetune_batch01\train.extxyz
sevennet_finetune_batch01\valid.extxyz
sevennet_finetune_batch01\input_finetune.yaml
sevennet_finetune_batch01\make_input_finetune.py
sevennet_finetune_batch01\run_finetune.sh
sevennet_finetune_batch01\README_finetune.md
```

주의: `input_finetune.yaml`에서 `checkpoint` 한 줄만 직접 바꾸는 방식은 권장하지 않습니다. `7net-0`와 `7net-omni`는 modality와 architecture가 다를 수 있기 때문입니다. 모델을 바꾸려면 `prepare_sevennet_finetune.py`를 다시 실행하세요.

## 6. SevenNet fine-tuning 실행

이 단계는 보통 Linux GPU 서버에서 실행합니다.

먼저 `sevennet_finetune_batch01` 폴더를 서버로 옮깁니다.

서버에서 폴더로 이동합니다.

```bash
cd sevennet_finetune_batch01
```

fine-tuning을 실행합니다.

```bash
bash run_finetune.sh
```

이 스크립트는 아래 작업을 자동으로 합니다.

```bash
python make_input_finetune.py
sevenn train input_finetune.yaml -s
sevenn get_model checkpoint_best.pth -o deployed_serial
sevenn get_model checkpoint_best.pth --get_parallel -o deployed_parallel
```

`checkpoint_best.pth`가 없으면 validation set 또는 `best_metric` 설정이 잘못된 것입니다. 이 경우 `run_finetune.sh`는 `checkpoint_last.pth`로 넘어가지 않고 중단합니다.

`make_input_finetune.py`는 서버에서 실제 pretrained checkpoint를 읽어서 `input_finetune.yaml`을 다시 생성합니다. 그래서 `7net-omni` 같은 multi-modal 모델도 `use_modality`와 `data_modality`가 자동으로 맞춰집니다.

## 7. SevenNet 모델로 LAMMPS input 만들기

fine-tuning이 끝나면 서버 폴더 안에 보통 아래 결과가 생깁니다.

```text
deployed_serial
deployed_parallel
```

LAMMPS input은 Windows에서 만들어도 되고 서버에서 만들어도 됩니다. 여기서는 Windows PowerShell 기준으로 적습니다.

### 7.1 Serial 모델용 LAMMPS input

```powershell
python scripts\prepare_lammps.py --gro YCOL160.gro --model sevennet_finetune_batch01\deployed_serial.pt --output-dir lammps_input_serial --pair-style e3gnn --elements C F H N O S --temperature 300 --run-steps 1000 --ensemble nvt
```

결과:

```text
lammps_input_serial\system.data
lammps_input_serial\in.sevennet.lmp
```

서버에서 실행:

```bash
cd lammps_input_serial
lmp -in in.sevennet.lmp
```

### 7.2 Parallel 모델용 LAMMPS input

`deployed_parallel` 폴더 안에 `deployed_parallel_0.pt`, `deployed_parallel_1.pt` 같은 파일들이 있어야 합니다.

예를 들어 shard가 3개이면 `--parallel-model-count 3`을 사용합니다.

```powershell
python scripts\prepare_lammps.py --gro YCOL160.gro --model sevennet_finetune_batch01\deployed_parallel --output-dir lammps_input_parallel --pair-style e3gnn/parallel --parallel-model-count 3 --elements C F H N O S --temperature 300 --run-steps 1000 --ensemble nvt
```

서버에서 2 GPU로 실행하는 예시:

```bash
cd lammps_input_parallel
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0,1 mpirun --allow-run-as-root -np 2 lmp_mpi -in in.sevennet.lmp
```

정상적으로 2 GPU 병렬이면 LAMMPS 로그에 아래처럼 나와야 합니다.

```text
2 by 1 by 1 MPI processor grid
```

만약 아래처럼 나오면 병렬이 아닙니다.

```text
1 by 1 by 1 MPI processor grid
```

이 경우 `lmp`가 MPI-enabled LAMMPS인지 확인합니다.

```bash
ldd $(which lmp) | grep -i mpi
which lmp_mpi
```

## 8. LAMMPS MD 결과에서 active learning 후보 뽑기

LAMMPS 실행 후 아래 파일이 생겼다고 가정합니다.

```text
dump.sevennet.lammpstrj
```

dimer 후보 20개를 뽑습니다.

```powershell
python scripts\sample_active_learning.py --dump dump.sevennet.lammpstrj --reference-gro YCOL160.gro --cluster-sizes 2 --n-samples 20 --output-dir active_learning_samples_batch01
```

dimer 20개와 trimer 10개를 같이 뽑습니다.

```powershell
python scripts\sample_active_learning.py --dump dump.sevennet.lammpstrj --reference-gro YCOL160.gro --cluster-sizes 2 3 --n-samples 20 --output-dir active_learning_samples_batch01
```

너무 심하게 깨진 구조를 제외하고 싶으면 `--hard-reject-distance-angstrom` 값을 키웁니다.

```powershell
python scripts\sample_active_learning.py --dump dump.sevennet.lammpstrj --reference-gro YCOL160.gro --cluster-sizes 2 3 --n-samples 20 --output-dir active_learning_samples_batch01 --hard-reject-distance-angstrom 1.0
```

결과:

```text
active_learning_samples_batch01\2mol
active_learning_samples_batch01\3mol
```

각 폴더에는 아래 파일이 생깁니다.

```text
*.xyz
active_learning_samples.csv
active_learning_samples.json
```

## 9. Active learning 후보를 Gaussian GJF로 변환

```powershell
python scripts\xyz_to_gjf.py --input active_learning_samples_batch01 --output-dir gaussian_inputs_active_batch01 --mem 16GB --nprocshared 16
```

이 `.gjf`들을 Gaussian으로 계산합니다. 계산이 끝난 `.log` 파일들을 새 폴더에 모읍니다.

예시:

```text
gaussian_logs_active_batch01
```

그 다음 다시 학습 데이터로 변환합니다.

```powershell
python scripts\logs_to_dataset.py --input gaussian_logs_active_batch01 --output-dir training_dataset_active_batch01 --require-normal-termination
```

## 10. 다음 fine-tuning batch 만들기

새 active learning 데이터까지 포함해서 fine-tuning을 다시 합니다.

가장 단순한 방법은 기존 Gaussian log 폴더와 새 Gaussian log 폴더를 하나의 폴더에 모으는 것입니다.

예시:

```text
gaussian_logs_total_batch02
```

그 다음:

```powershell
python scripts\prepare_sevennet_finetune.py --logs gaussian_logs_total_batch02 --output-dir sevennet_finetune_batch02 --pretrained 7net-omni --modal omol25_low --epoch 20 --batch-size 1 --lr 1e-5 --force-loss-weight 10 --data-divide-ratio 0.2 --best-metric Force_RMSE --huber-delta 0.1
```

서버에서:

```bash
cd sevennet_finetune_batch02
bash run_finetune.sh
```

이후 다시 LAMMPS MD를 돌리고, active learning 후보를 뽑고, Gaussian 계산을 추가합니다.

## 11. 전체 반복 흐름 요약

```text
YCOL160.gro + YCOL160.xtc
-> sample_clusters.py
-> xyz_to_gjf.py
-> Gaussian 계산
-> logs_to_dataset.py
-> prepare_sevennet_finetune.py
-> sevenn train
-> prepare_lammps.py
-> LAMMPS MD
-> sample_active_learning.py
-> xyz_to_gjf.py
-> Gaussian 계산 추가
-> 다시 fine-tuning
```

## 12. 자주 생기는 문제

### PowerShell에서 가상환경 실행이 안 됨

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### `ModuleNotFoundError: No module named MDAnalysis`

가상환경이 꺼져 있거나 패키지가 설치되지 않은 상태입니다.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### SevenNet 학습에서 stress 관련 오류가 남

Gaussian cluster 데이터에는 stress가 없으므로 `input_finetune.yaml`에 아래가 있어야 합니다.

```yaml
is_train_stress: false
```

### LAMMPS parallel인데 GPU 하나만 씀

LAMMPS 로그에 아래가 나오면 병렬 실행이 아닙니다.

```text
1 by 1 by 1 MPI processor grid
```

2 GPU라면 아래처럼 보여야 합니다.

```text
2 by 1 by 1 MPI processor grid
```

실행 예시:

```bash
CUDA_VISIBLE_DEVICES=0,1 mpirun --allow-run-as-root -np 2 lmp_mpi -in in.sevennet.lmp
```

### CUDA out of memory

먼저 GPU에 남은 프로세스가 있는지 확인합니다.

```bash
nvidia-smi
```

메모리 파편화 완화 옵션을 켭니다.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

그래도 부족하면 작은 시스템, 작은 batch, multi-GPU MPI LAMMPS를 사용합니다.
