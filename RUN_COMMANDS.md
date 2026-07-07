# MLP-MD Loop 실행 명령어 모음

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
