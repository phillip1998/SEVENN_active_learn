# MLP MD Loop

MLP MD Loop는 분자동역학(MD), 클러스터 샘플링, Gaussian DFT 계산, SevenNet 기반 MLP 파인튜닝을 연결하는 active learning 워크플로우를 위한 작업 공간입니다.

주요 구성:

- `mlp_md_loop/`: active learning, 데이터셋 처리, LAMMPS/Gaussian 연동 코드
- `scripts/`: 샘플링, 데이터 변환, LAMMPS/SevenNet 준비 실행 스크립트
- `DFT_API/`: Gaussian 계산 작업을 관리하기 위한 API 코드
- `example_data/`: 테스트와 실행 예시에 사용할 샘플 데이터
- `active_learning_loop.example.toml`: active learning 루프 설정 예시

## SevenNet 출처

이 저장소의 `SevenNet-main/` 및 SevenNet 기반 학습/추론 흐름은 SNU MDIL의 SevenNet 프로젝트를 기반으로 사용합니다.

Original source: [mdil-snu/sevennet](https://github.com/mdil-snu/sevennet)
