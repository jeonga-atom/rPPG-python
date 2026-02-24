# rPPG Python Project

RealSense RGB 기반 비접촉식 심박 추정(rPPG) 연구 프로젝트.

## Quick Start
1. `00_start_here/README.md` 먼저 읽기
2. `src/`에 파이프라인 코드 추가
3. `configs/`에 실험 설정 관리
4. `data/`는 로컬 데이터 저장(원본/전처리 분리)

## Folder Structure
- `00_start_here/`: 시작 가이드
- `data/`: UBFC-rPPG 데이터셋 보관 총 49개의 데이터셋이 있음
- `dataset/`: rPPG-Toolbox의 dataset 파일
- `final_model_release/`: rPPG-Toolbox의 학습된 모델 가중치 파일(.pth)
- `log/`: realtime_rppg_realsense 코드 실행 후 bpm로그가 뜸
- `neural_methods/`: rPPG-Toolbox의 학습된 모델 파일
- `src/`: -
- `requirements.txt/`: 코드를 실행하기 위해 필요한 페키지 목록
- `realtime_rppg_realsense/`: 실시간 BPM탐지를 위한 메인 코드
