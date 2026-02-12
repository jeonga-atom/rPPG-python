# Start Here

이 문서는 프로젝트 시작과 GitHub 연동 점검용입니다.

## 1) 현재 상태 확인
```bash
git status
git remote -v
```

## 2) 첫 커밋 만들기
```bash
git add .
git commit -m "chore: initialize project structure"
```

## 3) 원격으로 push
```bash
git push -u origin main
```

## 4) VS Code에서 GitHub 연동 확인
- 좌측 `Source Control`에서 변경 파일이 보이는지 확인
- `Commit` 버튼으로 커밋 가능한지 확인
- 우측 하단 계정 아이콘에서 GitHub 로그인 상태 확인
- Command Palette(`Ctrl+Shift+P`) -> `Git: Show Git Output` 실행

## 5) 문제가 있을 때
### 인증 문제
```bash
git config --global user.name "YOUR_NAME"
git config --global user.email "YOUR_EMAIL"
```

### 원격 저장소 재설정
```bash
git remote set-url origin https://github.com/jeonga-atom/rPPG-python.git
git remote -v
```

## 6) 다음 작업 권장
- `src/capture/realsense_capture.py`: RGB 프레임 수집
- `src/preprocess/roi_extractor.py`: 얼굴 ROI 추출
- `src/inference/hr_estimator.py`: band-pass + FFT BPM 추정
- `src/ui/live_plot.py`: 실시간 파형 시각화
