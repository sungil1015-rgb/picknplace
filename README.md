# CMES Bin Picking — 라벨링/시각화 GUI 툴 (donguk-ui-tool)

동욱이 만든 GUI 라벨링 툴 + 3D 뷰어 + 파지점(피킹) 로직 풀세트입니다.
이 브랜치만 받으면 툴의 **모든 기능과 단축키가 그대로** 동작합니다.
(데이터·모델 가중치는 포함되지 않으니 아래 안내대로 직접 배치하세요.)

---

## 1. 구성

```
label_tool.bat              # 실행 런처 (Windows)
config/default.yaml         # 클래스 정의 등 설정
scripts/
  label_tool.py            # 라벨링/뷰어 GUI 본체 (tkinter)
  view_3d.py               # 3D 뷰어 (Open3D)
  label_logic/             # GUI가 쓰는 로직 모듈
    picking.py             #   └ 파지점(피킹) 알고리즘
    quality.py             #   └ 폴리곤 품질/통계
    heightmap.py           #   └ height/normal map
    prelabel.py            #   └ watershed 프리라벨
    suction_score.py       #   └ 흡착 점수
    depth_recovery.py      #   └ 3D 깊이 복원
    box_roi.py / import_zip.py / data_manager.py / auto_classify.py / sam2_predictor.py
```

## 2. 설치

Python 3.11 권장. 필수 패키지:

```bash
pip install numpy opencv-python pillow pyyaml matplotlib scikit-image open3d scipy
```

선택(아래 기능 쓸 때만):
- YOLO 자동 검출 / SAM2 보조 라벨링 → `pip install ultralytics torch`

> `tkinter`는 표준 라이브러리(파이썬 기본 포함)입니다.

## 3. 실행

```bash
# Windows
label_tool.bat

# 또는 직접
python scripts/label_tool.py
```

> ⚠️ `label_tool.bat`은 파이썬 경로가 `C:\ProgramData\Anaconda3\python.exe`로
> 하드코딩돼 있습니다. **본인 PC의 파이썬 경로에 맞게 그 한 줄만 고쳐주세요.**

3D 뷰어는 GUI 안에서 호출되거나 단독으로:
```bash
python scripts/view_3d.py <shot 경로>
```

## 4. 데이터 / 모델 배치 (직접 준비)

이 브랜치엔 코드만 있습니다. 다음은 본인이 따로 넣어야 합니다.

- **데이터셋** — 촬영 샷(RGB + organized PLY + `_info.json` intrinsics)을 GUI에서 폴더로 열어서 사용
- **모델 가중치** — YOLO seg `.pt`, SAM2 체크포인트는 별도 배치
  (용량 문제로 git에 포함하지 않음 → `.gitignore`)

## 5. 주요 단축키 (라벨러)

| 키 | 동작 |
|---|---|
| `0`~`4` | 선택 폴리곤에 클래스 지정 (지정 후 다음 미라벨로 이동) |
| `Tab` | 다음 미라벨 폴리곤으로 이동 |
| `Space` | 선택 폴리곤 verified 토글 (△ ↔ ✓) |
| `W` | 그리기 모드 (Enter/더블클릭/우클릭=닫기, Esc/W=취소) |
| `E` | 편집 모드 (E=확정, Esc=취소) |
| `D` | 선택 폴리곤 삭제 |
| `U` | 실행 취소(undo) |
| `S` / `Q` | 저장 |
| `N` / `P` | 다음 / 이전 샷 (자동 저장) |
| `Y` | train/val split override 토글 |
| `H` | height map 레이어 토글 |
| `M` | normal map 레이어 토글 |
| `K` | heatmap 레이어 토글 |
| `O` | OOD(이상탐지) 오버레이 토글 |
| `F` | 화면 맞춤(zoom fit) |
| `+` / `-` | 확대 / 축소 |
| 방향키 | 화면 패닝 |

마우스: 좌클릭=선택/점찍기, 드래그=이동, 우클릭=메뉴/폴리곤 닫기,
더블클릭=폴리곤 닫기, 휠=확대/축소, 휠클릭 드래그=패닝.

SAM2 모드: `Esc`/`A`=종료, `0`~`4`=클래스 지정.

## 6. 참고

- 클래스 정의(nc=6)는 `config/default.yaml` 기준입니다.
- 파지점(피킹) 로직은 `scripts/label_logic/picking.py`에 있으며 GUI가 직접 호출합니다.
