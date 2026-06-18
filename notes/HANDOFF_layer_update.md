# 핸드오프 — 불량탐지 detector 클래스별 best layer 적용 (서버 작업)

대상: 회사 서버에서 코드 통합 담당하는 분
브랜치: **`detector`** (커밋 `07e7349`). ⚠️ **main 건드리지 마세요.**
작성: 2026-06-18

---

## 0. 한 줄 요약

layer ablation으로 찾은 **클래스별 최적 DinoV2 layer**를 config에 반영해뒀습니다 (코드/설정만, "staged").
회사 서버에서 **① 메모리뱅크를 새 layer로 재빌드 → ② threshold 재튜닝** 두 스크립트만 돌리면 운영 가능합니다.
둘 다 라벨링 데이터만 있으면 코드로 처음부터 계산되며, **본인 서버에 이미 데이터 있음 = 전제조건 충족.**
지금 상태로 detector를 로드하면 **layer 가드가 일부러 에러로 멈춥니다** (뱅크가 아직 옛 L8이라). 아래 작업이 그 매칭을 맞추는 것.

> ⚠️ **사전 산출물(`outputs/eda/` 의 ablation 뱅크·τ)은 회사 서버에 없습니다** (작성자 학교 A5000에만 존재).
> 따라서 **단축 복사 경로 없음 — 아래 재빌드 + 재튜닝을 처음부터 수행**하세요. 데이터만 있으면 됩니다.

---

## 1. 코드에서 이미 바뀐 것 (커밋 `07e7349`, 손댈 필요 없음)

| 파일 | 변경 |
|---|---|
| `configs/defect_detection.yaml` | 클래스별 `dinov2_layer` 갱신 + threshold 재튜닝 TODO 주석 |
| `src/defect/patchcore_lite.py` | config가 layer 기준(source of truth). **뱅크 layer ≠ config layer면 명시적 에러** (조용히 틀린 결과 방지) |
| `scripts/build_patchcore_refs.py` | `--layer` 인자 추가 (클래스별 빌드 가능) |

## 2. 클래스별 목표 상태

| class | 이전 | **목표 layer** | 메모리뱅크 파일 | polygon class_id | 비고 |
|---|---|---|---|---|---|
| haribo | L8 | **L5** | `weights/patchcore_haribo.npz` | **1** | cascade. 색상뱅크는 그대로(아래 주의) |
| pencil_case | L8 | **L5** | `weights/patchcore_pencil_case.npz` | **5** | patchcore 단독 |
| metal_case | L8 | **L7** | `weights/patchcore_metal_case.npz` | **3** | patchcore 단독. SEPARATED 도달 layer |
| mango | L8 | **L8 (유지)** | `weights/patchcore_mango_v2.npz` | 2 | **손대지 마세요.** fusion, `enabled:false` 그대로 |

> polygon class_id 출처: `scripts/layer_ablation_build.py`의 `TARGETS = [(1,haribo),(3,metal_case),(5,pencil_case)]`.
> 그래도 **실행 전 기존 npz로 한 번 확인** 권장:
> `python -c "import numpy as np; d=np.load('weights/patchcore_metal_case.npz'); print(int(d['class_id']), str(d['class_name']))"`

---

## 3. 서버에서 할 일

### Step 0 — 브랜치
```bash
git fetch origin && git checkout detector && git pull
# ⚠️ main 으로 merge/push 금지
```

### Step 1 — 기존 뱅크 백업 (덮어쓰기 전)
재빌드가 `weights/patchcore_*.npz`를 덮어씀. L8 원본 보존:
```bash
cp weights/patchcore_haribo.npz      weights/patchcore_haribo.L8.bak.npz
cp weights/patchcore_pencil_case.npz weights/patchcore_pencil_case.L8.bak.npz
cp weights/patchcore_metal_case.npz  weights/patchcore_metal_case.L8.bak.npz
```

### Step 2 — 메모리뱅크 재빌드 (새 layer로)
라벨링 데이터(`data/labeled/`)로 처음부터 빌드. 각 1~2분, GTX4080이면 충분(DinoV2-base, VRAM 여유):
```bash
python scripts/build_patchcore_refs.py --class_id 1 --class_name haribo      --layer 5
python scripts/build_patchcore_refs.py --class_id 5 --class_name pencil_case --layer 5
python scripts/build_patchcore_refs.py --class_id 3 --class_name metal_case  --layer 7
```
→ 결과 npz에 `dinov2_layer`가 박혀서 이후 가드/튜닝이 layer를 자동 인식함.
**mango는 빌드하지 마세요** (L8 그대로, `patchcore_mango_v2.npz` 유지).

### Step 3 — threshold 재튜닝
layer 바뀌면 점수 척도가 달라져 기존 threshold 무효. 새 layer 기준으로 다시 계산:
```bash
python scripts/synth_roc_tune_loo.py
```
- 이 스크립트는 **각 뱅크의 `dinov2_layer` 메타를 읽어 같은 layer로 테스트 patch를 뽑습니다** (layer-aware로 수정됨). 그래서 재빌드만 해두면 추가 인자 없이 그대로 돌리면 됨.
- 출력 마지막의 `=== 최종 권장 threshold 표 ===`에서 클래스별 **F1-best τ**를 읽어 config에 반영:
  - haribo: `detectors.0.signals.patchcore.threshold` ← (현재 8.6, 무효 placeholder)
  - pencil_case: `detectors.2.threshold` ← (현재 9.2, 무효 placeholder)
  - metal_case: `detectors.3.threshold` ← (현재 6.0, 참고값. 실측으로 교체)
  - **운영점 기준: F1-best** (원래 방식). 안전 우선이면 표의 FPR 보고 FPR≤5% 지점으로.
- **mango 행은 무시**하세요 — fusion 기반이라 이 plain-patchcore τ는 운영값 아님 (뱅크 없으면 자동 skip됨).

### Step 4 — 검증
```bash
python -c "from src.defect.registry import DefectRegistry; r=DefectRegistry('configs/defect_detection.yaml'); print(r)"
# layer 가드 에러 안 나면 매칭 성공. 에러 나면 뱅크 layer ≠ config layer → Step 2 재확인
```
합성 결함으로 smoke 한 번 돌려 정상/결함 점수 분포가 threshold 기준 갈리는지 확인.

---

## 4. 함정 / 주의 (꼭 읽기)

1. **`--class_id`는 polygon 세그 id** (classifier class_index와 다름). haribo=1, metal=3, pencil=5. 추측 말고 위 표/기존 npz로 확인.
2. **색상 뱅크(`color_gmm_haribo.npz`)는 건드리지 마세요.** LAB 픽셀 GMM이라 **DinoV2 layer와 무관** → 재빌드 불필요. haribo의 color/topology/normal 신호 threshold도 layer 무관이라 유지. **재튜닝 대상은 cascade 안의 `signals.patchcore.threshold` 하나뿐.**
3. **mango는 손대지 마세요.** L8 유지, `enabled:false` 유지. fusion 가중치 w=0.25는 실물 찢김 데이터 있어야 검증 가능 — 이번 작업 범위 밖.
4. **뱅크 layer 메타 필수.** 수정된 `build_patchcore_refs.py`로 빌드하면 npz에 `dinov2_layer`가 박혀 가드가 동작. 옛 방식(layer 메타 없는) 뱅크 재활용 금지 — 가드가 못 잡고 조용히 틀림.
5. **main 금지.** 모든 작업은 `detector` 브랜치.
6. **GTX4080(Ada) 환경.** 연산/VRAM은 전혀 문제 없음(DinoV2-base ~350MB). 단 PyTorch가 Ada 지원하는 빌드여야 함 (CUDA 11.8+ / cu118 이상). 기존 환경 그대로면 신경 안 써도 됨.
7. **튜닝 스크립트는 layer-aware로 수정됨** (`synth_roc_tune_loo.py`가 뱅크 메타에서 layer 읽음). 옛 버전은 L8 하드코딩이라 새 layer에서 틀린 τ가 나왔음 — 반드시 최신 `detector` pull 후 사용.

---

## 5. 완료 기준 (DoD)

- [ ] haribo/pencil L5, metal L7 뱅크가 `weights/`에 있고 config layer와 일치
- [ ] 위 3개 threshold가 새 layer 기준으로 재튜닝돼 config 반영
- [ ] `DefectRegistry(...)` 로드 시 가드 에러 없음
- [ ] 합성 결함 smoke 통과
- [ ] mango / color 뱅크 / main 미변경 확인
