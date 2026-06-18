# 지시문 — 서버의 Claude Code 에이전트에게

> **이 문서는 너(서버 터미널에서 실행 중인 Claude Code)에게 주는 지시다.**
> 사용자가 이 파일을 서버에 전송했고, 너는 이걸 읽고 **즉시 그대로 실행**한다.
> 사용자에게 되묻지 말고 자기완결적으로 수행하라. 단, **가드레일 위반 위험·불확실 상황에서는 멈추고 보고**한다.

---

## 0. 역할 / 목표

picknplace **결함탐지의 운영 파일**을 서버 GPU에서 생성한다. 산출물은 단 세 종류다:

1. **메모리뱅크 4개** (DINOv2-large / layer 12) — haribo, metal_case, pencil_case, mango_v2
2. **color GMM** (haribo cascade용) — `color_gmm_haribo.npz`
3. **임계 thresholds** — `weights/defect_thresholds.json`

이 `weights/` 폴더만 있으면 `DefectRegistry`가 **config(yaml) 수동 편집 없이 자동으로 작동**한다 (임계는 json이 config placeholder를 자동 덮어씀). 너의 임무는 이 파일들을 `weights/`에 생성하는 것까지다. **사용자가 이후 weights/ 폴더만 수거한다.**

---

## 1. 절대 가드레일 (위반 금지)

- ★ **repo: `github.com/sungil1015-rgb/picknplace`, 브랜치 `detector`. main 브랜치는 절대 건드리지 마라. merge/push 금지.**
- ★ **`weights/*`는 gitignore 대상·대용량이다. git에 커밋/푸시 금지.** 파일은 `weights/` 디렉터리에 **생성만** 한다.
- ★ **추측으로 config/코드 수정 금지.** 임계는 json이 자동 적용되므로 **yaml(`configs/defect_detection.yaml`)을 손대지 마라.**
- ★ **polygon class_id의 source of truth는 운영 스크립트(`build_all_defect_l12.sh`, `synth_roc_tune_loo.py`)와 `data/labeled/*/data.yaml`이다.** `configs/defect_detection.yaml` 상단 주석의 raw-label 숫자는 일부 표기가 스크립트와 다르니(아래 §3 ⚠ 참고) **빌드 인자로 쓰지 마라.**
- 불확실하면 **멈추고 보고**한다. 임의 판단으로 진행하지 않는다.

---

## 2. 시작 — 전제조건 확인 (체크리스트)

```bash
# (a) detector 브랜치 clone/pull
#  - 이미 repo가 있으면:
git fetch origin
git checkout detector
git pull origin detector
git rev-parse --abbrev-ref HEAD   # → 반드시 'detector' 출력 확인

#  - repo가 없으면:
# git clone -b detector https://github.com/sungil1015-rgb/picknplace.git
# cd picknplace

# (b) ★ 정상 라벨 데이터 존재 확인 — 없으면 중단·보고
ls -d data/labeled/ && ls data/labeled/
```

- **`data/labeled/`(정상 라벨 데이터)가 없으면 즉시 중단하고 보고하라.** 데이터 없이는 뱅크도 color GMM도 만들 수 없다.
- 현재 브랜치가 `detector`가 아니면 중단하고 보고하라.

### 환경 확인

```bash
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.version.cuda)"
python -c "import transformers, sklearn, cv2, matplotlib; print('deps ok')"
```

- **GPU 권장.** torch는 cu118+ 권장. 필요: `transformers`(dinov2-large ~1.2GB 다운로드), `scikit-learn`, `opencv-python`, `matplotlib`(튜닝 ROC/히스토그램 figure 저장에 사용).
- dinov2-large는 첫 forward에서 자동 다운로드된다(이미 받아져 있을 수 있음).

---

## 3. 핵심 실행 — 이거 하나로 전부 된다

```bash
bash scripts/build_all_defect_l12.sh 2>&1 | tee /tmp/defect_build.log
```

이 스크립트가 **고정된 순서로 전 과정을 한 번에 수행**한다(중간 단계를 수동으로 건너뛰지 마라):

1. **메모리뱅크 4개 빌드** (large / L12): haribo, metal_case, pencil_case, **mango_v2**
2. **color GMM** (`color_gmm_haribo.npz`) — 이미 있으면 재사용(layer 무관). **없으면** 스크립트가 자동으로 `scripts/build_color_prior.py --class_id 1 --class_name haribo`를 실행한다(haribo color 신호도 `data/labeled/` 필요).
3. **임계 튜닝** → `weights/defect_thresholds.json` (`synth_roc_tune_loo.py`, strict LOO ROC)
4. **검증**: thresholds.json **4키 단언** + **registry strict 로드 검증** (layer/dim/가드 불일치면 여기서 에러로 멈춤)

스크립트는 `set -euo pipefail`이라 중간 실패 시 즉시 멈춘다. 위처럼 **전체 로그를 `tee`로 캡처하라**(보고에 필요).

### polygon class_id (빌드 인자 — 운영 스크립트 확정값)

| class | polygon class_id (`--class_id`) | 비고 |
|---|---|---|
| haribo | **1** | type=cascade (color GMM 필요) |
| mango (젤리껌) | **2** | build가 `--output weights/patchcore_mango_v2.npz`로 저장 |
| metal_case | **3** | |
| pencil_case | **5** | |

> ⚠ **두 가지 주의:**
> 1. polygon class_id는 classifier class_index(0~5)와 **다른** raw label이다.
> 2. `configs/defect_detection.yaml` 상단 주석에는 metal_case=4 / pencil_case=3 / bottle=5로 적힌 표기가 있으나, **실행 스크립트(build/tune)가 쓰는 값은 metal_case=3, pencil_case=5다.** 빌드 인자는 **반드시 위 표(=스크립트값)를 따르라.** yaml 주석 숫자를 빌드 인자로 쓰지 마라.
>
> 확실히 검증하려면 라벨 매핑 원본을 보라:
> ```bash
> cat data/labeled/*/data.yaml | head -40   # polygon class_id ↔ 이름 매핑 원본
> ```
> (기존 npz가 **이미 있을 때만** 메타로 교차확인 가능 — 첫 빌드 전엔 npz가 없으니 이 방법은 재실행 시에만 쓴다:)
> ```bash
> python -c "import numpy as np; d=np.load('weights/patchcore_metal_case.npz'); print('class_id meta:', int(d['class_id']))"
> ```

---

## 4. 산출물 (이게 weights/에 생겨야 한다)

```
weights/
├─ patchcore_haribo.npz          (cascade)
├─ patchcore_metal_case.npz      (patchcore)
├─ patchcore_pencil_case.npz     (patchcore)
├─ patchcore_mango_v2.npz        (patchcore, "운에 맡김" 모드)
├─ color_gmm_haribo.npz          (haribo color 신호, layer 무관)
└─ defect_thresholds.json        (임계 자동 적용, 키 4개 = memory_bank 파일명)
```

`defect_thresholds.json` 키는 **memory_bank 파일명**이다: `patchcore_haribo.npz`, `patchcore_mango_v2.npz`, `patchcore_metal_case.npz`, `patchcore_pencil_case.npz` 4개.

---

## 5. 검증 (빌드 스크립트 외 독립 확인)

```bash
python -c "from src.defect.registry import DefectRegistry; r=DefectRegistry('configs/defect_detection.yaml'); print(r); print(r.threshold_overrides)"
```

**성공 기준:**
- 에러 없이 `DefectRegistry(enabled=True, classes=[0, 1, 2, 3])` 출력
- `threshold_overrides`(= `r.threshold_overrides`)에 클래스별 τ가 **채워져** 있음 (비어있지 않음)

> 참고: `classes=[0, 1, 2, 3]`이 기대값이다(이번 작업은 mango disable 금지이므로 1번 포함). 만약 누군가 mango(1)를 disable했다면 `[0, 2, 3]`이 되는데, 너는 **yaml을 건드리지 않으므로** [0,1,2,3]이어야 정상이다.

성공하면 운영 파일 준비 완료다.

---

## 6. 오류 대응 (이대로 처리)

| 증상 | 원인 | 조치 |
|---|---|---|
| **(a)** `'layer mismatch'` ValueError | 뱅크가 L12로 안 빌드됨 | `--layer 12`로 재빌드 (default가 이미 12이므로 명시적으로 `python scripts/build_patchcore_refs.py --class_id <id> --class_name <name> --model facebook/dinov2-large --layer 12` 실행) |
| **(b)** `'feature dim mismatch'` | 구 base(dim 768) 뱅크 혼입 / 모델 잘못 | 정상 large 뱅크는 **dim 1024**다. 768이면 구 base 뱅크 혼입 → dinov2-**large** 사용 확인하고 weights/의 구 뱅크 제거 후 재빌드 |
| **(c)** build가 `"No objects found"` | polygon class_id 틀림 | `data/labeled/.../data.yaml`(원본 매핑) 또는 기존 npz의 `class_id` 메타로 polygon id 재확인 후 올바른 `--class_id`로 재빌드. **yaml 주석 숫자 말고 §3 표값을 신뢰하라.** |
| **(d)** `synth_roc_tune_loo.py` exit 1 | 일부 클래스 뱅크 실패/스킵 (thresholds.json 불완전) | 로그에서 `튜닝 실패/스킵 클래스: [...]` 확인 → 해당 뱅크 재빌드. ★재빌드 후엔 **반드시 `python scripts/synth_roc_tune_loo.py`를 다시 실행**해야 thresholds가 뱅크와 정합한다(LOO는 뱅크의 객체 순서에 의존). |
| **(e)** GPU OOM | 가능성 낮음 (객체 1개씩 forward) | 발생 시 보고 |

**원칙:** 위 표에 없는 에러, 또는 원인 불명확 시 **임의 수정 말고 로그와 함께 보고**하라.

---

## 7. mango (젤리껌) 특별 주의

- mango는 검증된 fusion 대신 **plain patchcore "운에 맡김" 모드**로 운영된다 (TPR 정량검증 없음, "정상에서 벗어나면 잡는다"는 작동은 함).
- ★ **mango 임계는 보수적(=널널하게, 더 높은 τ)으로 운영한다.** 근거: mango(젤리껌)는 비닐 재질이라 정상도 조금씩 구겨지는데, PatchCore는 외관 거리 기반이라 **정상 구겨짐도 이상으로 본다.** 임계를 세게(낮게) 주면 정상품이 오거부된다. 그래서 `synth_roc_tune_loo.py`가 mango에 한해 `τ = max(F1-best, normal_p99.5)`(정상 LOO 분포 99.5분위)를 적용해 **정상품 오거부를 ~0.5%로 통제**한다. 합성결함이 아니라 **신뢰 가능한 정상 분포(구겨진 정상 포함)**에 기준을 맞추는 것이다. **이 분기는 코드에 내장돼 있으니 yaml/threshold를 손대지 마라.** 다른 클래스(haribo/metal_case/pencil_case)는 F1-best τ 유지.
- 튜닝 출력에서 **mango의 operating τ와 출처(F1-best vs normal_p99.5 중 채택값), normal_p99/p99.5, AUROC / separation(OVERLAP vs SEPARATED)을 반드시 보고**하라. operating τ가 F1-best보다 높으면 보수적 의도대로 정상 동작이다.
- ⚠ 한계(보고 시 함께 명시): higher τ는 정의상 **실제 결함 미탐 위험을 키운다**(오거부 0.5% 통제 ≠ 결함 누락 없음). 또 빌드 데이터의 정상 꼬리가 시연환경보다 덜 구겨져 있으면 실제 오거부가 0.5%보다 클 수 있으니 normal_p99/p99.5를 함께 봐 시연환경에서 재확인한다.
- **약하면(예: AUROC < 0.7 이거나 OVERLAP) 사용자에게 알리고**, "config에서 class 1(mango)을 `enabled: false`로 둘 수 있다"고 **안내만** 하라. (과거 plain-patchcore mango AUROC는 0.7대였으니 0.6 미만만 약한 게 아니다.)
- ★ **절대 강제로 disable하거나 yaml을 수정하지 마라. 보고만 한다.**

---

## 8. 마무리 보고 형식 (이 형식으로 보고하라)

1. **빌드된 뱅크별 객체 수** — 각 클래스 `objects extracted`, `memory bank shape`(dim이 1024인지 확인)
2. **`defect_thresholds.json` 내용** — 클래스별(키=memory_bank 파일명) τ 전부. mango는 operating τ(=max(F1-best, normal_p99.5))이며 채택 출처(F1-best vs p99.5)를 명시.
3. **tune AUROC 표 요약** — class / AUROC / sep? / F1-best τ / **operating τ + 출처** / TPR / FPR / normal_p99·p99.5 (synth_roc_tune 최종 표 그대로)
4. **경고 / 이상 항목** — OVERLAP 발생, mango 약함, 스킵된 클래스, 비정상 객체 수, color GMM을 새로 빌드했는지 여부 등
5. **생성 파일 목록 + 절대경로** — `weights/` 내 6개 파일의 절대경로

마지막 줄에 명시: **"사용자는 이 `weights/` 폴더만 받으면 된다."**

---

## 9. 다시 한번 (핵심 요약)

- 실행: `bash scripts/build_all_defect_l12.sh 2>&1 | tee /tmp/defect_build.log`
- main 금지 / push·commit 금지 / yaml 수정 금지
- `data/labeled/` 없으면 중단·보고
- polygon class_id는 §3 표값(스크립트 기준): haribo=1, mango=2, metal_case=3, pencil_case=5 — yaml 주석 숫자 쓰지 말 것
- 뱅크 재빌드하면 `synth_roc_tune_loo.py`도 반드시 재실행
- mango 약하면(AUROC<0.7 또는 OVERLAP) 보고만 (disable 강제 금지)
- 불확실하면 멈추고 보고