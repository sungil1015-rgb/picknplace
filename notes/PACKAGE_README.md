# picknplace 결함탐지 — 2026-06-12 통합 패키지

mango 복구 + layer ablation 최적 layer 확정 + 평가/시각/스크립트 일체.
**파지 우선순위/석션 포인트 개선 제안 (pickability-first)은 포함 안 됨** — 별도 설계 문서로만 존재 (notes/pick_logic_algorithm.html), 코드 반영 X.

---

## 시작점 (사람에 따라)

| 상황 | 먼저 보세요 |
|---|---|
| **5분 요약 결론만** | `notes/MANGO_QUANT_EVAL.md` § TL;DR |
| **mango 왜 실패했고 어떻게 고쳤나** | `notes/PROJECT_LOG.md` § 2026-06-12 (세션 3 mango 근본원인 + 세션 4 복구 + 세션 5 정량평가) |
| **클래스별 최적 layer** | `notes/LAYER_ABLATION_REPORT.md` + `notes/MANGO_QUANT_EVAL.md` § 7 |
| **운영 yaml 어떻게 바뀌었나** | `configs/defect_detection.yaml` (헤더 주석에 요약, mango id=1 블록이 fusion으로 교체됨) |
| **재현 (다른 layer 시도 등)** | 아래 § 재현 명령 |

---

## 핵심 변경 사항 (코드 반영분)

### 1. mango: cascade → WeightedFusionDetector (신규)
- `src/defect/fusion.py` 신규: `fused = 0.75·z(PatchCore max) + 0.25·z(Laplacian)`, mask 경계 7px erosion, normal 없으면 PatchCore 단독 fallback
- `configs/defect_detection.yaml` mango (id 1) 블록: `type: cascade` → `type: fusion`. **enabled: false 유지** — 실물 찢김 TPR 검증 후 전환
- `weights/patchcore_mango_v2.npz` 신규 (오염 1건 제외, 309 obj)
- `outputs/eda/mango_exclude_list.txt`: 정제 대상 — `0515/0000000455_20260515_141455:0` (mango 폴리곤 내 파란 HARIBO 제품 혼입, 육안 확정)

### 2. 합성 tear: 절대 픽셀 → mask 면적 비율
- `src/defect/synthesize.py`: `synthesize_tear(..., area_mode="mask_ratio", area_ratio_range=(0.03, 0.07))` 옵션 추가
- 근거: mango crop이 haribo의 2.1배라 224 resize 후 tear 점유율 4.2배 희석. 비율화로 동일 조건 평가
- 주의: 같은 물리 크기 결함이 큰 객체에서 희석되는 실제 운영 약점은 별개 — crop 448 / 타일링은 후속

### 3. PatchCoreLite top_k_mean 통계 옵션
- `src/defect/patchcore_lite.py`: `patch_aggregate="top_k_mean", top_k=4` 추가
- max vs top_k 클래스별 결과 (단순 max 우월 vs 결합 우월):
  - rigid (metal/pencil): top_k가 우월 (metal_case는 SEPARATED 달성, FPR 4%→1%)
  - deformable (mango/haribo): max가 우월 (소형 tear는 hot patch 적어 평균이 신호 희석)

### 4. build 스크립트 라벨 제외 지원
- `scripts/build_patchcore_refs.py --exclude <file>`: `<set>/<stem>:<line_idx>` 줄 단위 제외

### 5. registry — fusion 타입 추가
- `src/defect/registry.py`: `type: fusion` → `WeightedFusionDetector`

---

## 평가 결과 요약

### LOO ROC (비율 tear 3~7%, 80 obj × 5 변형, strict LOO)

| class | bank | AUROC (max) | AUROC (top_k) | 운영점 (FPR 5%) |
|---|---|---|---|---|
| metal_case | patchcore_metal_case.npz (기존) | 0.998 | **0.998 SEPARATED** (FPR 1%) | TPR 100% |
| pencil_case | patchcore_pencil_case.npz (기존) | 0.986 | **0.993** | TPR 95% |
| haribo | patchcore_haribo.npz (기존) | 0.880 | 0.796 | TPR 81% |
| **mango_v2** | patchcore_mango_v2.npz (신규) | **0.880** | 0.816 | TPR 70% |

### Layer ablation (11 layer × 4 class, 비율 tear)

| class | L1 | L3 | L5 | **L7** | **L8 (현행)** | L10 | L11 |
|---|---|---|---|---|---|---|---|
| metal_case | 0.866 | 0.940 | 0.992 | **0.999 SEP** | 0.998 | 0.964 | 0.996 |
| pencil_case | 0.927 | 0.978 | 0.996 | 0.992 | 0.986 | 0.924 | 0.996 |
| haribo | 0.745 | 0.824 | 0.911 | **0.913** | 0.883 | 0.705 | 0.796 |
| mango_v2 | 0.591 | 0.713 | 0.833 | 0.861 | **0.870** | 0.774 | 0.853 |

- L10 dip 4번째 재현 — 운영 금지 layer
- mango는 L8이 최적 (현행 유지 근거), 단 multilayer L4+L8+L11 결합은 0.870 → 0.888 (+0.018)
- 통합 전환 시 L7이 유일 후보 (mango -0.009 잡음 vs 다른 3 클래스 모두 개선)

### mango 정량평가 9종 배터리 핵심

| 테스트 | 결과 |
|---|---|
| 기본 AUROC | 0.885 (95% CI [0.850, 0.917]) |
| 지연 | 24.5 ms/객체 (A5000), 예산 16~30ms 충족 |
| 탐지 한계 | tear ~3% (mask 면적 기준) |
| **LOSO (신규 세션 리스크)** | **FPR 21~40%** (4~8배 폭증) — 신규 촬영 세션 시 정상 샘플 bank 보강 필수 |
| 글레어 스트레스 | FPR 44% — bank 글레어 모드 보강 필요 |
| mask 팽창 +10px | FPR 21% — Mask2Former 실 mask와 라벨 폴리곤 괴리 확인 필요 |
| 결함 종류 커버리지 | scratch 0.985 / 구멍 0.968 / 얼룩 0.961 / tear 0.891 |

---

## 운영 권장 액션 (우선순위)

1. **로봇 테스트 (오늘)**: 실 FPR + **mango 1~2봉지 일부러 찢어 투입** (fusion Laplacian 채널 TPR 실측 — 유일한 검증 수단)
2. **신규 촬영 세션 운영 수칙 제정**: bank 보강 후 가동 (LOSO 근거 — 가장 중요)
3. 테스트 후: mango `enabled: true` 전환 + lap_mu/sigma 캘리브 + fused τ 재tune
4. 후속: layer 통합 전환 (L7) 검토
5. 후속: 큰 객체 해상도 희석 대응 (crop 448 또는 2x2 타일링)

---

## 패키지 구조

```
mango_recovery_20260612/
├── PACKAGE_README.md                    ← 이 파일
├── src/defect/                          ← 코드 (8 파일)
│   ├── base.py
│   ├── cascade.py                       (기존)
│   ├── color_prior.py                   (기존)
│   ├── fusion.py                        ★ 신규
│   ├── logging.py
│   ├── patchcore_lite.py                ★ top_k_mean 추가
│   ├── registry.py                      ★ fusion 타입 추가
│   └── synthesize.py                    ★ 비율 tear 추가
├── scripts/                             ← 재현 스크립트
│   ├── build_patchcore_refs.py          ★ --exclude 옵션 추가
│   ├── mango_recovery_eval.py           ★ 신규 (LOO ROC 4클래스 재평가)
│   ├── mango_full_eval.py               ★ 신규 (9종 정량평가 배터리)
│   ├── mango_subtype_check.py           ★ 신규 (sub-type 가설 검증 — 기각)
│   ├── find_bad_labels.py               ★ 신규 (라벨 오염 자동 탐지)
│   ├── multilayer_combo.py              ★ 신규 (L4+L8+L11 결합 평가)
│   ├── layer_ablation_mango.py          ★ 신규 (mango v2 layer ablation)
│   ├── layer_ablation_build.py          (기존)
│   ├── layer_ablation_eval.py           (기존)
│   └── synth_roc_tune_loo.py            (기존, 비교 기준)
├── configs/defect_detection.yaml        ★ mango fusion + layer 주석
├── weights/                             ← 운영 메모리뱅크 (1.7GB)
│   ├── patchcore_mango_v2.npz           ★ 신규 (정제 309 obj)
│   ├── patchcore_mango_jelly.npz        (구버전 비교용)
│   ├── patchcore_haribo.npz
│   ├── patchcore_metal_case.npz
│   ├── patchcore_pencil_case.npz
│   └── color_gmm_*.npz
├── outputs/
│   ├── eda/
│   │   ├── mango_exclude_list.txt       ★ 라벨 정제 대상
│   │   ├── recovery_eval_<class>.json   ★ 4클래스 LOO 재평가
│   │   ├── layer_ablation_mango.json    ★ mango+haribo layer ablation v2
│   │   ├── layer_ablation_summary.md    (기존 — 6/10 metal/pencil/haribo)
│   │   ├── multilayer_combo.json        ★ L4+L8+L11 결합
│   │   ├── fulleval_<test>.json × 9     ★ 정량평가 배터리
│   │   └── memory_banks/                ← layer ablation banks (6.5GB)
│   │       ├── mb_L{1..11}_mango_v2.npz × 11   ★ 신규
│   │       ├── mb_L{1..11}_haribo.npz × 11
│   │       ├── mb_L{1..11}_metal_case.npz × 11
│   │       └── mb_L{1..11}_pencil_case.npz × 11
│   └── figs/
│       ├── mango_recovery/              ← 복구 ROC (4 클래스)
│       ├── mango_rootcause/             ← 4가설 검증 figure (15장)
│       ├── mango_fulleval/              ← 정량평가 시각
│       └── layer_ablation/              ← v1 (6/10) + v2 (6/12 mango 정제)
└── notes/                               ← 보고서
    ├── MANGO_QUANT_EVAL.md              ★ 정량평가 메인 보고서
    ├── LAYER_ABLATION_REPORT.md         (기존 — 6/12 v1)
    ├── PROJECT_LOG.md                   ← 세션 1~5 누적 일지
    └── 2026-06-12_layer_ablation_eda/   ← v1 정리 폴더
```

---

## 재현 명령

```bash
cd <패키지 풀고 작업 위치>
source ~/miniconda3/etc/profile.d/conda.sh && conda activate picknplace

# (1) mango bank 재빌드 (오염 제외)
CUDA_VISIBLE_DEVICES=0 python scripts/build_patchcore_refs.py \
    --class_id 2 --class_name mango_v2 \
    --exclude outputs/eda/mango_exclude_list.txt

# (2) LOO ROC 재평가 (클래스별, GPU 분산)
for c in mango haribo metal_case pencil_case; do
    CUDA_VISIBLE_DEVICES=0 python scripts/mango_recovery_eval.py --target $c
done

# (3) 정량평가 9종 배터리
for t in severity defect_types photometric geometric stress loso aggregate fulldata latency; do
    CUDA_VISIBLE_DEVICES=0 python scripts/mango_full_eval.py --test $t
done

# (4) layer ablation v2 (mango_v2 + haribo, 비율 tear)
CUDA_VISIBLE_DEVICES=0 python scripts/layer_ablation_mango.py --stage all

# (5) multilayer combo (L4+L8+L11)
CUDA_VISIBLE_DEVICES=0 python scripts/multilayer_combo.py
```

각 스크립트는 5~25분. (1)(2)(3)(4) 순서로 (5는 (4) 메모리뱅크 의존).

---

## 정직 고지사항

1. **fusion의 Laplacian 채널 TPR은 오프라인 검증 불가** — 합성 결함은 RGB만 바꾸고 normal map은 변경 안 됨 (Zivid PLY 조작 필요). 실물 찢김 테스트만이 유일한 검증 수단. w=0.25는 사전 설정값.
2. **비율 tear (3~7%)로 AUROC 회복은 평가 기준 변경** — 검출 능력 개선이 아님. 고정 카메라에서 절대 픽셀 = 고정 물리 크기 결함이므로, 같은 물리 크기 결함이 큰 객체 (mango)에서 224 resize에 희석되는 실제 약점은 그대로. 운영 enable 판단 시 함께 고려.
3. **LOSO 평가 (FPR 21~40%)는 배포 시 실제 보게 될 수치** — 본 LOO 평가 (FPR 5%)는 "같은 세션이 bank에 있다" 가정. 신규 촬영 세션은 반드시 bank 보강 후 가동.
4. **layer ablation 메모리뱅크 (outputs/eda/memory_banks 6.5GB) 포함** — fp32 텐서라 zip 압축 거의 0%. 이미지 (png/jpg 등)는 제외 (서버에 보관, `outputs/figs/`).

---

CMES picknplace · 2026-06-12 · 작성: 통합 정리
