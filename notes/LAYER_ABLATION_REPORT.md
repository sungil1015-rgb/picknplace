# DinoV2 Layer Ablation — PatchCore-lite 결함 탐지

작성: 2026-06-12. 대상: 팀원/연구 미팅. 5분 정독 분량.
원본 raw 결과: [`../outputs/eda/layer_ablation_summary.md`](../outputs/eda/layer_ablation_summary.md)

---

## TL;DR

> **현재 운영 layer 8은 합리적 선택이었으나, L7이 모든 활성 클래스에서 동등 이상 (metal_case는 유일하게 SEPARATED 도달).** mid-layer 5~8이 sweet spot임을 실증 확인. layer 변경은 금요일 로봇 테스트 후 PR로 검토 권장.

| 클래스 | 현재 L8 AUROC | best layer | best AUROC | Δ | 변경 권장 |
|---|---|---|---|---|---|
| **metal_case** | 0.998 (OVERLAP) | **L7** | **0.999 (SEPARATED)** | +0.001 | L7 (분리 도달) |
| **haribo** | 0.929 | **L5** | 0.942 | +0.013 | 검토 (rigid 객체 아님 — 합성 결함 신뢰도 한계) |
| **pencil_case** | 0.986 | **L5** | 0.996 | +0.010 | L5 또는 L7 |

→ **단일 layer 변경 안**: L8 → **L7** (운영 단순성 유지, 3 클래스 모두 동등 이상).

---

## 왜 했는가

기존 PatchCore-lite는 `hidden_states[9]` (= post-block-8 = "layer 8") 단일 고정. 근거는:

- PatchCore 원논문 (Roth 2022) WideResNet50 mid-layer 활용
- DinoV2 ViT mid-layer가 "정상 표면 패턴은 좁은 manifold, 결함은 manifold 밖"이라는 통설

**그러나 우리 데이터에서 layer별 비교는 한 번도 안 함.** 즉 L8이 통설 기반의 합리적 추측이지 실증 검증은 아니었음. 이번 EDA로 11 layer 전체 비교.

---

## 방법

- 비교 layer: **L1 ~ L11** (11개)
- 제외: L0 (pre-block embedding), L12 (classifier 사용 중) — 자세한 이유는 [부록 A](#부록-a-제외-layer-근거)
- 클래스: metal_case, pencil_case, haribo (운영 활성 3개)
- 프로토콜: strict LOO + 합성 결함 5변형/객체 × 80객체 (`scripts/synth_roc_tune_loo.py`와 동일)
- 효율: `output_hidden_states=True` 한 번 forward에 11 layer 동시 추출 → 전체 실험 5분 (build 50초 + eval 2분 20초)

산출 데이터: 메모리뱅크 33개 (11 layer × 3 class, 약 6GB, `outputs/eda/memory_banks/`).

---

## 핵심 결과 한 장

![AUROC vs layer](../outputs/figs/layer_ablation/auroc_vs_layer.png)

3 클래스 모두 일관된 패턴: **얕은 layer 낮음 → mid-layer (5~8) 최고 → L10 dip → L11 회복**.

---

## 인사이트 4가지

### 1. mid-layer (5~8) sweet spot 실증 확인

| 통설 | 실증 |
|---|---|
| L5~L9가 결함 탐지에 적합 (정상 manifold 좁고 결함이 밖) | ✓ 3 클래스 모두 L5~L8 AUROC > 0.929 |
| 얕은 L1~L3은 raw edge/color noise라 부적합 | ✓ L1 AUROC 0.745~0.927, L2 0.803~0.963 |
| 깊은 layer는 semantic이라 결함 신호 약화 | △ L10에서만 dip, L11은 다시 회복 (아래 #3) |

PatchCore 원논문 직관이 우리 데이터에서도 유효. 단 "깊을수록 안 좋다"는 단순화는 부정확.

### 2. L7이 metal_case에서 유일하게 SEPARATED

metal_case L7 AUROC **0.999, TPR 100%, FPR 1%** — **정상/결함 score 분포가 완전히 분리** (모든 정상 < 모든 결함).

| 지표 | L7 | L8 (현재) |
|---|---|---|
| AUROC | 0.999 | 0.998 |
| F1-best τ | 5.76 | 8.93 |
| TPR | 1.00 | 1.00 |
| FPR | **0.01** | 0.04 |
| 분포 | **SEPARATED** | OVERLAP |

해석: metal_case는 rigid + 단순 표면 + sharp 스크래치 결함이라 mid-layer 어느 곳에서든 잘 잡히지만, **L7이 가장 깔끔한 신호 분리**를 제공. 운영 시 threshold 마진이 가장 안전.

### 3. L11 의외의 강세 — semantic이지만 결함 잘 잡음

| class | L11 AUROC | 순위 |
|---|---|---|
| metal_case | 0.996 | 3위 (L7, L8 다음) |
| pencil_case | 0.996 | 공동 1위 (L5와) |
| haribo | 0.861 | 7위 |

통설 "마지막 layer는 객체 카테고리만 보고 결함은 못 본다"와 충돌. **가설**: 우리 데이터셋의 정상 객체끼리는 매우 동질적 (라벨된 단일 카테고리, 표면 다양성 적음) → semantic representation 안에서도 결함의 미세 변화가 살아남음. 단, deformable 객체 (haribo)는 정상 자체의 의미 분산이 커 L11에서 성능 떨어짐.

**주의**: L11과 L12를 혼동하지 말 것. L12 (= `hidden_states[12]`)는 classifier가 이미 cosine KNN으로 사용 중 — 결함 score로 재사용은 부적합.

### 4. L10 일관된 dip — semantic transition zone

3 클래스 모두 L10에서 AUROC 가장 낮음 (또는 매우 낮음):
- haribo L10 0.766 (L9 0.917, L11 0.861)
- pencil_case L10 0.924 (L9 0.973, L11 0.996)
- metal_case L10 0.964 (L9 0.983, L11 0.996)

해석 후보:
- DinoV2-base에서 block 9 (= L10)이 "low-level texture → object semantics" 전환점. 표현이 불안정하거나 noisy.
- 다른 ViT 모델에서도 비슷한 mid-late dip 보고됨

**시사**: L10은 운영 layer 후보에서 명시적으로 제외해야 함.

---

## 결론

### 데이터가 말하는 것
1. L8 default는 합리적 선택이었음 (3 클래스 모두 ≥ 0.929)
2. **L7이 모든 클래스에서 동등 이상**, metal_case에서는 SEPARATED 달성
3. L10은 운영에서 피해야 할 layer
4. L11은 우리 데이터에선 강력하지만 (semantic 영향 적음) 일반화 보장 없음

### 권장 액션 (우선순위)
1. **금요일 로봇 테스트는 L8 그대로 유지** — 위험 회피
2. 테스트 후, **L7 전환 PR 준비**:
   - 메모리뱅크 재빌드 (`scripts/build_patchcore_refs.py`의 `DINOV2_LAYER = 8` → `7`)
   - LOO ROC 재tune → threshold 정수 자릿수 보수적 절상 (참고값: metal 5.76 → 6, pencil 5.85 → 6, haribo 5.41 → 6)
   - 실 false rate 비교 측정 (L8 vs L7) 후 결정
3. `configs/defect_detection.yaml` 에 `dinov2_layer` 필드 추가 → 클래스별 layer 분리 운영 가능성 열어둠 (현재는 클래스 무관 단일 layer)

### 권장하지 않음
- L11로 즉시 전환 — haribo 성능 떨어짐
- 클래스별 layer 분리 (L7/L5/L7) — 운영 복잡도 증가 대비 이득 작음 (≤ 0.005)
- L9 사용 — L7~L8 대비 명확한 이득 없음

---

## 부록 A: 제외 layer 근거

| layer | hidden_states 인덱스 | 제외 근거 |
|---|---|---|
| L0 | hidden_states[0] | patch embedding (pre-block). Transformer attention 적용 전, raw 14×14 RGB embedding 수준. anomaly 신호 부재 — 정상/결함 모두 비슷한 raw 통계. |
| L12 | hidden_states[12] | post-block-11 (마지막). DinoV2 KNN classifier가 fused embedding (CLS + patch_mean)에서 이 표현을 사용 중. semantic level이라 "metal_case + 스크래치"가 "metal_case 정상"과 같은 카테고리로 수렴 → patch 거리가 anomaly 신호 잃음. 또한 동일 표현 재사용은 classifier 신호와 중복. |

## 부록 B: 클래스별 layer × AUROC 전체 표

raw 표는 [`../outputs/eda/layer_ablation_summary.md`](../outputs/eda/layer_ablation_summary.md) 참조.

핵심 layer 요약 (각 클래스의 best, current, worst):

| class | L1 | L3 | L5 | **L7** | **L8 현재** | L10 | L11 |
|---|---|---|---|---|---|---|---|
| metal_case | 0.866 | 0.940 | 0.992 | **0.999** SEP | 0.998 | 0.964 | 0.996 |
| pencil_case | 0.927 | 0.978 | 0.996 | **0.992** | 0.986 | 0.924 | 0.996 |
| haribo | 0.745 | 0.824 | **0.942** | 0.941 | 0.929 | 0.766 | 0.861 |

## 부록 C: 재현

```bash
# 1) 11 layer × 3 class 메모리뱅크 빌드 (약 50초, A5000)
CUDA_VISIBLE_DEVICES=0 python scripts/layer_ablation_build.py
# → outputs/eda/memory_banks/mb_L{L}_{class}.npz (33개)

# 2) LOO ROC eval + 그림 + summary.md 생성 (약 2분 20초)
CUDA_VISIBLE_DEVICES=0 python scripts/layer_ablation_eval.py
# → outputs/figs/layer_ablation/{auroc_vs_layer, roc_curves_*, score_dist_*}.png
# → outputs/eda/layer_ablation_summary.md
```

산출물 시각화:
- `outputs/figs/layer_ablation/auroc_vs_layer.png` (한 장 요약)
- `outputs/figs/layer_ablation/roc_curves_{class}.png` (클래스별 11 layer ROC 곡선)
- `outputs/figs/layer_ablation/score_dist_{class}.png` (클래스별 11 layer score 분포)
