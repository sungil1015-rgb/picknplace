# 불량 탐지(Defect Detection) 알고리즘 구조 분석 보고서

> 대상: 논문/발표자료 작성자. picknplace `detector` 브랜치 `src/defect/` 서브시스템과
> DINOv2 classifier(`src/classifier/dinov2.py`)의 상호작용을 코드 단위로 분해한 연구 보고서.
> 수식·구조는 소스에서, **임계값·통계(8.6/0.10/0.05/1.8/z=3.0/μ,σ/0.30 등)는 대부분
> `configs/defect_detection.yaml` 런타임 주입값 또는 별도 평가 산출물**임에 유의(코드 하드코딩 아님).
> 검증되지 않은 부분은 명시적으로 표기한다. (본 문서는 multi-agent 코드 대조 검증을 거침.)
>
> 작성: 2026-06-18 · 기준 커밋: detector @ 091ea75

---

## 0. 요약 (Executive Summary)

본 시스템은 빈피킹(bin-picking) 파이프라인에서 **분류된 객체 인스턴스 단위로 표면 결함을 판정**하는
무결함(normal-only) 학습 기반 이상 탐지(anomaly detection) 모듈이다. 핵심 설계는 다음과 같다.

1. **공유 기반(shared substrate)**: 객체 crop 규격과 DINOv2 backbone을 classifier와 공유한다.
   classifier는 **최종 layer**의 전역 임베딩(CLS+patch-mean)으로 *무엇인지*를 판정하고,
   defect detector는 **중간 layer**의 국소 patch 특징으로 *정상 표면에서 벗어났는지*를 판정한다.
   서로 다른 layer를 쓰기에 신호가 겹치지 않고 상보적이다.

2. **클래스별 독립 detector**: 분류 결과(`class_index`)가 `DefectRegistry`를 통해
   클래스 전용 detector로 라우팅된다. 객체 물성(rigid/deformable/투명)에 따라
   세 가지 전략 중 하나가 배정된다.
   - **PatchCore-lite** (rigid: metal_case, pencil_case): 단일 신호(patch L2 거리).
   - **Cascade** (deformable: haribo): 4신호 다수결 voting(≥2 of 4).
   - **Weighted Fusion** (mango): patch 거리 + 기하(normal Laplacian) z-가중 융합.
   - bottle(투명)은 제외, unknown은 우회.

3. **무결함 학습 + 합성 결함 캘리브**: memory bank는 정상 객체 patch만으로 구성하고,
   임계값은 정상 LOO 분포 vs 합성 결함(scratch/tear) 분포의 ROC F1-best로 오프라인 산출한다.

---

## 1. 시스템 컨텍스트 — 파이프라인 내 위치

추론 메인 루프(`src/pick_n_place.py:run()`)의 객체 처리 순서:

```
RGB-D + normal + ROI
   │
   ▼
① Mask2Former segmentation        detector.inference()          → instance masks
   │
   ▼
② mask 정제 + polygon             largest_component_mask, mask_to_polygon
   │
   ▼
③ DINOv2 KNN 분류                 classifier.classify_instances() → class_index, similarity, …
   │
   ▼  ┌─────────────────────────────────────────────────────────┐
   │  │ ★ DEFECT DETECTION 삽입 지점 (설계됨, 현재 미배선)        │
   │  │   DefectRegistry.get(class_index).score(rgb, mask,       │
   │  │                       normal, dinov2_features)           │
   │  │   → DefectResult(score, is_defect, score_map, votes,     │
   │  │                   components)                            │
   │  │   (action은 결과 필드가 아니라 detector 클래스 속성)     │
   │  └─────────────────────────────────────────────────────────┘
   ▼
④ Grasp priority scoring          priority_scorer.score_instances()
   │
   ▼
⑤ Suction point (top priority)    suction_pipeline.compute()
   │
   ▼
⑥ 정렬 + 결과 반환                sort by (suction, priority, similarity, score)
```

**삽입 지점의 논리**: defect는 ③ 분류 직후에 동작해야 한다. 이유는 (a) `class_index`가 있어야
클래스 전용 detector를 고를 수 있고, (b) ③에서 이미 수행한 DINOv2 forward를 재사용할 수 있으며,
(c) 판정 결과(`action`)가 ④ 우선순위·⑥ 정렬에 피드백되어 결함 객체를 파지 대상에서
배제(`reject`)하거나 후순위로 내릴(`priority_down`) 수 있기 때문이다.

> **현재 배선 상태 (정직)**: `src/defect/`를 import하는 코드는 `scripts/`(오프라인 평가)뿐이며,
> `pick_n_place.py`에는 아직 연결돼 있지 않다. 위 ★ 블록은 **설계된 통합 지점**이고
> 코드 반영은 후속 작업이다. 본 보고서는 detector 서브시스템 자체의 알고리즘을 기술한다.
>
> 부가 단서: (a) ③ classifier는 `type∈{dinov2,dino_v2}` **이면서** 별도 classifier config의
> `classifier.enabled==True`(**기본 False**)일 때만 로드된다. 미로드 시 `classifier=None` →
> `class_index`가 부착되지 않고 `class_id`는 `prediction.label`/0으로 fallback → defect 라우팅
> 입력이 비게 된다. (b) `run()` docstring의 "더미(detector=None)→state=-2" 서술은 **오기**다 —
> 실제 더미 분기는 하드코딩 polygon 2개로 `state=1`을 반환하고, `state=-2`는 polygon이 전부
> 탈락했을 때다.

---

## 2. 공유 기반(Shared Substrate)

### 2.1 객체 crop 규격 — classifier ⇄ defect 동일

두 모듈은 동일한 crop 규약을 독립 구현으로 갖는다
(`DinoV2KnnClassifier._crop_instance`, `PatchCoreLite._crop_with_mask`,
`build_patchcore_refs.crop_with_mask_alpha`). 규격:

| 항목 | 값 | 의미 |
|---|---|---|
| 형태 | 정사각형 | `side = ceil(max(bbox_w, bbox_h) · (1 + 2·0.03))` |
| 패딩 비율 | 0.03 (양변) | bbox 외곽 여유 |
| 배경 | gray 127 | mask 밖 픽셀을 일정 회색으로 채움 (배경 누설 차단) |
| 중심 | bbox 중심 정렬 | crop이 bbox 중앙에 오도록 |
| 리사이즈 | 224×224 | DINOv2 입력 |
| patch grid | 16×16 (=224/14) | ViT patch_size 14 → 256 token |

crop 규격이 동일하므로 동일 crop에서 **forward 1회**로 두 모듈의 입력을 동시에 만들 수 있다.

### 2.2 DINOv2 backbone — layer 소유권 분리

| | DINOv2 layer | 무엇을 보나 | 표현 |
|---|---|---|---|
| **Classifier** | **최종**(`last_hidden_state`, base=L12) | semantic 카테고리 | CLS(token0) [+ masked patch-mean] · RGB/gray 융합 → cosine KNN |

> classifier 임베딩은 `cls_weight·CLS + patch_mean_weight·patch_mean`인데 **기본값이
> `cls_weight=1.0, patch_mean_weight=0.0`이라 기본 임베딩은 CLS-only**다. patch-mean 항은
> config가 `patch_mean_weight>0`을 설정할 때만 기여(config-driven). 또한 classifier는
> `model(**inputs)`로 `last_hidden_state`만 쓰고 `output_hidden_states`를 요청하지 않는다 —
> 중간 layer 노출(§2.3 캐시)은 통합 시 추가해야 할 부분.
>
> 분류 출력 정확성: `confidence`(=class_score)는 승자 클래스 이웃의 **softmax 가중합**이라
> 정규화 확률이 아니다(클래스 합<1 가능; "probability"로 부르면 오류). `margin`은 승자 vs
> 차순위 클래스 mean-cosine 차이로, top-k에 단일 클래스만 있으면 `+inf`. unknown gating은
> `min_class_score/min_similarity/min_vote_ratio/min_margin` **4개의 OR**(하나라도 트립→unknown).
> 뱅크 라벨 {1..5}는 0-based로 shift(`labels−1`) 후에만 `UNKNOWN_CLASS_ID=5` 매핑 성립.
> classifier crop도 `crop_padding`/`masked_crop` config 키를 파싱하나 런타임은 하드코딩
> 상수(0.03/127)를 써 **비-configurable**.
| **Defect** | **중간**(`hidden_states[L+1]`, base L5~8) | 국소 표면 패턴 | mask 내 patch token L2 거리 |

이 분리가 핵심이다. semantic 최종 layer는 "metal_case 정상"과 "metal_case 스크래치"를
**같은 카테고리로 수렴**시켜 이상 신호를 잃지만, 중간 layer는 정상 표면 manifold가 좁아
결함 patch가 manifold 밖으로 튄다. 그래서:

- **layer ablation에서 최종 layer(L12)는 명시적으로 제외** — classifier가 이미 쓰고,
  semantic이라 anomaly 거리가 죽기 때문.
- backbone forward는 객체당 1회면 충분(두 모듈 공유). feature dim은 base 768, large 1024.

### 2.3 상호작용 계약(interaction contract) — feature 캐시

`PatchCoreLite.score(..., dinov2_features)`는 다음 dict 키를 기대한다:

```
dinov2_features["patch_layer_{layer}"]  # (n_tokens, D) torch tensor, device 상주
dinov2_features["crop_mask"]            # (side, side) bool — patch 선택용
```

- **캐시 경로(설계된 최적)**: classifier forward가 중간 layer patch token + crop_mask를
  위 키로 노출하면, defect는 **추가 forward 0회**로 동작(`_extract_patches_from_cache`).
- **fallback 경로(현재 동작)**: classifier는 `last_hidden_state`만 쓰고 중간 layer를
  노출하지 않으므로, 캐시 미스 시 defect가 **자체 backbone을 lazy 로드해 직접 forward**
  (`_extract_patches_via_forward` → `_ensure_backbone`). 정확하지만 연산 중복.
- **캐시 계약의 함정**: `_extract_patches_from_cache`는 `dinov2_features`에 `patch_layer_{L}`이
  있어도 **`crop_mask` 키가 없으면 None을 반환해 forward fallback**한다. 즉 캐시가 patch token만
  담고 crop_mask를 누락하면 "0-forward"가 성립하지 않는다(통합 시 둘 다 노출 필요).

> **통합 시 과제**: classifier가 클래스별 detector가 쓰는 layer 집합을
> `output_hidden_states=True`로 한 번에 추출해 캐시에 담으면 fallback이 사라진다.
> 클래스마다 layer가 다르면(예: metal L14, haribo L10) 캐시는 그 layer들을 모두 보유해야 한다.

---

## 3. 핵심 추상화

### 3.1 `DefectDetector` (ABC) — `src/defect/base.py`

모든 detector의 통일 인터페이스. 단일 메서드:

```python
score(rgb_bgr, mask, normal_map=None, dinov2_features=None) -> DefectResult
```

- `rgb_bgr`: (H,W,3) uint8 BGR
- `mask`: (H,W) bool/uint8 — 객체 1개
- `normal_map`: (H,W,3) float32 — 표면 법선(기하 신호용, 선택)
- `dinov2_features`: 캐시(선택)

클래스 속성(기본값): `threshold=0.0`, `action="reject"`, `enabled=True`.
`score()`는 `@abstractmethod`(본문 `raise NotImplementedError`)로 **구현 강제가 코드 레벨에서
보장**된다 — "통일 인터페이스" 주장의 실제 근거. 단 `action`의 3개 값과 `votes` 0~4 범위는
**base.py에서 검증·분기되지 않는 주석 규약**일 뿐(런타임 enforce 없음).

### 3.2 `DefectResult` (dataclass)

| 필드 | 타입 | 의미 |
|---|---|---|
| `score` | float | 이미지 레벨 스칼라 결함 점수 |
| `is_defect` | bool | 임계 통과 여부 |
| `score_map` | (H,W) ndarray | 시각화용 heatmap (없을 수 있음) |
| `votes` | int | cascade 전용(0~4) |
| `components` | dict | 디버그용 raw 신호값 |

필수/기본값: **`score`만 필수**, 나머지는 기본값(`is_defect=False`, `score_map=None`,
`votes=0`, `components=field(default_factory=dict)` — mutable 기본값 회피). 이 dataclass 계약은
detector 구현·호출·로깅 정확성에 load-bearing.

---

## 4. Detector 전략 (심층 분해)

### 4.1 PatchCore-lite — `src/defect/patchcore_lite.py`

**개념**: 정상 객체들의 중간 layer patch feature를 모은 memory bank를 두고,
테스트 patch마다 bank 내 **최근접 L2 거리**를 구해 그 통계를 결함 점수로 쓴다.
(PatchCore[Roth 2022]의 경량판 — coreset subsampling 없이 전 patch bank.)

**연산 흐름**:

```
crop(224) ─DINOv2 L→ patch tokens (256, D) ─mask select→ test_patches (n_q, D)
                                                              │
memory_bank (N, D)  ←─ 정상 patch 누적(오프라인 build)        │
                                                              ▼
          L2 NN:  d²(q, ·) = ‖q‖² + ‖m‖² − 2·q·mᵀ   (clamp≥0, sqrt)
                  nn_dist(q) = min_m d(q, m)              (n_q,)
                                                              │
                       aggregate ─┬─ max                      ▼
                                  ├─ top_k_mean (k=4)     score (scalar)
                                  └─ top_p_mean (p=0.01)
                                                              │
                       is_defect = score > threshold ◄────────┘
                       score_map = scatter(nn_dist)→grid→resize(W,H)
```

**세부**:
- L2 거리는 `(a−b)² = a²+b²−2ab` 전개로 행렬곱 1회에 계산(`_mb_sq` 사전 계산).
- **aggregate 분기 구조**: `max` / `top_k_mean(k=4)` / **`else`(top_p_mean이 default)**.
  `else`이므로 `"max"`·`"top_k_mean"` 외 임의 문자열(오타 포함)은 **silent하게 top_p_mean으로
  fallback**된다(주의).
- **aggregate 전략의 클래스 의존성**:
  - `max`: 단일 hot patch에 민감 → **소형 결함(작은 tear)**에 유리. **현행 배포 config는
    pencil/metal/mango 전부 `patch_aggregate: max`** (라우팅 표 §6과 일치).
  - `top_k_mean(k=4)`: 단일 outlier(글레어 등) 완화 → rigid에서 **유리할 수 있으나(효용),
    현행 config는 max를 채택**. metal_case가 SEPARATED(FPR 4%→1%)를 달성한 것은 layer(L7) 효과.
- **추출 실패 경로(false-negative 주의)**: crop이 비거나 grid에 객체 patch가 없으면 score()는
  `DefectResult(score=0.0, is_defect=False)`를 **silent 반환** — 실패가 "결함 아님"으로 처리됨.
- **input_size/grid 출처**: 224·16×16·256 token은 뱅크 메타(`input_size`)에서 우선 결정
  (`grid=input_size//14`)되므로 뱅크 의존(코드 상수 단정 불가).
- **heatmap**: 각 patch nn_dist를 16×16 grid에 흩뿌린 뒤 (W,H) bilinear 확대 → 결함 위치 시각화.
- **layer 가드(2026-06 추가)**: bank에 저장된 `dinov2_layer`가 config `dinov2_layer`와
  다르면 즉시 `ValueError`. 테스트 patch(L_config)와 bank(L_bank)의 layer 불일치 시
  거리가 무의미해지는 것을 **조용히 통과시키지 않고 시끄럽게 차단**.
  **단 가드는 뱅크에 `dinov2_layer` 메타가 있을 때만 작동** — legacy(메타 없는) 뱅크는
  검증을 건너뛰어(`bank_layer=None`) 다른 layer로 빌드돼도 silent 통과한다.

**입력 파라미터**: `memory_bank`(.npz), `threshold`, `dinov2_layer`(기본 8),
`patch_aggregate`, `top_k`/`top_p`, `crop_padding_ratio`(0.03), `crop_background`(127),
`model_name`("facebook/dinov2-base"), `input_size`(224).

### 4.2 Cascade — `src/defect/cascade.py` (haribo, deformable)

**문제**: deformable 객체(젤리/하리보)는 **정상 구겨짐**과 **찢어짐**이 모두 표면 변형을
만든다. PatchCore 단독은 정상 구겨짐을 결함으로 오인(false reject)하기 쉽다.

**해법**: 4개 독립 신호의 **다수결 voting (≥2 of 4; `voting_min=2`)**. (코드 주석은 "AND gate"로
적혀 있으나 실제 부울 의미는 **AND가 아니라 임계 충족 신호 수 ≥2의 다수결**이다 — "AND-gate"는
전 신호 동시 충족을 뜻하므로 잘못된 표현.) 정상 구겨짐은 PatchCore 1표에 그치지만,
찢어짐은 색/기하/형상 신호가 동시에 떠 2표 이상 → 결함.

아래 임계는 모두 **`configs/defect_detection.yaml`의 haribo 블록에서 런타임 주입**(cascade.py 코드
상수 아님). 각 신호는 **2단 임계**(픽셀 단계 + vote 단계)를 가진다:

| # | 신호 | 픽셀 단계 임계 | vote 임계(haribo, config) |
|---|---|---|---|
| s1 | PatchCore L2 | — (§4.1 score) | `patchcore.threshold` 8.6 |
| s2 | Color outlier ratio | Mahalanobis > `z_threshold`(3.0) | `outlier_ratio_threshold` 0.10 |
| s3 | Normal Laplacian peak ratio | \|∇²n_z\| > `peak_threshold`(0.30) | `peak_ratio_threshold` 0.05 |
| s4 | Mask topology raggedness | — `P²/(4π·A)` | `raggedness_threshold` 1.8 |

```
votes = 𝟙[s1>8.6] + 𝟙[s2>0.10] + 𝟙[s3>0.05] + 𝟙[s4>1.8]
is_defect = votes ≥ voting_min(2)
score = max(s1, s2, s3, s4)   # 보고/로깅용 스칼라 — 판정엔 미사용
```

- s2: BGR→LAB 변환 후 mask 픽셀만 ColorPrior(§5.1)에 통과.
- s3: normal map z채널 라플라시안 절댓값이 peak 넘는 픽셀 비율(기하 불연속 = 찢김 경계).
  **normal_map이 None/ndim≠3/shape[2]<3이면 s3=0.0** → normal 미공급 시 사실상 3신호 동작.
- s4: 둘레²/(4π·면적) — 원이면 1, 너덜너덜할수록 큼. **findContours 실패 시 s4=1.0**(정상 원형값).
  현재 임계 1.8이라 안전측이나 임계<1.0 설정 시 빈 contour가 항상 1표 되는 함정.
- **`score=max(s1..s4)`는 단위가 다른 4값의 max라 결함 강도 척도가 아니며 is_defect 판정에
  쓰이지 않는다**(판정은 전적으로 `votes`). 로깅·디버그용.
- **voting의 의의**: 단일 신호 임계가 아니라 **다수결**이라 개별 신호의 false 한 건이
  최종 판정을 뒤집지 못함(강건성).

### 4.3 Weighted Fusion — `src/defect/fusion.py` (mango)

**근거(2026-06-12 근본원인 분석)**: mango는 bulk 정상 CV 0.126으로 rigid급으로 타이트해
PatchCore가 주신호로 충분하나, 결함 모드가 **찢김(tear)**이라 기하 신호(normal 불연속)를
**보조로 가중 융합**한다. cascade를 안 쓴 이유: 임계 4개 튜닝 부담 + 다색 인쇄라 color 기여 낮음.

```
z_pc  = (pc_score − pc_mu) / pc_sigma          (μ=7.95, σ=1.18 — max aggregate 기준 정상 LOO 통계)
z_lap = (lap_ratio − lap_mu) / lap_sigma       (μ=0, σ=1 — 미실측 placeholder)
fused = (1 − w)·z_pc + w·z_lap                 (w = lap_weight = 0.25,  PatchCore 가중 = 1−w = 0.75)
is_defect = fused > threshold(1.8)
```

- `pc_mu/pc_sigma`는 **`patch_aggregate=max` 조건** 산출값(출처 `recovery_eval_mango.json`,
  bank v2 309 obj). §4.1의 rigid 효용 논의와 혼동 금지.
- `lap_ratio`: mask를 **7px erosion 커널**로 침식 후(윤곽 정상 불연속 배제),
  \|Laplacian(n_z)\| > `lap_peak_threshold`(0.30 — mango config; **코드 `__init__` 기본은 0.15**)
  픽셀 비율. erosion 후 유효 픽셀 < 50이면 None.
- **단위 불일치 주의**: 현 config `lap_mu=0, lap_sigma=1`이면 `z_lap = lap_ratio`(0~1 raw 비율)
  그대로라 `fused = 0.75·z_pc + 0.25·lap_ratio`가 되어 **실효 lap 기여가 매우 작다**(z_pc는
  보통 수 단위). fusion 효과 정량 해석 시 중요.
- **PatchCore 단독 fallback 조건**: normal 없음뿐 아니라 ndim≠3/shape[2]<3, erosion 후
  유효픽셀<50에서도 `z_lap` 비활성 → `fused = z_pc`(재정규화 안 함, 보수적). 작은 객체/좁은
  mask에서 조용히 z_lap이 꺼진다.
- 내부 `PatchCoreLite`는 `threshold=0.0`으로 생성돼 내부 판정 비활성, **융합이 최종 is_defect
  담당**. PatchCore raw 거리 척도와 fusion z-score 임계(1.8)는 **서로 다른 척도**.
- `components` 센티넬: `lap_ratio` 없음=`-1.0`, `z_lap` 없음=`-999.0`(로그 사후분석 시 실측값 오해 방지).
- `patch_aggregate = max` (mango는 top_k가 AUROC 0.880→0.816으로 하락 — 소형 tear 희석).

> **정직한 제약(코드 주석에 명시)**: 합성 결함은 **RGB만** 바꾸고 normal map은 변경하지 않으므로
> (Zivid PLY 조작 필요), **Laplacian 채널의 TPR 기여는 오프라인 검증 불가**. `w=0.25`는
> "실제 찢김은 기하 신호를 남긴다"는 사전 지식이며 **실물 찢김 테스트로만 검증 가능**.
> FPR 쪽(`lap_mu/sigma`)은 정상 데이터로 캘리브 가능하나 현재 placeholder(0/1).
> → config에서 `enabled: false`로 막혀 있음(실물 검증 후 전환).

---

## 5. 보조 컴포넌트

### 5.1 ColorPrior — `src/defect/color_prior.py`

정상 객체 mask 픽셀의 **LAB 색 분포를 GMM(3 component, full covariance)**으로 적합.
픽셀별 **Mahalanobis 거리**(가장 가까운 component까지의 표준 Mahalanobis 거리 중 **최솟값**;
**GMM `weights_`는 거리 계산에 미사용** — min 기반):

```
d(x) = min_k √((x−μ_k)ᵀ Σ_k⁻¹ (x−μ_k))
outlier_ratio(z) = mean( 𝟙[d(x) > z] )      # z = cascade가 config에서 주입 (haribo=3.0)
```

- `z`는 **cascade가 `signals.color.z_threshold`(haribo=3.0)를 주입**한다. ColorPrior 함수의
  기본 z=3.0은 cascade 경로에선 미적용(우연히 같은 값).
- cascade 신호 s2에서만 사용. layer/DINOv2와 **무관**(원시 LAB 픽셀 기반).
- 별도 `.npz`로 저장(`color_gmm_haribo.npz`). **layer 변경 시 재빌드 불필요**.

### 5.2 합성 결함 생성기 — `src/defect/synthesize.py`

임계 ROC 튜닝을 위한 결함 시뮬레이터. 정상 crop에 인공 결함을 덮어 "결함 분포"를 만든다.

| 함수 | 대상 | 방법 |
|---|---|---|
| `synthesize_scratch` | rigid 표면 | mask 내 random dark line(색 0~39, 두께 2~6 택1, 길이 40~179, 3~7개), sharp edge |
| `synthesize_tear` | deformable | 경계 밴드(7×7 erode 커널 → 약 3px 폭)에서 random-walk blob → 속살색(BGR 30,60,90)+noise로 덮음 |

> `rng.integers` 상한은 배타적이라 실제 범위는 색 0~39, 길이 40~179, 선 3~7개. 두께는
> (2,3,4,5,6) 중 택1. 경계 밴드 폭은 커널 크기(7)가 아니라 반경 ≈(7−1)/2 = 3px.

- `synthesize_tear`의 `area_mode`:
  - `"absolute"`: [400,1500) 범위에서 **매 호출 무작위 샘플**(실제 400~1499).
  - `"mask_ratio"`(3~7%): mask 면적 비율로 샘플 → 객체 크기가 달라도 224 resize 후
    tear 점유율 일정(**mango 스케일 희석 문제 해결**). (코드 주석 근거: "비율 3~7%는 haribo
    절대픽셀 ~4.5% 기준". 구체 배수(2.1배/4.2배)는 코드 근거 없어 표기하지 않음.)
- **한계**: RGB 도메인만 합성 → normal/depth 채널 미변경(§4.3 제약의 근원).

### 5.3 DefectLogger — `src/defect/logging.py`

운영 false rate 사후 분석용 누적 로거.
- 호출당 **jsonl 1줄**: timestamp, source_id, class index/name, score, is_defect, votes, components, heatmap 경로.
- `is_defect=True` & `score_map` 존재 시에만 **viridis heatmap PNG** 저장(디스크 절약).
- matplotlib `Agg` backend 강제(헤드리스 서버 GUI 의존 차단).

### 5.4 DefectRegistry — `src/defect/registry.py`

yaml(`configs/defect_detection.yaml`)을 읽어 **class_index → detector** 매핑 구성.
- `type` 필드로 빌드 분기: `patchcore`→PatchCoreLite, `cascade`→CascadeDetector, `fusion`→WeightedFusionDetector.
- **2단 enabled 게이트**: (a) yaml 최상위 `enabled: false`면 detector를 아예 빌드 안 하고
  `get()`도 항상 None(**전체 kill-switch**); (b) per-class `enabled: false`(mango)는 해당
  클래스만 스킵.
- **실패 모드**: 알 수 없는 `type`은 `_build_detector`가 `ValueError`로 **hard-fail**(yaml 오타
  시 레지스트리 생성 자체가 죽음). 단 빈/누락 config는 `or {}`로 graceful 빈 레지스트리.
  config 파일 부재는 `FileNotFoundError`.
- `get(class_index)` → detector 또는 None(미등록/비활성).

---

## 6. 클래스별 구성 & 운영점

class_index 매핑(2026-06-10 확정): 0=haribo, 1=mango, 2=pencil_case, 3=metal_case, 4=bottle, 5=unknown.

> **class_id 이중 의미 주의(재빌드 시 치명)**: 위 `class_index`(=classifier 출력 인덱스)와
> **polygon class_id**(=라벨링 raw label, memory_bank 파일명 alphabet순)는 **다르다**.
> `build_patchcore_refs.py --class_id`는 **polygon class_id**(haribo=1, metal_case=4 등)를 받는다.
> 재빌드 시 classifier index로 오해해 넣으면 엉뚱한 클래스 뱅크가 만들어진다.

| id | class | detector | layer¹ | threshold¹ | enabled | AUROC | 운영점 |
|---|---|---|---|---|---|---|---|
| 0 | haribo | cascade(≥2/4) | L5 | pc 8.6 | ✅ | 0.929 (best L5 0.942) | TPR 81% / FPR 5% |
| 1 | mango | fusion(w=0.25) | L8 | fused 1.8 | ❌ 검증대기 | 0.704→0.880² | TPR 70% / FPR 5% |
| 2 | pencil_case | patchcore(max) | L5 | 9.2 | ✅ | 0.986 | TPR 96% / FPR 5% |
| 3 | metal_case | patchcore(max) | L7 | 6.0 | ✅ | 0.998 | TPR 100% / FPR 4%³ |
| 4 | bottle | — (제외) | — | — | — | — | 투명 plastic → 빛 투과로 mask 내 타 객체 patch 혼입 |
| 5 | unknown | — (우회) | — | — | — | — | classifier가 거름 |

> ² mango 0.704→0.880은 **fusion AUROC가 아니라 PatchCore(max) 단독값**이다. 개선은 라벨
> 오염 1건 제거 + bank v2 + 비율 tear 덕분이며, **Laplacian 융합 채널의 TPR 기여는 합성으로
> 검증 불가**(§4.3). 즉 "fusion이 0.88을 만들었다"는 잘못된 해석.
> ³ metal TPR100%/FPR4%는 **L8@τ=8.93 측정값**이다. config가 채택한 L7(SEPARATED)은 FPR≈1%
> (별도 측정). 즉 운영점 수치와 채택 layer가 어긋나 있으므로 재빌드 후 L7 기준 재측정 필요.

> ¹ **layer/threshold staged 상태**: layer 값(haribo/pencil L5, metal L7)은 layer ablation
> 클래스별 best를 config에 반영했으나, 운영 memory bank는 아직 L8이다. 해당 layer로
> **재빌드 + threshold 재튜닝 전까지** PatchCoreLite 가드가 로드 시 에러로 멈춘다(§4.1).
> threshold 값들은 재튜닝 대기 placeholder(metal 6.0은 L7 참고 τ=5.76 보수 절상).

---

## 7. 오프라인 캘리브레이션

### 7.1 Memory bank 빌드 — `scripts/build_patchcore_refs.py`

라벨링 polygon 데이터에서 클래스별 정상 객체를 crop → DINOv2 `--layer` patch token →
mask 영역 patch만 누적 → `.npz` 저장. 저장 메타에 `dinov2_layer`/`input_size`/`class_id`/
`patches_per_object` 포함. `--exclude`로 오염 라벨 줄단위 제외(mango v2: 1건 제외, 309 obj).

### 7.2 임계값 튜닝 — `scripts/synth_roc_tune_loo.py` (LOO ROC)

self-match 함정을 피한 proper validation:
- **정상 LOO score**: 객체 i의 patch를 bank에서 제외하고 나머지에 대한 nn_dist max.
  클래스당 **max_obj=80 무작위 subsample**(seed=42, replace=False) — 80 초과 클래스의 수치는
  부분표본값(전수 아님).
- **합성 결함 score**: 객체당 N(=5) 변형(scratch/tear)을 같은 LOO 조건으로 → 정상:결함 ≈ **1:5
  불균형**(아래 균형지표는 비율에 민감).
- 두 분포로 ROC → 최적 threshold. **이 지표는 `argmax 2·TPR·(1−FPR)/(TPR+(1−FPR))`로,
  precision 기반 표준 F1이 아니라 TPR·specificity의 조화평균(균형정확도 계열)**이다. ROC만으로는
  precision을 못 구하므로 "F1-best"는 오해를 부르는 표기 — 정확히는 *balanced TPR/specificity τ*.
- **layer-aware**(2026-06 수정): 각 bank의 `dinov2_layer` 메타를 읽어 테스트 patch를
  **같은 layer**에서 추출. (이전 L8 하드코딩 → L5/L7 재빌드 시 틀린 τ 산출하던 버그 수정.)
- **LOO 인덱스 정합 주의**: 튜너는 build 스크립트의 객체 순서(set정렬→txt정렬→줄순서)를 재현해
  self-match를 제외하는데, **build의 `--exclude`는 LOO collect 경로에 미반영**이라 mango v2
  (1건 제외, 309 obj)에서 인덱스 경계가 어긋날 위험이 있다.

### 7.3 Layer ablation — `scripts/layer_ablation_{build,eval}.py`

`output_hidden_states=True` forward 1회로 L1~L11 동시 추출 → layer×class AUROC 비교.
**base(dinov2-base, 12 block) 결과**:

| class | best layer | best AUROC | 현행 L8 | 출처 |
|---|---|---|---|---|
| metal_case | **L7** | 0.999 (SEPARATED) | 0.998 | `LAYER_ABLATION_REPORT.md` |
| pencil_case | **L5** | 0.996 | 0.986 | `LAYER_ABLATION_REPORT.md` |
| haribo | **L5** | 0.942 | 0.929 | `LAYER_ABLATION_REPORT.md` |
| mango | **L8**(단일) / L4+L8+L11(0.888) | 0.870 | 0.870 | `MANGO_QUANT_EVAL.md`(별도 tear 벤치) |

> **출처 분리(중요)**: `LAYER_ABLATION_REPORT.md`의 ablation은 **metal/pencil/haribo 3클래스만**
> 다룬다. mango의 L8 "best" 근거는 별도 산출물 `MANGO_QUANT_EVAL.md`(다른 tear 벤치마크)이며,
> 한 ablation 실험의 4번째 행이 아니다.

전체 패턴: 얕은 L1~3 낮음 → **mid L5~8 sweet spot** → L10 dip(semantic transition) → L11 회복.
L0(pre-block)·L12(classifier semantic)는 제외.

> **large 전환 시(dinov2-large, 24 block) 이론적 예측**: 최적 layer는 절대 위치가 아니라
> **상대 깊이**로 옮겨진다고 보면(×2): haribo/pencil ~L10, metal ~L14, mango ~L16,
> sweet spot 밴드 **L10~L16**. 단 이는 prior일 뿐 데이터 재-ablation으로 확정해야 함
> (base는 large에서 distill된 모델이라 1:1 대응 보장 없음).

---

## 8. Action 의미론 & 파지 우선순위 피드백

> **현재 미배선 (정직)**: `pick_n_place.run()`의 정렬 key·`priority_scorer` 어디에도
> `action`/defect를 소비하는 코드가 **없다**. "grasp priority 감점" 같은 메커니즘은 부재하며,
> 인스턴스에 실제로 부착되는 피드백은 `priority_score`의 13개 `grasp_*` 속성뿐이다.
> 아래 표는 **설계 의도(가정법)**이다.

detector의 `action` 속성에 따라 파이프라인이 반응하도록 **설계될 예정**:

| action | 의미 | 파이프라인 효과(설계) |
|---|---|---|
| `reject` | 결함 → 파지 대상 제외 | 정렬에서 후순위/배제 |
| `priority_down` | 의심 → 후순위 | grasp priority 감점 |
| `report_only` | 기록만 | 판정엔 영향 없이 로깅 |

현재 모든 활성 클래스의 `action`은 `reject`(클래스 속성)이나, 이를 소비하는 통합 코드는 미작성.
통합 시 ⑥ 정렬 key에 결함 플래그를 추가하는 형태가 자연스럽다.

---

## 9. 알려진 한계 / 정직성 노트

1. **Laplacian 채널 TPR 미검증** — 합성이 RGB만 바꿔 fusion의 기하 신호 효과는 실물로만 검증 가능. → mango `enabled:false`.
2. **mango 운영 비활성** — 실물 찢김 테스트(1~2봉지 투입) 후 전환 예정.
3. **detector 미배선** — `pick_n_place.py`에 통합 코드 없음(설계 단계).
4. **layer staged** — config는 클래스별 best layer이나 bank는 L8. 재빌드+재튜닝 전 가드 차단.
5. **fusion `lap_mu/sigma` placeholder** — FPR 캘리브 미실시(0/1).
6. **합성 결함 도메인 갭** — scratch/tear가 실제 결함 분포를 완전히 대표하지 못함.
7. **소형 결함 스케일 희석** — 큰 객체에서 작은 결함이 224 resize로 희석(crop 448/타일링은 후속).

---

## 부록 A. 기호/표기

| 기호 | 의미 |
|---|---|
| q, m | 테스트 patch / memory bank patch 벡터 |
| nn_dist(q) | q의 bank 내 최근접 L2 거리 |
| z_pc, z_lap | PatchCore·Laplacian 신호의 z-정규화 값 |
| w | fusion Laplacian 가중치(0.25) |
| sᵢ | cascade 신호 i (i=1..4) |
| τ | threshold |
| P, A | 윤곽 둘레, 면적(topology) |
| n_z | 표면 법선 z 성분 |
| L | DINOv2 layer index (PatchCore convention = post-L-th-block = hidden_states[L+1]) |

## 부록 B. 파일 인덱스

| 파일 | 역할 |
|---|---|
| `src/defect/base.py` | DefectDetector ABC, DefectResult |
| `src/defect/patchcore_lite.py` | PatchCore-lite (기반 신호) |
| `src/defect/cascade.py` | 4신호 voting (haribo) |
| `src/defect/fusion.py` | PatchCore+Laplacian 융합 (mango) |
| `src/defect/color_prior.py` | LAB GMM Mahalanobis (cascade s2) |
| `src/defect/synthesize.py` | 합성 결함(scratch/tear) |
| `src/defect/logging.py` | jsonl + heatmap 로거 |
| `src/defect/registry.py` | class_index → detector 라우팅 |
| `src/classifier/dinov2.py` | DINOv2 KNN 분류기(상호작용 상대) |
| `configs/defect_detection.yaml` | per-class detector 등록 |
| `scripts/build_patchcore_refs.py` | memory bank 빌드 |
| `scripts/synth_roc_tune_loo.py` | LOO ROC 임계 튜닝 |
| `scripts/layer_ablation_*.py` | layer 비교분석 |
