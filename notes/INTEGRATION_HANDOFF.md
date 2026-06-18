# 통합 핸드오프 — 결함탐지 모듈을 본체(pick_n_place)에 합치기

대상: 본체 코드 통합 담당 팀원. 브랜치 `detector`. ⚠️ main 직접 변경 금지.

## TL;DR — 팀원이 할 일은 둘뿐

1. **코드 가져다 합치기**: `src/defect/` 모듈을 본체에 포함 + 아래 §2 "배선 4줄"을 `pick_n_place.run()`에 추가.
2. **아침에 파일만 받기**: 서버에서 만든 `weights/patchcore_*.npz`, `weights/color_gmm_haribo.npz`, `weights/defect_thresholds.json`을 `weights/`에 떨구면 끝. **config·코드 수정 불필요**(임계는 json이 자동 적용).

---

## 1. 모듈 공개 인터페이스 (이것만 알면 됨)

```python
from src.defect import DefectRegistry, DefectResult, DefectLogger

registry = DefectRegistry("configs/defect_detection.yaml")   # 1회 로드
det = registry.get(class_index)        # 클래스별 detector 또는 None
if det is not None:
    result: DefectResult = det.score(rgb_bgr, mask, normal_map, dinov2_features=None)
    # result.score (float), result.is_defect (bool), result.action ← det.action,
    # result.score_map (heatmap|None), result.votes, result.components
```

- **입력**: `rgb_bgr`(H,W,3 uint8 BGR=원본 RGB), `mask`(H,W bool/uint8 객체1개), `normal_map`(H,W,3 float32|None), `dinov2_features`(None이면 detector가 자체 forward).
- **출력 `DefectResult`**: `score`만 필수, 나머지 기본값 있음.
- `action`은 detector 인스턴스 속성(`det.action`): `"reject"|"priority_down"|"report_only"` (현재 전부 `reject`).
- `registry.get()`가 None → 그 클래스는 결함검사 대상 아님(물병=제외, 언노운=우회, 비활성 클래스).

## 2. 배선 — `pick_n_place.run()`에 추가 (분류 직후)

`PickNPlace.__init__`에 1회 로드:
```python
from src.defect import DefectRegistry, DefectLogger
self.defect_registry = DefectRegistry("configs/defect_detection.yaml")
self.defect_logger = DefectLogger("log/defect")   # 선택(jsonl+heatmap 누적)
```

`run()`에서 classifier가 `class_index`를 채운 직후(현재 `classify_instances` 블록 다음), 인스턴스 루프에 추가:
```python
for instance in kept_predictions:
    det = self.defect_registry.get(getattr(instance, "class_index", -1))
    if det is None:
        instance.is_defect = False
        continue
    res = det.score(rgb_image, instance.mask, normal_image, dinov2_features=None)
    instance.defect_score = res.score
    instance.is_defect = res.is_defect
    instance.defect_action = det.action
    self.defect_logger.log(str(instance.label), instance.class_index,
                           getattr(instance, "class_name", ""), res)
```

피드백(선택, 설계 의도): 정렬/우선순위에서 `is_defect and action=="reject"`인 객체를 후순위/배제.
예) 정렬 key 맨 앞에 `not getattr(item[3], "is_defect", False)` 추가.

> 주의: 위 배선은 **현재 미배선 상태를 채우는 예시**다. detector 모듈 자체와 오프라인 검증은 완료됐고, 본체 루프 연결만 팀원 작업이다.

## 3. 아침에 받는 파일 → 어디에

서버 산출물(`bash scripts/build_all_defect_l12.sh` 결과)을 `weights/`에 그대로:

```
weights/
├─ patchcore_haribo.npz        patchcore_mango_v2.npz
├─ patchcore_pencil_case.npz   patchcore_metal_case.npz
├─ color_gmm_haribo.npz        ← haribo color 신호(layer 무관)
└─ defect_thresholds.json      ← 임계 자동 적용(registry가 로드)
```

검증: `python -c "from src.defect.registry import DefectRegistry; print(DefectRegistry('configs/defect_detection.yaml'))"`
→ 에러 없이 `classes=[0, 1, 2, 3]`이면 준비 완료.

## 4. 현재 운영 사양 (요약)

| class(idx) | detector | layer | model | 비고 |
|---|---|---|---|---|
| 하리보(0) | cascade(≥2/4) | 12 | dinov2-large | color GMM 필요 |
| 젤리껌/mango(1) | patchcore | 12 | dinov2-large | "운에 맡김"(정량검증X, 작동O) |
| 필통(2) | patchcore | 12 | dinov2-large | |
| 메탈케이스(3) | patchcore | 12 | dinov2-large | |
| 물병(4)·언노운(5) | — | — | — | 제외/우회 |

- 임계: `defect_thresholds.json`이 config placeholder를 자동 덮어씀(키=memory_bank 파일명).
- backbone: DINOv2-large(24 block), layer 12 통일.
- 가드: 뱅크 layer≠config layer면 로드 시 ValueError(시끄럽게 차단 — 조용한 오판 방지).

## 5. 의존성

`requirements.txt` + `transformers`(DINOv2-large 다운로드), `torch`(cu118+), `scikit-learn`(color GMM/ROC), `opencv-python`, `matplotlib`(로거 heatmap, Agg).
