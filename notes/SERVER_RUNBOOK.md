# 서버 런북 — 결함탐지 "뱅크+임계만 만들면 작동" (DINOv2-large / L12)

대상: 서버에서 메모리뱅크 + 임계만 만들어 폴더에 넣을 사람.
브랜치: `detector`. ⚠️ main 건드리지 말 것.

## 한 줄

```bash
bash scripts/build_all_defect_l12.sh
```

이거 하나면 `weights/`에 **메모리뱅크 4개 + 색GMM + 임계json**이 생기고,
그 `weights/` 폴더 + 코드만 있으면 **DefectRegistry가 자동으로 작동**한다.
**config(yaml) 수동 편집은 전혀 필요 없다.**

---

## 왜 자동으로 작동하나 (설계)

| 단계 | 산출물 | 어떻게 적용되나 |
|---|---|---|
| 뱅크 빌드 | `weights/patchcore_*.npz` (large·L12, 메타에 dinov2_layer=12) | config의 layer/model과 일치 → 가드 통과 |
| 임계 튜닝 | `weights/defect_thresholds.json` | **registry가 로드 시 자동 덮어씀** (config의 placeholder τ 무시) |
| 색 GMM | `weights/color_gmm_haribo.npz` | haribo cascade의 color 신호. layer 무관(기존 재사용 가능) |

→ 임계는 yaml에 손으로 안 넣어도 된다. json 키 = memory_bank 파일명이라 자동 매칭.

## 산출 파일 목록 (weights/에 있어야 작동)

```
weights/
├─ patchcore_haribo.npz          (cascade)
├─ patchcore_mango_v2.npz        (patchcore, "운에 맡김" — 정량검증 없이 작동)
├─ patchcore_pencil_case.npz     (patchcore)
├─ patchcore_metal_case.npz      (patchcore)
├─ color_gmm_haribo.npz          (haribo color 신호, layer 무관)
└─ defect_thresholds.json        (임계 자동 적용)
```

## 검증

```bash
python -c "from src.defect.registry import DefectRegistry; r=DefectRegistry('configs/defect_detection.yaml'); print(r); print('thresholds:', r.threshold_overrides)"
```
- 에러 없이 `DefectRegistry(enabled=True, classes=[0, 1, 2, 3])`가 뜨면 성공.
- **layer mismatch ValueError**가 뜨면: 뱅크가 L12로 안 빌드된 것 → `--layer 12`로 재빌드.

## 클래스별 polygon class_id (build_patchcore_refs --class_id)

| class | class_id | 비고 |
|---|---|---|
| haribo | 1 | |
| mango(젤리껌) | 2 | 출력은 `--output weights/patchcore_mango_v2.npz` |
| metal_case | 3 | |
| pencil_case | 5 | |

> class_id는 classifier class_index(0~5)와 **다른** polygon raw label이다. 위 값 사용.
> 확신 안 서면 기존 npz로 확인: `python -c "import numpy as np;d=np.load('weights/patchcore_metal_case.npz');print(int(d['class_id']))"`

## 주의 / 한계 (정직)

- **mango(젤리껌)**: fusion(미검증 Laplacian) 대신 plain patchcore로 운영 — TPR 정량검증은 못 했으나 "정상에서 벗어나면 잡는다"는 작동은 함(운에 맡김). 나중에 실물 찢김 데이터+캘리브 여유 생기면 `type: fusion`으로 전환(코드 보존됨).
- **layer 12**는 large(24 block) 클래스별 plateau 교집합으로 택한 안전 공통값. 정밀화하려면 메탈 L14·젤리껌 L16 등 클래스별 sweep(후속).
- **bottle/unknown**은 결함탐지 제외(투명/우회).
- detector → pick_n_place **배선은 별개 작업**(이 런북은 detector 자체 작동까지).
