# donguk-picking — 파지점(grasp/suction) 로직

동욱이 만든 bin-picking 파지점 알고리즘입니다. 최신 `main` 위에 picking 로직만 얹은 브랜치예요.
(검출/분류는 main의 DINOv2+Mask2Former를 그대로 쓰고, **파지점 계산만 이 로직으로**.)

## 구성 (의존 닫힘)

```
scripts/label_logic/
  picking.py          # 파지점 알고리즘 본체
  suction_score.py    # 흡착 점수 / 평면도·접근박스 검사
  depth_recovery.py   # 3D 깊이 복원 (원기둥 fit 등)
  __init__.py
config/default.yaml   # 임계값 / 상수 (picking.py가 모듈 로드시 읽음)
```

- `picking.py`는 `config/default.yaml`을 `parents[2]/config/default.yaml` 경로로 읽으므로
  위 `scripts/label_logic/` + `config/` 구조를 유지해야 그대로 동작합니다.
- 의존 패키지: numpy, scipy, pyyaml, pillow.

## 비고

- 클래스 스킴은 nc=6 (`config/default.yaml`의 `classes.names`: bottle/haribo/mango/metal_case/object/pencil_case).
- main의 자체 picking(`src/pipeline/suction_pipeline.py` 등)과는 별개. 통합(교체) 여부는 추후 협의.
