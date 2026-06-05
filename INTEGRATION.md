# 파지점(병) 로직 통합 가이드

이 브랜치는 teammate `main` 위에 **물병 파지점 로직**을 추가합니다(기존 파일 무수정,
순수 추가). 병(bottle) 인스턴스의 흡착 파지점을 우리 로직으로 계산해 **main과 동일한
suction point 형식**으로 돌려줍니다.

## 추가되는 파일 (teammate/main 대비)
```
config/default.yaml                          # 임계값/게이트/병 파라미터
scripts/label_logic/__init__.py
scripts/label_logic/picking.py               # 파지점 엔진(전 클래스)
scripts/label_logic/suction_score.py         # 흡착 점수/충돌/평면도
scripts/label_logic/depth_recovery.py        # 투명/반사 깊이 복원(원기둥 fit)
scripts/label_logic/main_suction_adapter.py  # → main suction_points 형식 변환
scripts/label_logic/main_bridge.py           # teammate 입력 → 병 파지점 브리지
```

## 의존성
`requirements.txt`에 이미 충족됨: numpy / opencv-python / Pillow / PyYAML. **추가 설치 불필요.**

## 통합 단계 (병합 후 해야 할 것)

### 1) import 경로 — `scripts/`를 sys.path에 추가
`src/pipeline/suction_pipeline.py` 상단(또는 진입점)에서:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
```

### 2) compute() 에서 병일 때 우리 경로로 분기
`SuctionPipeline.compute()`의 인스턴스 루프 안:
```python
from label_logic.main_bridge import bottle_suction_point

for instance in instances:
    label = getattr(instance, "label", None)
    if label is not None and int(label) == 4:            # 4 = bottle (그쪽 매핑)
        point = bottle_suction_point(
            instance.mask, depth_image, normal_image,
            intrinsic, extrinsic, rgb_image=None)        # rgb는 선택(3) 참고
        suction_points.append([point] if point is not None else [])
        continue                                          # 기존 generic/class4 경로 skip
    # ... 기존 경로 ...
```
반환: `[[x,y,z](소수3), [qx,qy,qz,qw](소수6)]` (robot frame) 또는 `None`. main 형식과 동일.

### 3) (선택) rgb 전달로 캡 정확도 향상
`compute()`는 rgb_image를 받지 않습니다. `None`이면 동작하나 병 캡 밝기 매칭/복원
prior가 약해집니다. 정확도를 높이려면 `pick_n_place` → `compute`로 rgb_image를
내려보내 `rgb_image=rgb_image`로 전달하세요.

### 4) 해상도 일치
`bottle_suction_point`는 이미지 크기가 `config/default.yaml`의
`camera.resolution`(기본 1224×1024)과 같아야 합니다. 다르면 명확한 에러를 던지니,
같은 카메라를 쓰거나 config의 `camera.resolution`을 맞추세요.

## 결정 필요 — 기존 `class4_bottle`와의 관계
teammate `suction_pipeline`에는 이미 자체 병 로직(`class4_bottle` 전략 →
`estimate_class4_bottle_surface`)이 있습니다. 위 분기는 병에서 **우리 경로를 쓰고
기존 class4_bottle을 우회**합니다. 둘 중 무엇을 쓸지(또는 비교 후 채택)는 리뷰에서
결정해 주세요.
- 우리 경로 채택: 위 2)처럼 분기 + class4 dispatch 비활성/제거
- 병행/비교: 둘 다 계산해 비교(별도 비교 스크립트 가능)

## 좌표/단위 규약 (브리지 입력)
- depth_image: mm, (H, W)
- normal_image: (H, W, 3) 카메라 좌표 법선
- intrinsic: 3×3 (`pixel_to_camera`와 동일 공식으로 역투영)
- extrinsic: 4×4 카메라→로봇
