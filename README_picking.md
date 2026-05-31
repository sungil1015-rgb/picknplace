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

## 파이프라인 연결 (B — 실제 배선됨)

`pick_n_place.py` 의 파지점 계산을 **팀 SuctionPipeline → 동욱 picking 으로 교체**했습니다.

```
src/pipeline/donguk_picking.py   # 어댑터 (신규)
  depth(mm)+rgb+normal+intrinsic → organized grid(구조화 배열, Zivid 포맷)
  → label_logic.picking.compute_pick → PickResult(camera frame)
  → 팀 geometry 헬퍼로 robot 좌표 [[x,y,z],[qx,qy,qz,qw]] 변환
  PICKABLE 아니면 [] (팀 포맷 동일). 진단값은 instance.pick_status/confidence/tier 로 부착.

src/pick_n_place.py  (수정)
  __init__: self.donguk_picking 로드 + self.use_donguk_picking 플래그
  run():    compute_suction_pts 시 SuctionPipeline 대신 donguk_picking.compute 호출
```

- **토글**: `pick_n_place.yaml` 에 `picking: {use_donguk: false}` 추가하면 팀 SuctionPipeline 으로 폴백 (기본 true). 어댑터 로드 실패 시에도 자동 폴백.
- **검증**: 합성 scene 으로 end-to-end 동작 확인 (PICKABLE → robot 좌표/쿼터니언 정상 생성). **실제 robot depth/normal/extrinsic 데이터 검증은 미완** (다음 단계).

## 비고

- 클래스 스킴은 nc=6 (`config/default.yaml`의 `classes.names`: bottle/haribo/mango/metal_case/object/pencil_case).
- 팀 자체 picking(`src/pipeline/suction_pipeline.py`)은 폴백용으로 그대로 보존 (삭제 안 함).
