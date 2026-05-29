# PickNPlace

robot pick-and-place 추론 서버. gRPC 로 RGB / depth / normal 이미지를 받아
instance segmentation + 3D picking point 계산 결과를 반환한다.

---

## 폴더 구조

```text
PickNPlace/
├── cmes_inference.py           # 서버 엔트리 포인트
├── cmes_client.py              # 테스트용 gRPC 클라이언트
├── generate_example_data.py    # ./img/ 에 더미 입력 데이터 생성
├── inference.opt               # 모델 / 모드 / 옵션 설정 (JSON)
├── requirements.txt            # Python 의존성
├── README.md
│
├── configs/                    # 본인 모델 config 넣는 곳
├── weights/                    # 본인 학습 checkpoint 넣는 곳
│
├── runner/                     # 통신 / 디스패치 (수정 금지)
│   ├── cmes_ai_runner.py       # option_file 읽고 mode 분기 → gRPC 서버 기동
│   └── modes/
│       ├── base_mode.py        # 모드 베이스 클래스. model_list 매칭 + inference_model 생성
│       ├── grpc_mode.py        # gRPC 핸들러. agnostic / is_connected dispatch
│       ├── protos/
│       │   ├── cmes_ai.proto           # gRPC 프로토콜 정의 (Request / Reply)
│       │   ├── cmes_ai_pb2.py          # 자동 생성된 message stub
│       │   └── cmes_ai_pb2_grpc.py     # 자동 생성된 service stub
│       └── utils/
│           ├── etc.py                  # logger / timestamp / exception 헬퍼
│           ├── message_manager.py      # JSON encoder / decoder (image, point cloud 직렬화 포함)
│           └── option_manager.py       # inference.opt 로더
│
└── src/                        # 학생 구현 영역
    ├── inference_factory.py    # model_name → PickNPlace 인스턴스 매핑
    ├── pick_n_place.py         # 추론 파이프라인 (핵심)
    ├── detector/
    │   ├── __init__.py             # InstanceData (객체별 결과 dataclass)
    │   └── example_detector.py     # 2D detector 인터페이스 예시
    ├── pipeline/
    │   ├── __init__.py
    │   └── example_pipeline.py     # 3D pipeline / suction 계산 인터페이스 예시
    └── utils/
        └── __init__.py
```

---

## PickNPlace 외부 API

`grpc_mode` 가 [`PickNPlace`](src/pick_n_place.py) 인스턴스에 대해 호출하는 메서드는 4개다.

| # | 메서드 | 호출 시점 | 역할 |
|---|---|---|---|
| 1 | `__init__(logger, config, weight, options, cuda)` | 서버 기동 시 1회 | 모델 / 파라미터 로드 |
| 2 | `set_intrinsic(cx, cy, fx, fy)` | request 에 `intrinsic` 키 있을 때 | 카메라 intrinsic 갱신 |
| 3 | `run(rgb, depth, normal, ...)` | 매 request | 추론 메인 — `(result, predictions)` 반환 |
| 4 | `save_result(rgb, predictions, polygons, ...)` | `save_result` 옵션 켜졌을 때 | 시각화 PNG 저장 |

---

## 통신 흐름

```
cmes_client.py
    │  JsonEncoder.serialize() → bytes
    ▼
cmes_ai_pb2.Request(request_function_name="agnostic", request_data=<bytes>)
    │  gRPC over :50051
    ▼
grpc_mode.Excute
    │  getattr(self, "agnostic")(request_data)
    ▼
grpc_mode.agnostic
    1) JSON 파싱 → model / unique_id / level
    2) ./img/ 폴더에서 RGB / depth / normal 로드
    3) intrinsic → inference_model.set_intrinsic(...)
       2D ROI 파싱
    4) inference_model.run(rgb, depth, normal, ...)
    5) result dict → JSON bytes
    │
    ▼
cmes_ai_pb2.Reply(reply_function_name="agnostic", reply_data=<bytes>)
```

---

## 요청 JSON 필드

| 키 | 타입 | 의미 | 기본값 |
|---|---|---|---|
| `model` | str | 모델 이름 (로깅용) | — |
| `level` | str | 로그 레벨 ("DEBUG" 등) | "INFO" |
| `intrinsic` | "cx,cy,fx,fy" | 카메라 intrinsic | 미적용 |
| `roi` | "xmin,xmax,ymin,ymax" | 2D ROI | 전체 이미지 |
| `save_result` | 0/1 | 결과 PNG 저장 여부 | 0 |
| `vis_pcd` | 0/1 | 3D point cloud 시각화 | 0 |
| `save_ply` | 0/1 | PLY 파일 저장 | 0 |

---

## 응답 JSON 형식

```json
{
    "result_data":    [[[150,120],[300,120],[300,280],[150,280]], ...],
    "state":          1,
    "class_id":       [0, 0],
    "pts_per_object": [1, 1],
    "suction_points": [
        [[[-44.899, -72.088, 1192.901], [0.597, 0.801, -0.006, -0.052]]],
        [[[-91.016, 165.696, 1138.743], [0.212, 0.969, -0.124, 0.036]]]
    ],
    "2d_roi":         [0, 640, 0, 480],
    "3d_roi":         []
}
```

### 필드 설명

| 키 | 타입 | 설명 |
|---|---|---|
| `result_data` | `List[Polygon]` | 객체별 polygon 좌표 `[N_pts, 2(x,y)]` |
| `state` | int | 1=정상, -1=객체있으나 suction 실패, -2=객체 미검출, 0=내부 예외 |
| `class_id` | `List[int]` | 객체별 분류 id |
| `pts_per_object` | `List[int]` | 객체별 suction 후보 개수 |
| `suction_points` | `List[List[((x,y,z),(qx,qy,qz,qw))]]` | **핵심 출력** — 로봇 베이스 좌표 + 자세 |
| `2d_roi` | `List[int]` | 사용된 ROI |
| `3d_roi` | `List[float]` | 항상 `[]` |

### `suction_points` 상세 구조

```
suction_points[i][j] = [[x, y, z], [qx, qy, qz, qw]]
                  │  │         │              │
                  │  │         │              └─ 자세 (quaternion)
                  │  │         └─ 로봇 베이스 좌표 (mm)
                  │  └─ j번째 suction 후보
                  └─ i번째 객체
```

### 길이 관계 (state == 1 기준)

```
len(result_data) == len(class_id) == len(pts_per_object) == len(suction_points) == N_obj
pts_per_object[i] == len(suction_points[i])
```

---

## 입력 파일 약속

테스트 client 가 호출하면 grpc_mode 는 다음 파일을 읽는다 (file IO 모드):

| 파일 | 형식 | shape / dtype |
|---|---|---|
| `./img/rgb.png` | BGR 이미지 | (H, W, 3), uint8 |
| `./img/depth.png` | 16비트 depth | (H, W), uint16, 단위 mm |
| `./img/normal.bin` | 헤더 + float32 raw | header: `int32 height, width, channels`; 이후 `H * W * 3` float32 |

---

## 설치

```powershell
conda create -n picknplace python=3.9 -y
conda activate picknplace
pip install -r requirements.txt
```

[requirements.txt](requirements.txt) 는 networking 패키지(grpc / numpy / opencv / open3d / matplotlib / pytz)만 포함한다.
AI 관련 패키지(torch, mmcv, mmdet, ultralytics 등)는 본인 구현에 맞춰 직접 추가한다.

---

## 빠른 시작 (Quick Start)

### 1) 예시 데이터 생성

```powershell
python generate_example_data.py
```

`./img/` 폴더에 더미 입력 파일이 생성된다:
- `rgb.png` — BGR uint8, 640x480, 사각형 객체 2개
- `depth.png` — uint16 mm, 배경 800mm / 객체 500~600mm
- `normal.bin` — float32, 모든 픽셀 z-up (0,0,1)

### 2) 서버 실행

```powershell
python cmes_inference.py
```

더미 모드(detector=None)로 기동되며, gRPC :50051 에서 대기한다.

### 3) 클라이언트로 테스트

별도 터미널에서:

```powershell
# gRPC 통신 확인
python -c "from cmes_client import run; run('is_connected')"

# 추론 파이프라인 전체 확인
python cmes_client.py
```

### 4) 기대 출력 (더미 모드)

```text
reply from: agnostic
state: 1
  result_data    : [[[150, 120], [300, 120], [300, 280], [150, 280]], [[350, 200], [520, 200], [520, 380], [350, 380]]]
  class_id       : [0, 0]
  pts_per_object : [1, 1]
  suction_points : [[[[-44.899, -72.088, 1192.901], [0.597, 0.801, -0.006, -0.052]]], [[[-91.016, 165.696, 1138.743], [0.212, 0.969, -0.124, 0.036]]]]
  2d_roi         : [0, 640, 0, 480]
  3d_roi         : []
```

더미 모드에서도 예시 picking point 가 반환된다.
본인 detector + 3D 변환을 구현하면 실제 결과로 교체된다.

### 5) 트러블슈팅

**`Failed to bind to address [::]:50051`**

이전 서버 프로세스가 포트를 점유 중일 때 발생한다.

```powershell
# 포트 점유 프로세스 확인
netstat -ano | findstr 50051

# PID 로 종료 (예: PID=12345)
taskkill /F /PID 12345
```

---

## 구현 가이드

학생이 수정할 파일:

| 파일 | 할 일 |
|---|---|
| `src/pick_n_place.py` | `__init__`에서 모델+extrinsic 로드, `set_intrinsic` 구현, `run()`에서 추론+3D변환, `save_result()`에서 시각화 |
| `src/detector/example_detector.py` | 2D detection wrapper (선택 — 구조 참고용) |
| `src/pipeline/example_pipeline.py` | 3D suction point 계산 (선택 — 구조 참고용) |
| `configs/` | 본인 모델 config 넣기 |
| `weights/` | 본인 학습 checkpoint 넣기 |
| `inference.opt` | config / weight 경로, OPTIONS 수정 |

### 학생 구현 범위

```
pixel (u,v) + depth
    → 카메라 3D 좌표       (intrinsic 역변환 — set_intrinsic 에서 받은 값)
    → 로봇 베이스 3D 좌표   (extrinsic 곱 — __init__ 에서 로드한 값)
    → suction 자세 계산     (quaternion)
    → suction_points 에 ((x,y,z), (qx,qy,qz,qw)) 로 담아 반환
```

- **intrinsic**: 스튜디오에서 gRPC 로 매 요청마다 전달
- **extrinsic**: 캘리브레이션 결과를 별도로 전달받아 `__init__` 에서 로드

`runner/` 폴더는 수정하지 않는다.

# calibration

cal_x:-881.97
cal_y:72.227
cal_z:838.867
cal_rx:178.454  
cal_ry:-1.685
cal_rz:-90.682

---

## 현재 진행 과정 정리

### 1) 현재 추론 흐름

현재 `PickNPlace` 파이프라인은 다음 순서로 동작한다.

```text
RGB / depth / normal 입력
    → 고정 ROI 기준 Mask2Former instance segmentation
    → 각 mask의 largest connected component만 유지
    → DINOv2 KNN class 추론
    → grasp priority 계산
    → suction point / 자세 계산
    → priority 순서로 결과 정렬
```

ROI는 segmentation 기준으로 고정값을 사용한다.

```text
[left, right, top, bottom] = [150.822, 1186.162, 0.0, 866.0]
```

`src/utils/debug_local.py`도 같은 ROI를 사용하도록 맞춰서, 로컬 디버그 결과와 실제 파이프라인 기준이 어긋나지 않게 했다.

### 2) Class 추론 baseline

DINOv2 classifier는 `configs/dinov2.yaml` 기준으로 동작한다.

- reference bank: `weights/dinov2_reference_bank_crops2/all_reference_bank.pt`
- crop 방식: `inference.md` 기준의 square masked crop
- background: gray `127`
- RGB/gray fusion 사용
- top-k: `3`

현재는 LogisticRegression classifier를 사용하지 않고, DINOv2 KNN 방식만 사용한다.

### 3) Grasp priority baseline

priority는 `configs/grasp_priority.yaml`에서 관리한다.

현재 모드는 `support_then_depth`다.

```text
1차 support score:
    clearance_score + area_score

2차 depth 선택:
    support score 상위 후보군 안에서 depth가 가장 가까운 물체 선택
```

중앙성(`center_weight`)은 현재 0으로 둔다. ROI가 여유분 포함 crop이라 실제 박스 중심과 어긋날 수 있고, 이 경우 특정 방향으로 priority가 쏠리는 bias가 생기기 때문이다.

현재 기준:

- `clearance_score`: 주변 다른 물체와 얼마나 떨어져 있는지
- `area_score`: 절대 면적이 아니라 현재 프레임 내 면적 순위
- `depth`: support 후보군 안에서 최종 선택 기준
- `depth_percentile`: mask 내부 depth 중 가까운 표면을 대표 depth로 보기 위한 percentile

### 4) Suction point baseline

기존에는 `distanceTransform` 최댓값을 suction point로 사용했지만, 긴 직사각형 물체에서는 최댓값 plateau 때문에 한쪽 끝에 point가 잡힐 수 있었다.

현재는 다음 방식으로 바꿨다.

```text
mask foreground 픽셀의 중심 계산
    → 중심이 mask 안이면 그 점 사용
    → 중심이 mask 밖이면 중심에 가장 가까운 mask 내부 픽셀 사용
```

컵 방향은 mask의 PCA 긴 축 방향을 기준으로 잡는다. 그래서 긴 물체에서는 두 흡착컵이 긴 축과 평행하게 들어가고, point는 물체 중심 쪽으로 온다.

### 5) Debug 결과

`python -m src.utils.debug_local` 실행 시 `debug_result`에 다음 정보가 저장된다.

- 전체 segmentation/class/priority 결과
- 최종 grasp 1순위 물체 표시
- suction point 표시
- dual suction cup footprint 표시
- class 추론 상세값
- priority 상세값

흡착컵 규격은 현재 다음 가정으로 시각화한다.

```text
컵 지름: 25 mm
컵 중심 간 거리: 35 mm
전체 길이: 약 60 mm
```

### 6) 코드상 확인된 보완 포인트

현재 baseline에서 바로 큰 구조 문제는 없지만, 다음 부분은 이후 고도화 대상이다.

1. **Footprint 통과 검사**

   지금은 footprint를 디버그에 그리기만 한다. 다음 단계에서는 두 컵 원이 mask 안에 충분히 들어오는지 검사해야 한다.

   예:

   ```text
   cup1 inside ratio >= 0.85
   cup2 inside ratio >= 0.85
   ```

   기준을 만족하지 못하면 해당 물체를 제외하거나 suction point 후보를 다시 골라야 한다.

2. **Suction 후보 여러 개 생성**

   현재 suction point는 mask 중심 1개다. 안정성을 높이려면 중심 주변에서 후보를 여러 개 만들고, footprint 통과율과 depth 안정성을 기준으로 고르는 방식이 필요하다.

3. **Depth valid ratio**

   mask 내부 depth가 너무 적으면 대표 depth를 믿기 어렵다. 다음 기준을 추가하는 것이 좋다.

   ```text
   valid_depth_pixels / mask_pixels
   ```

4. **Shape complexity**

   물체가 온전히 드러난 경우 polygon 점 개수가 적고 contour가 단순한 경향이 있다. 이후에는 `vertex_count`, `solidity`, `convexity`를 이용해 occlusion 가능성이 높은 물체를 뒤로 보내는 보조 점수를 추가할 수 있다.

5. **Class별 예외 처리**

   하리보처럼 작거나 구겨질 수 있는 물체는 일반 suction footprint 기준에서 먼저 걸러보고, 그래도 문제가 남을 때 class별 예외 처리로 분리하는 것이 좋다.

6. **불필요한 classifier 정리**

   현재 운영 기준은 DINOv2 KNN이다. LogisticRegression 관련 코드는 남아 있지만 사용하지 않는 상태이므로, 이후 코드 정리 단계에서 제거하거나 완전히 분리하는 것이 좋다.

### 7) 다음 개발 우선순위

가장 실무적인 다음 순서는 다음과 같다.

```text
1. dual suction cup footprint inside ratio 계산
2. footprint 실패 물체 reject 또는 후보 재탐색
3. depth valid ratio 추가
4. polygon/solidity 기반 shape complexity 추가
5. 실제 실패 케이스 기준으로 class별 예외 처리
```

현재 baseline의 핵심은 “우선순위는 단순하게 유지하고, 실제 흡착 실패를 막는 안전장치를 하나씩 추가한다”는 방향이다.
