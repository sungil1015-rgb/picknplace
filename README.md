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