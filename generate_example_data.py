"""
예시 입력 데이터 생성 스크립트.

./img/ 폴더에 gRPC 서버가 읽을 수 있는 더미 이미지를 생성한다.
실제 카메라 데이터가 없을 때 파이프라인 동작을 확인하는 용도.

사용법:
    python generate_example_data.py
"""

import os
import numpy as np
import cv2

IMG_DIR = "./img"
HEIGHT, WIDTH = 480, 640

os.makedirs(IMG_DIR, exist_ok=True)


# ── 1) RGB 이미지 (BGR, uint8, shape=(H, W, 3)) ──
# 컬러 그라데이션 + 가운데 사각형 두 개 (가짜 객체)
rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
rgb[:, :, 0] = np.linspace(60, 200, WIDTH, dtype=np.uint8)   # B
rgb[:, :, 1] = np.linspace(40, 180, HEIGHT, dtype=np.uint8)[:, None]  # G
rgb[:, :, 2] = 100  # R

# 객체 A (빨간 사각형)
cv2.rectangle(rgb, (150, 120), (300, 280), (0, 0, 220), -1)
# 객체 B (초록 사각형)
cv2.rectangle(rgb, (350, 200), (520, 380), (0, 200, 0), -1)

cv2.imwrite(os.path.join(IMG_DIR, "rgb.png"), rgb)
print(f"[OK] rgb.png  shape={rgb.shape}  dtype={rgb.dtype}")


# ── 2) Depth 이미지 (16-bit, uint16, shape=(H, W), 단위=mm) ──
# 배경 800mm, 객체A 500mm, 객체B 600mm
depth = np.full((HEIGHT, WIDTH), 800, dtype=np.uint16)
depth[120:280, 150:300] = 500   # 객체 A
depth[200:380, 350:520] = 600   # 객체 B

cv2.imwrite(os.path.join(IMG_DIR, "depth.png"), depth)
print(f"[OK] depth.png  shape={depth.shape}  dtype={depth.dtype}")


# ── 3) Normal map (binary: int32 h,w,c 헤더 + float32 raw) ──
# 모든 픽셀이 위쪽을 가리키는 (0, 0, 1) normal
normal = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
normal[:, :, 2] = 1.0  # z-up

normal_path = os.path.join(IMG_DIR, "normal.bin")
with open(normal_path, "wb") as f:
    np.array([HEIGHT], dtype=np.int32).tofile(f)
    np.array([WIDTH], dtype=np.int32).tofile(f)
    np.array([3], dtype=np.int32).tofile(f)
    normal.tofile(f)
print(f"[OK] normal.bin  shape={normal.shape}  dtype={normal.dtype}")


print(f"\n예시 데이터 생성 완료 → {os.path.abspath(IMG_DIR)}/")
print("  rgb.png   : BGR uint8 (H={}, W={})".format(HEIGHT, WIDTH))
print("  depth.png : uint16 mm  (배경=800mm, 객체A=500mm, 객체B=600mm)")
print("  normal.bin: float32    (모든 픽셀 z-up)")
