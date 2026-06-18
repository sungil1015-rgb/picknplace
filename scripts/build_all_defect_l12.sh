#!/usr/bin/env bash
# 결함탐지 운영 파일 일괄 생성 — DINOv2-large / layer 12.
# 서버에서 이 스크립트 하나만 돌리면 weights/에 [메모리뱅크 + 색GMM + 임계json]이 생성되고,
# 그 weights/ 폴더 + 코드만 있으면 DefectRegistry가 자동으로 작동한다.
#
# 사전조건: data/labeled/ (정상 라벨 데이터), GPU(권장), transformers/torch/sklearn.
# 산출물: weights/patchcore_{haribo,metal_case,pencil_case}.npz, weights/patchcore_mango_v2.npz,
#         weights/color_gmm_haribo.npz, weights/defect_thresholds.json
#
# 사용: bash scripts/build_all_defect_l12.sh
set -euo pipefail

cd "$(dirname "$0")/.."   # repo 루트

MODEL="facebook/dinov2-large"
LAYER=12

echo "==== [1/3] 메모리뱅크 빌드 (large / L${LAYER}) ===="
# polygon class_id: haribo=1, mango=2, metal_case=3, pencil_case=5 (라벨 data.yaml 기준)
python scripts/build_patchcore_refs.py --class_id 1 --class_name haribo      --model "$MODEL" --layer "$LAYER"
python scripts/build_patchcore_refs.py --class_id 3 --class_name metal_case  --model "$MODEL" --layer "$LAYER"
python scripts/build_patchcore_refs.py --class_id 5 --class_name pencil_case --model "$MODEL" --layer "$LAYER"
# mango(젤리껌) v2 — config memory_bank가 patchcore_mango_v2.npz 이므로 --output 지정.
#   오염 제외 목록이 있으면 아래에 --exclude outputs/eda/mango_exclude_list.txt 추가(없어도 동작).
MANGO_EXCLUDE=""
[ -f outputs/eda/mango_exclude_list.txt ] && MANGO_EXCLUDE="--exclude outputs/eda/mango_exclude_list.txt"
python scripts/build_patchcore_refs.py --class_id 2 --class_name mango \
    --output weights/patchcore_mango_v2.npz --model "$MODEL" --layer "$LAYER" $MANGO_EXCLUDE

echo "==== [2/3] 색상 GMM (haribo cascade s2, layer 무관) ===="
# 기존 weights/color_gmm_haribo.npz가 있으면 재사용해도 됨(이 단계 생략 가능).
if [ ! -f weights/color_gmm_haribo.npz ]; then
    python scripts/build_color_prior.py --class_id 1 --class_name haribo
else
    echo "  weights/color_gmm_haribo.npz 이미 존재 → 재사용(layer 무관)"
fi

echo "==== [3/3] 임계 튜닝 → weights/defect_thresholds.json (registry 자동 적용) ===="
python scripts/synth_roc_tune_loo.py   # 실패/스킵 클래스 있으면 exit 1 (set -e로 여기서 멈춤)

echo "==== [검증] thresholds.json 4키 + registry 로드 ===="
python - <<'PYCHK'
import json, sys
from pathlib import Path
need = {"patchcore_haribo.npz", "patchcore_mango_v2.npz",
        "patchcore_metal_case.npz", "patchcore_pencil_case.npz"}
p = Path("weights/defect_thresholds.json")
assert p.is_file(), "defect_thresholds.json 없음 — 튜닝 실패"
keys = set(json.loads(p.read_text(encoding="utf-8")).keys())
miss = need - keys
assert not miss, f"thresholds 키 누락: {miss}"
from src.defect.registry import DefectRegistry
r = DefectRegistry("configs/defect_detection.yaml")   # strict — τ 누락/dim/가드 불일치면 여기서 에러
print("OK:", r, "| thresholds:", r.threshold_overrides)
PYCHK

echo
echo "==== 완료 — 시연 준비됨 ===="
echo "weights/: 뱅크4 + color_gmm_haribo + defect_thresholds.json. registry strict 로드 통과."
