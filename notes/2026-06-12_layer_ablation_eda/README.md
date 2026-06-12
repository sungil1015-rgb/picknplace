# 2026-06-12 작업 산출물 — Layer Ablation EDA + 파이프라인 시각화

이 폴더는 2026-06-12 (금)에 한 두 가지 작업의 산출물을 한 곳에 모은 것입니다.

1. **Layer ablation EDA** — PatchCore-lite의 DinoV2 mid-layer 선택을 11 layer 전체 비교 검증
2. **파이프라인 시각화** — Zivid 빈 스캔에서 로봇 명령까지 전체 추론 알고리즘 한 장 다이어그램 (회사 발표용)

처음 보시는 분은 **`1_REPORT.md` 부터 읽으세요** (5분 정독).

---

## 폴더 구조 + 읽는 순서

```
2026-06-12_layer_ablation_eda/
├── README.md                  ← 이 파일 (폴더 안내)
├── 1_REPORT.md                ← ★ 5분 정독 보고서 (TL;DR + 인사이트 + 결론 + 권장)
├── 2_raw_results.md           ← raw 결과 표 (layer × class × AUROC/F1/TPR/FPR)
├── figs/                      ← 모든 시각자료
│   ├── auroc_vs_layer.png         ← ★ 한 장 핵심 요약 (3 클래스 layer × AUROC 곡선)
│   ├── roc_curves_haribo.png      ← 클래스별 11 layer ROC 곡선
│   ├── roc_curves_metal_case.png
│   ├── roc_curves_pencil_case.png
│   ├── score_dist_haribo.png      ← 클래스별 11 layer score 분포 (정상 vs 결함)
│   ├── score_dist_metal_case.png
│   ├── score_dist_pencil_case.png
│   └── pipeline_overview_ko.png   ← ★ 전체 추론 파이프라인 다이어그램 (회사 발표용, 3300×1980)
└── scripts/                   ← 재현용 코드 (스냅샷)
    ├── layer_ablation_build.py    ← 11 layer × 3 class 메모리뱅크 빌드 (forward 1회로 동시 추출)
    ├── layer_ablation_eval.py     ← LOO ROC eval + 시각화 + summary 자동 생성
    └── plot_pipeline_overview.py  ← 파이프라인 다이어그램 생성 (matplotlib)
```

### 권장 읽는 순서

| 누구 | 무엇을 먼저 보세요 |
|---|---|
| **시간 5분, 결론만** | `1_REPORT.md` TL;DR + 핵심 결과 표 + `figs/auroc_vs_layer.png` |
| **결정자 (운영 layer 변경 여부)** | `1_REPORT.md` 전체 → "결론" + "권장 액션" 섹션 |
| **분석 검토** | `1_REPORT.md` "인사이트 4가지" → `figs/score_dist_*.png` 확인 → `2_raw_results.md` |
| **재현 / 다른 layer 추가 비교** | `scripts/` 안 두 스크립트 |
| **회사 발표 자료 필요** | `figs/pipeline_overview_ko.png` (470KB PNG) — 그대로 슬라이드 삽입 |

---

## 핵심 결과 한 줄 (TL;DR)

> 현재 운영 layer 8은 합리적 선택이었으나, **L7이 모든 활성 클래스에서 동등 이상이고 metal_case는 유일하게 SEPARATED (AUROC 0.999) 달성**.
> mid-layer 5~8 sweet spot이 실증 확인됨. layer 변경은 **금요일 로봇 테스트 후 PR로 검토 권장**.

| 클래스 | 현재 L8 | best layer | best AUROC | Δ |
|---|---|---|---|---|
| metal_case | 0.998 OVERLAP | **L7** | **0.999 SEPARATED** | +0.001 |
| haribo | 0.929 | L5 | 0.942 | +0.013 |
| pencil_case | 0.986 | L5 / L11 | 0.996 | +0.010 |

---

## 작업 메타

- 실행 환경: curie 서버, RTX A5000, conda env `picknplace` (Python 3.9.25), torch 2.8 + cu128
- 실행 시간: build 50초 + eval 2분 20초 = **총 3분 10초**
- 효율 비법: `output_hidden_states=True` 한 번 forward에 11 layer hidden_states 동시 추출 → layer당 별도 forward 안 함
- 임시 메모리뱅크: `outputs/eda/memory_banks/mb_L{L}_{class}.npz` × 33 (약 6GB, 이 폴더 외부 — 재실행 시 재생성 가능)
- 작업 일지: `notes/PROJECT_LOG.md` 세션 3 (2026-06-12) 누적 기록

## 원본 위치 (이 폴더는 사본 — 원본 변경 시 sync 필요)

| 사본 | 원본 |
|---|---|
| `1_REPORT.md` | `notes/LAYER_ABLATION_REPORT.md` |
| `2_raw_results.md` | `outputs/eda/layer_ablation_summary.md` |
| `figs/auroc_vs_layer.png` 외 ROC/score_dist | `outputs/figs/layer_ablation/` |
| `figs/pipeline_overview_ko.png` | `outputs/figs/pipeline_overview_ko.png` |
| `scripts/*.py` | `scripts/*.py` (프로젝트 루트) |
