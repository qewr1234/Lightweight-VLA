# SmolVLA-FAST DCT

**Jetson Orin Nano 8GB에서 동작하는 경량 Vision-Language-Action 모델.**
KV cache로 260ms의 비전 연산을 제어 경로에서 분리하고, VLM backbone을 16→8 layer로 줄여
decode step을 85ms → 43ms로 단축했습니다.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![base](https://img.shields.io/badge/base-SmolVLA-orange.svg)](https://arxiv.org/abs/2506.01844)
[![hardware](https://img.shields.io/badge/hardware-Jetson%20Orin%20Nano%208GB-76B900.svg)](https://developer.nvidia.com/embedded/jetson-orin)
[![verify](https://github.com/qewr1234/Lightweight-VLA/actions/workflows/verify.yml/badge.svg)](https://github.com/qewr1234/Lightweight-VLA/actions/workflows/verify.yml)

> **DCT 주파수 도메인 Flow Matching과 비전-액션 디커플링을 활용한 경량 VLA 모델의 Edge 디바이스 실시간 배포**
> 이유준, 변정민, 김민엽, 최규상\* · 영남대학교 정보통신공학과 (\*교수)

## 저장소 현황

| | 상태 |
|---|---|
| 경량화 포크 (SmolVLA-Lite) | **있음 · 검증됨** — 매 push마다 CI 통과 |
| 논문 구조 스켈레톤 (`smolvla_fast_dct.py`) | 있음 · **미학습** (재학습 전까지 출력 무의미) |
| 논문의 학습·배포 코드 | **없음** — 비동기 파이프라인·SO-101 I/O, Jetson Docker 환경. 정리 후 업로드 예정 |

즉 이 저장소의 코드는 lerobot 0.5.1 기반 경량화 실험이고, 논문 모델(lerobot 0.4.4, Jetson, SO-101 실기)과는
별개입니다. 포크의 상세 문서는 [`docs/lightweight-fork.md`](docs/lightweight-fork.md).

## 빠른 시작

```bash
# Python >= 3.12 (lerobot 0.5.x 요구사항)
pip install "lerobot[smolvla,async]" "transformers>=5.3,<5.4"

python tests/test_lightweight.py      # 등가성 21체크
python smolvla_fast_dct.py            # 논문 구조 조립 → 275.0M / 50.8M, chunk (1, 10, 6)
python benchmark.py --random --device cpu --iters 3
```

네트워크가 막힌 환경이면 가중치 없는 SmolVLM2 캐시를 먼저 만드세요 (~1.5MB):

```bash
python tests/offline_vlm_cache.py
export HF_HOME=$PWD/.hf_offline HF_HUB_OFFLINE=1
```

`.github/workflows/verify.yml`이 push마다 동일한 순서를 실행합니다. 배포·검증 상세는
[`docs/lightweight-fork.md`](docs/lightweight-fork.md).

## 제안 방법

**1. KV cache 기반 비전-액션 디커플링.** SmolVLA는 prefix(시각+상태 129토큰)와 suffix(action)를
attention mask로 구분합니다. 구조를 바꾸지 않고 prefix의 KV cache를 공유 메모리에 두면, Vision
스레드가 다음 cache를 만드는 동안 Action 스레드가 이전 cache로 계속 생성할 수 있습니다. 비전 처리에
260ms가 걸려 동기식 추론은 약 3.8 FPS에 묶이지만, 이 분리로 그 하한이 사라집니다. KV cache 추가
메모리는 2.1MB입니다.

**2. VLM backbone 축소 (16 → 8 layer).** SmolVLM2-500M의 16 layer 중 8개만 사용하고 2 layer 단위로
self/cross-attention을 교대 배치했습니다. Action Expert에는 width multiplier 0.75를 적용했습니다.
측정된 decode step 감소(85ms → 43ms)는 주로 이 축소에 기인합니다.

**3. DCT 주파수 도메인 flow matching.** 정규화된 action chunk에 type-II DCT(ortho)를 적용한 계수를
flow matching 목표로 씁니다. t ~ Beta(1.5, 1.0)으로 후반 구간에 가중을 두고, 추론은 Euler 적분 후
IDCT로 복원합니다. Loss는 실제 6-DoF에만 적용해 32차원 zero-padding의 영향을 제거했습니다.

> **범위에 대한 명시**: 본 연구의 DCT는 계수 절단 없이 chunk 길이와 같은 수의 계수를 쓰므로 토큰 수와
> 연산량이 시간 도메인과 동일한 **직교 기저 변환**입니다. DCT 자체가 지연을 줄이지는 않습니다.
> 기대 효과는 저주파 지배 분포에서의 수렴 특성과 궤적 평활화이며, 동일 조건(8 layer, chunk 10)에서의
> ablation은 **아직 수행하지 않았습니다** — 관측된 지연 감소를 DCT에 귀속할 수 없습니다.

## 성능

| 항목 | SmolVLA Baseline | SmolVLA-FAST DCT |
|---|---|---|
| 총 / 학습 파라미터 | 450.0M / 99.9M | **322.3M / 50.6M** (-28% / -49%) |
| VLM layers · chunk | 16 · 50 | **8 · 10** |
| 모델 크기 (FP16) | 920 MB | **657 MB** |
| decode step (Jetson, FP16) | 85 ms | **43 ms** |

동기식 추론은 비전 260ms가 병목이라 약 3.8 FPS입니다. 비동기 모드의 최대 처리율은 약 **23 FPS**로,
목표한 30 FPS에는 도달하지 못했습니다. decode step 시간은 제어 주파수가 아닙니다 — chunk 10 기준
queue 절반(약 167ms) 소진 시간이 decode보다 길어, 이 설정에서 decode는 병목이 아닙니다.

**DCT 에너지 집중** (학습 데이터 5,845 chunk): DC 단독 99.47%, 상위 2개 99.96%. 단 chunk 10은 30fps
기준 0.33초라 DC 지배에는 구간 길이의 기여도 있습니다. 잔여 계수의 분산이 매우 작아, 계수별 정규화 시
저에너지 성분이 과대 가중될 수 있습니다.

**궤적 복원**: cosine similarity 0.9966, 예측 범위도 GT와 일치(GT [-53, 95] vs Pred [-53, 98]).
MSE 10.19로 step 단위 세부 움직임에는 오차가 남습니다.

## 실험 설정

SO-101 follower/leader 텔레오퍼레이션으로 수집한 큐브 파지 시연 **80 에피소드**(178,050 프레임,
640×480 RGB ×2 + 6-DoF 관절 위치). batch 128 · lr 1e-4 · 30,000 step · bfloat16 · cosine warmup,
데이터 증강 3배. 배포: JetPack 6.2 / CUDA 12.6 / `dustynv/l4t-pytorch:r36.4.0`.
PyPI 표준 PyTorch는 sm_87 커널이 부족해 NVIDIA 공식 컨테이너가 필요합니다.

## 한계

- **정밀 파지에 도달하지 못했습니다.** baseline과 제안 모델 모두 큐브 접근은 되지만 안정적 파지는
  실패했습니다. 본 연구는 edge 실시간 추론 가능성을 확인한 **초기 검증**입니다.
- 원인 후보 미분리: (i) 80 에피소드의 제한된 데이터, (ii) Vision 갱신 주기가 260ms라 Action 스레드가
  최대 260ms 이전 시각 정보를 참조 — 파지 구간에서 불리할 수 있습니다.
- DCT의 기여가 분리 측정되지 않았습니다 (`--no-dct`가 대조군 경로).
- 30 FPS 실시간 제어에 미달(23 FPS), baseline 대비 task 정확도 비교 없음.

**향후**: DCT ablation → 200+ 에피소드로 확장 → INT8/INT4 양자화 → 주파수 계층적 디코딩 →
TensorRT/ONNX Runtime 가속.

## 저장소 구조

```
smolvla_fast_dct.py         논문 구조(8 layer · chunk 10 · DCT) 스켈레톤 — 미학습
configuration_smolvla.py    포크 전용 플래그 5개 + 경량 기본값
modeling_smolvla.py         SmolVLAPolicy / VLAFlowMatching — 비전 배칭, posmap, dtype 캐스팅
smolvlm_with_expert.py      VLM + action expert — SDPA, KV 프로젝션 캐싱, lm_head 제거
_local_smolvla.py           설치된 lerobot 위에 이 폴더를 주입하는 import shim
benchmark.py                chunk 지연 측정
optimize_for_inference.py   배포용 체크포인트 생성
tests/                      등가성 21체크 + 비전 배칭 + 오프라인 VLM 캐시 생성기
experiments/                posmap · 스텝 수 · DCT A/B 연구 (RESULTS.md)
onnx_export/                prefill/denoise 2-그래프 ONNX export + numpy 러너
docs/lightweight-fork.md    경량화 포크 상세 (변경 내역 · 측정 · Jetson 배포 · 제약)
```

## 참고문헌

1. M. Shukor et al., "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics," arXiv:2506.01844, 2025.
2. K. Black et al., "π0: A Vision-Language-Action Flow Model for General Robot Control," arXiv:2410.24164, 2024.
3. K. Pertsch et al., "FAST: Efficient Action Tokenization for Vision-Language-Action Models," RSS, 2025.
4. M. J. Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model," arXiv:2406.09246, 2024.
5. J. Chen et al., "NanoVLA: Routing Decoupled Vision-Language Understanding for Nano-sized Generalist Robotic Policies," arXiv:2510.25122, 2025.

## Acknowledgement

본 논문은 과학기술정보통신부 및 정보통신기획평가원의 "SW중심대학사업(기업연계 멘토링)" 지원을 받아
수행되었습니다.

## 라이선스

`huggingface/lerobot` v0.5.1의 `src/lerobot/policies/smolvla/`에서 파생된 수정본이므로
**Apache License 2.0**입니다 — [`LICENSE`](LICENSE) 참조. 수정·신규 파일 목록은
[`docs/lightweight-fork.md`](docs/lightweight-fork.md).
