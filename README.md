# SmolVLA-FAST DCT

**Jetson Orin Nano 8GB에서 동작하는 경량 Vision-Language-Action 모델.**
KV cache 기반 비전-액션 디커플링으로 260ms의 비전 연산을 제어 경로에서 분리하고,
VLM backbone을 절반으로 줄여 decode step을 85ms → 43ms로 단축했습니다.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![base](https://img.shields.io/badge/base-SmolVLA-orange.svg)](https://arxiv.org/abs/2506.01844)
[![hardware](https://img.shields.io/badge/hardware-Jetson%20Orin%20Nano%208GB-76B900.svg)](https://developer.nvidia.com/embedded/jetson-orin)

> **DCT 주파수 도메인 Flow Matching과 비전-액션 디커플링을 활용한 경량 VLA 모델의 Edge 디바이스 실시간 배포**
> *(Real-Time Edge Deployment of Lightweight VLA via DCT Flow Matching and Vision-Action Decoupling)*
>
> 이유준, 변정민, 김민엽, 최규상\*
> 영남대학교 정보통신공학과 · \*교수

---

> ### ⚠️ 저장소 현황
>
> **논문의 학습·배포 코드는 아직 이 저장소에 없습니다.** Jetson의 Docker 환경에서 동작하며
> 정리 후 업로드 예정입니다. 비동기 2-스레드 파이프라인과 SO-101 모터 I/O가 여기에 해당합니다.
>
> 다만 **아키텍처 스켈레톤은 있습니다** — [`smolvla_fast_dct.py`](smolvla_fast_dct.py)가 논문의
> 모델 구조(8 layer · chunk 10 · DCT 도메인)를 조립하며, 파라미터 수가 논문 표 1과 일치합니다.
> [사용법](#아키텍처-스켈레톤) 참조.
>
> 현재 이 저장소에 올라와 있는 코드는 **별개의 SmolVLA 경량화 실험**(lerobot 0.5.1 기반,
> RoPE position 재매핑 · SDPA · expert KV 프로젝션 캐싱)입니다. 문서는
> [`README_smolvla-lite.md`](README_smolvla-lite.md)를 보세요.
>
> | | 이 저장소의 현재 코드 | 본 논문 |
> |---|---|---|
> | lerobot | 0.5.1 | 0.4.4 |
> | 핵심 | posmap (RoPE 레이아웃 복원) | DCT + 비동기 디커플링 |
> | 측정 환경 | Mac (CPU/MPS) | Jetson Orin Nano 8GB |
> | 로봇 | 없음 (오프라인 평가) | SO-101 6-DoF 실기 |

---

## 개요

VLA 모델을 edge 디바이스에 배포할 때의 제약은 세 가지입니다. **메모리** — π0(3B, FP16 약 6GB)나
OpenVLA(7B, 약 14GB)는 Jetson Orin Nano 8GB의 가용 범위를 넘습니다. **지연** — 안정적인 제어
주기를 확보하기 어렵습니다. **파이프라인 비효율** — 시각 처리와 action 생성이 순차적으로 수행되어,
비전 연산 시간이 그대로 제어 주기의 하한이 됩니다.

본 연구는 SmolVLA를 기반으로 세 가지를 결합해 이 제약을 다룹니다.

## 제안 방법

### 1. KV Cache 기반 비전-액션 디커플링

기존 VLA는 매 추론마다 비전 인코딩부터 action denoising까지 전체 파이프라인을 순차 수행합니다.
본 구현 환경에서 비전 처리에만 약 **260ms**가 소요되어, 동기식 추론의 최대 제어 주파수는
약 **3.8 FPS**에 머뭅니다.

SmolVLA의 cross-attention 구조에서는 prefix(시각 + 상태 토큰, 129개)와 suffix(action 토큰)가
attention mask로 구분됩니다. 이 구조를 **아키텍처 변경 없이 그대로 두고**, prefix를 한 번
forward해 생성한 KV cache를 공유 메모리에 저장한 뒤 이후 denoising step들이 `past_key_values`로
재사용합니다. Vision 스레드가 새 cache를 만드는 동안에도 Action 스레드는 이전 cache로 action을
계속 생성할 수 있습니다.

KV cache의 추가 메모리 부담은 **129 토큰 기준 2.1MB**에 불과합니다.

### 2. VLM Backbone Layer 축소 (16 → 8)

SmolVLM2-500M-Video-Instruct의 16개 layer 중 8개만 선택적으로 사용하고, 2개 layer 단위로
self-attention과 cross-attention을 교대 배치해 시각-action 상호작용을 유지했습니다.
Action Expert에는 expert width multiplier 0.75를 적용했습니다.

**측정된 decode step 지연 감소(85ms → 43ms)는 주로 이 layer 축소에 기인합니다.**

### 3. DCT 주파수 도메인 Flow Matching

정규화된 action chunk에 type-II DCT(`norm=ortho`)를 적용해 얻은 주파수 계수를 conditional flow
matching의 목표 분포로 사용합니다. 시간 변수는 `t ~ Beta(1.5, 1.0)`으로 샘플링해 후반 구간에 더
높은 가중을 부여했고, 추론 시 Euler method로 denoising한 뒤 IDCT로 시간 도메인 action을 복원합니다.
Loss는 실제 action 차원인 6-DoF에만 MSE를 적용해 32차원 zero-padding의 영향을 제거했습니다.

> **범위에 대한 명시**: 본 연구에서 DCT는 **계수 절단 없이 chunk 길이와 동일한 수의 계수를
> 사용하므로, 토큰 수와 연산량이 시간 도메인과 동일한 직교 기저 변환**입니다. 따라서 DCT 자체가
> 추론 지연을 직접 감소시키지는 않습니다. 기대 효과는 (i) 저주파 지배적 목표 분포에서의 denoising
> 수렴 특성과 (ii) 고주파 억제를 통한 출력 궤적 평활화이며, **동일 조건(8 layer, chunk 10)에서
> 시간 도메인과 분리 비교하는 ablation은 아직 수행하지 않았습니다.** 관측된 지연 감소를 DCT의
> 기여로 귀속할 수 없습니다.

## 시스템 구성

### 모델

| 항목 | SmolVLA Baseline | SmolVLA-FAST DCT |
|---|---|---|
| 총 파라미터 | 450.0M | **322.3M** (-28.4%) |
| 학습 파라미터 | 99.9M | **50.6M** (-49.3%) |
| VLM Layers | 16 | **8** (-50%) |
| 모델 크기 (FP16) | 920 MB | **657 MB** (-28.6%) |
| Chunk Size | 50 | 10 |
| Action Domain | Time | DCT Frequency |

컴포넌트별 FP16 메모리: SigLIP 172.9MB · Action Expert+LM 468.4MB · KV cache 2.1MB.

**입력 구성**: 전방·상부 카메라 2대의 640×480 RGB를 `resize_with_pad`로 512×512 전처리 →
SigLIP 인코딩 → PixelShuffle로 이미지당 256 패치 → 64 시각 토큰. prefix는 시각 128 + 상태 1 =
**129 토큰**. 비전 인코더 가중치는 frozen입니다.

### 비동기 2-스레드 파이프라인

```
Vision 스레드 (~260ms 주기)          Action 스레드
─────────────────────────           ─────────────────────────
프레임 캡처 (2 cam)                  최신 KV cache 참조
resize_with_pad → [-1,1]      ┌───► denoising step (43ms)
SigLIP 인코딩                  │     action queue 저장
로봇 상태 읽기                  │     30 FPS 순차 전송
prefix forward                │     queue < chunk/2 → 보충
KV cache 갱신 ────────────────┘
        (공유 메모리)
```

시리얼 포트 접근 충돌을 막기 위해 `bus_lock`으로 Vision 스레드의 모터 위치 읽기와 Action 스레드의
목표 위치 쓰기를 상호 배제하고, 통신 오류 예외 처리로 간헐적 시리얼 실패가 전체 제어 루프를
중단시키지 않도록 했습니다.

### 배포 환경

- **NVIDIA Jetson Orin Nano 8GB** — JetPack 6.2, CUDA 12.6, Ampere sm_87
- **Docker**: `dustynv/l4t-pytorch:r36.4.0`
  > PyPI 표준 PyTorch는 데스크탑 GPU 커널(sm_80/86/89/90) 위주라 sm_87을 충분히 지원하지 않아,
  > `conv2d` 등에서 **no kernel image** 오류가 발생합니다. NVIDIA 공식 컨테이너가 필요합니다.
- **로봇**: SO-101 6-DoF follower arm, Feetech STS3215 서보 6개, `/dev/ttyACM0`
- **카메라**: USB 2대 (전방·상부), 640×480 @ 30 fps
- **모터 통신**: lerobot 0.4.4 `FeetechMotorsBus`, calibration으로 `homing_offset` /
  `range_min` / `range_max` 설정

## 아키텍처 스켈레톤

[`smolvla_fast_dct.py`](smolvla_fast_dct.py)는 논문의 **모델 구조**를 이 저장소의 기존 부품으로
조립합니다. 학습된 정책이 아니라 형태만 재현한 것입니다.

```bash
python smolvla_fast_dct.py            # DCT 도메인 — 구성 + 더미 forward
python smolvla_fast_dct.py --no-dct   # 시간 도메인 — DCT ablation의 대조군
```

출력 예:

```
Reducing the number of VLM layers to 8 ...
domain=DCT vlm_layers=8 chunk=10 resolution=(512, 512)
params: 275.0M total, 50.8M trainable
chunk: (1, 10, 6) in 392 ms (untrained weights)
OK - shape contract holds. Model is UNTRAINED; retrain before any claim.
```

**구현 대응**

| 논문 요소 | 스켈레톤에서 |
|---|---|
| VLM layer 16 → 8 | `num_vlm_layers=8` → `smolvlm_with_expert.py`가 `text_model.layers[:8]` 슬라이싱 |
| chunk size 50 → 10 | `chunk_size = n_action_steps = 10` |
| DCT 주파수 도메인 | `experiments/dct_flow.py::convert_policy_to_dct(k=10)` — chunk와 같은 K이므로 **절단 없는 직교 변환** |
| 비동기 파이프라인 · SO-101 I/O | **미포함** (Jetson 코드) |

**파라미터 수 검증** — config가 논문과 일치함을 확인했습니다.

| | 스켈레톤 | 논문 표 1 |
|---|---|---|
| 총 파라미터 | 322.3M (`strip_lm_head=False`) | 322.3M |
| 학습 파라미터 | 50.8M | 50.6M |

기본 실행이 275.0M로 나오는 것은 이 저장소가 SmolVLA에서 호출되지 않는 LM head 47.3M을 제거하기
때문이며, 되돌리면 논문 수치와 일치합니다(275.0 + 47.3 = 322.3).

> ### ⚠️ 이것으로 논문 수치를 재현할 수 없습니다
>
> `layers[:8]`은 사전학습된 트랜스포머의 절반을 버리므로, **재학습 전까지 출력은 무의미합니다.**
> 논문 모델을 얻으려면 80 에피소드 SO-101 데이터로 30,000 step 재학습이 필요하며, 43ms·23 FPS는
> Jetson 실측치입니다.
>
> 계수 통계(`coeff_stats`)를 넘기지 않으면 mean 0 / std 1로 폴백하면서 경고를 출력합니다. 실제
> 학습에는 `experiments/train_ab.py`의 `compute_coeff_stats`로 데이터셋에서 산출해야 합니다.

**ablation 용도** — `--no-dct` 플래그가 [한계](#한계)에 기재된 미수행 ablation의 대조군입니다.
8 layer · chunk 10을 고정한 채 이 플래그만 뒤집으면 DCT 표현의 기여를 분리할 수 있습니다.

## 실험

### 데이터

SO-101 follower arm과 leader arm 텔레오퍼레이션으로 수집한 **큐브 파지** 시연 데이터.

| | |
|---|---|
| 에피소드 | 80개 (평균 약 74초, 30 fps) |
| 총 프레임 | 178,050 |
| 프레임 구성 | 640×480 RGB ×2 + 6-DoF 관절 위치(degree) |
| 정규화 | MEAN_STD (action, state) |

### 학습 설정

batch 128 · lr 1×10⁻⁴ · 30,000 step · grad clip 10.0 · bfloat16 mixed precision ·
cosine warmup 1,000 step 후 2.5×10⁻⁶까지 감소.

데이터 증강 3배 — `aug_idx 1`: action noise σ=0.01, image noise 2%, brightness ±5% /
`aug_idx 2`: σ=0.02, 4%, ±10%. DCT 모델은 stride=1 슬라이딩 윈도우로 약 17만 샘플을 생성했습니다.
5,000 step마다 lerobot 호환 체크포인트(`config.json`, `model.safetensors`, normalizer)를 저장합니다.

### 추론 성능 (Jetson Orin Nano, FP16)

| 모드 | 모델 | decode step | 비고 |
|---|---|---|---|
| 동기 | — | — | 비전 260ms가 병목, 최대 약 3.8 FPS |
| 비동기 | Baseline (16L) | 85 ms | |
| 비동기 | **제안 (8L)** | **43 ms** | layer 절반 축소에 기인 |

> **decode step 시간 ≠ 제어 주파수.** 43ms는 action denoising 1회의 처리 시간입니다. 실제 제어
> 주기는 Action 스레드가 queue에서 목표 위치를 전송하는 주기로 결정되며, chunk size 10 기준
> queue 절반(5 step, 약 167ms) 소진 시간이 decode 시간보다 길어 **본 설정에서 decode는 제어 주기의
> 병목이 아닙니다.** 비동기 모드 최대 처리율은 약 23 FPS이며, 목표한 30 FPS에는 도달하지 못했습니다.

### DCT 에너지 집중 분석

실제 학습 데이터 5,845개 action chunk(chunk size 10)에 DCT를 적용한 계수별 에너지 분포:

| 계수 | 누적 에너지 |
|---|---|
| DC (index 0) 단독 | **99.47%** |
| 상위 2개 | 99.96% |
| 관절별 DC 비율 | 98.7 ~ 99.8% |

> **해석 주의**: chunk size 10은 30 fps 기준 약 **0.33초**에 해당하는 짧은 구간이므로, DC 성분의
> 지배적 비중은 DCT의 에너지 집중 특성뿐 아니라 **관측 구간의 길이에도 기인**합니다. 또한 잔여
> 계수의 분산이 매우 작다는 점은, 계수별 정규화 시 저에너지 성분이 학습 목표에서 과대 가중되어
> 예측 난이도가 높은 차원에 loss가 배분될 수 있음을 시사합니다. 계수 에너지에 비례하는 loss
> 가중이 대응책이 될 수 있습니다.

### 궤적 복원

Cosine similarity **0.9966** — 예측 궤적이 ground truth의 전반적 동작 추세를 따릅니다.
예측 범위도 GT와 대체로 일치했습니다(GT `[-53, 95]` vs Pred `[-53, 98]`), 정규화·복원 파이프라인이
정상 동작함을 확인했습니다. 한편 **MSE 10.19**는 step-to-step 세부 움직임에 여전히 오차가 존재함을
의미합니다.

## 한계

- **정밀 파지에 도달하지 못했습니다.** SO-101 실기 실험에서 baseline과 제안 모델 모두 큐브 방향으로의
  접근은 가능했으나 안정적인 정밀 파지에는 이르지 못했습니다. 따라서 본 연구는 성공적인 정밀 조작이
  아니라 **edge 환경에서의 실시간 추론 가능성과 부분적 행동 생성 가능성을 확인한 초기 검증**입니다.
- 원인 후보는 두 가지이며 아직 분리되지 않았습니다. (i) 80 에피소드 규모의 제한된 학습 데이터.
  (ii) **Vision 스레드의 갱신 주기가 약 260ms이므로 Action 스레드는 최대 260ms 이전의 시각 정보로
  생성된 KV cache를 참조합니다** — 시각 피드백의 즉시성이 요구되는 파지 구간에서 성능 저하 요인으로
  작용할 수 있습니다.
- **DCT의 기여가 분리 측정되지 않았습니다** (위 "범위에 대한 명시" 참조). 대조군 실행 경로는
  [스켈레톤](#아키텍처-스켈레톤)의 `--no-dct`에 준비되어 있습니다.
- 30 FPS 수준의 원활한 실시간 제어에는 도달하지 못했습니다.
- baseline 대비 제안 모델의 **task 정확도 비교가 아직 없습니다.**

## 향후 계획

1. **DCT ablation** — 동일한 8 layer · chunk 10 조건에서 시간 도메인 vs DCT 주파수 도메인을 분리
   비교해 DCT 표현의 기여를 정량화. 두 arm 모두 [스켈레톤](#아키텍처-스켈레톤)에 준비되어 있음
   (`--no-dct` 플래그)
2. **데이터 확장** — 200개 이상 에피소드로 일반화 성능과 조작 정밀도 향상
3. **양자화** — INT8/INT4로 모델 크기와 추론 지연 추가 감소
4. **주파수 계층적 디코딩** — 저주파 계수는 autoregressive하게, 고주파는 병렬 복원
5. **TensorRT / ONNX Runtime** 기반 Jetson 특화 추론 가속

## 참고문헌

1. K. Black et al., "π0: A Vision-Language-Action Flow Model for General Robot Control," arXiv:2410.24164, 2024.
2. Octo Model Team, "Octo: An Open-Source Generalist Robot Policy," RSS, 2024.
3. M. Shukor et al., "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics," arXiv:2506.01844, 2025.
4. K. Pertsch et al., "FAST: Efficient Action Tokenization for Vision-Language-Action Models," RSS, 2025.
5. M. J. Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model," arXiv:2406.09246, 2024.
6. J. Chen, J. Wang, L. Chen, C. Cai, and J. Lu, "NanoVLA: Routing Decoupled Vision-Language Understanding for Nano-sized Generalist Robotic Policies," arXiv:2510.25122, 2025.
7. T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware," RSS, 2023.
8. NVIDIA, "Jetson Orin Nano Developer Kit User Guide," 2024.

## Acknowledgement

본 논문은 과학기술정보통신부 및 정보통신기획평가원의 "SW중심대학사업(기업연계 멘토링)" 지원을 받아
수행되었습니다.

## 라이선스

SmolVLA / lerobot에서 파생되었으므로 **Apache License 2.0**입니다 — [`LICENSE`](LICENSE) 참조.
