# SmolVLA 경량화 버전

원본: `huggingface/lerobot` `src/lerobot/policies/smolvla/` (v0.5.1, 커밋 `19c6adef`)
목표: **Jetson Orin Nano 8GB에서 30Hz 이상 제어**, 약간의 성능 하락 허용.
원칙: **사전학습 체크포인트(`lerobot/smolvla_base`)와의 호환을 깨는 변경은 하지 않음** — 레이어 절단·백본 교체는 재학습 없이는 성능이 붕괴하므로(커뮤니티 실증: 256M 스왑 시 성공률 7.5%) 제외.

## 적용된 변경

### 무손실 (액션 출력이 원본과 동일 — 검증됨)

| 변경 | 위치 | 효과 |
|---|---|---|
| **미사용 lm_head 제거** (`strip_lm_head=True`) | `smolvlm_with_expert.py` 생성자 | **-47.3M 파라미터 (~10%)**, bf16 기준 ~95MB. SmolVLA는 LM head를 아예 호출하지 않고 임베딩도 untied라 완전 무손실. 액션 diff = 0 확인 |
| **Expert KV 프로젝션 캐싱** (`cache_expert_kv_projections=True`) | `forward_cross_attn_layer` | cross-attn 레이어의 expert k/v 프로젝션은 타임스텝과 무관 → 첫 디노이징 스텝에만 계산하고 재사용. 액션 diff = 0 확인 |

### 준-무손실 (부동소수점 노이즈 수준 — fp64 기준 검증됨)

| 변경 | 위치 | 효과 |
|---|---|---|
| **SDPA attention** (`attention_implementation="sdpa"`) | `sdpa_attention_forward` | eager 대비 attention 1.5~3× 가속 + L×L fp32 행렬 미실체화. 커널 단독 diff 5.4e-07. fully-masked row(패딩)에서 SDPA의 NaN을 0으로 치환하는 안전장치 포함 |

### 약간의 성능 하락 트레이드오프 (신규 config의 기본값만 변경)

| 변경 | 기본값 | 효과 / 리스크 |
|---|---|---|
| **디노이징 스텝 축소** | `num_steps: 10 → 4` | 지연 ~40%↓, 액션 MAE 0.037 (512px 기준) — 싼 편. **태스크 성공률로 A/B 검증 권장** |
| **이미지 해상도 축소** | `resize_imgs_with_padding: 512 → 384` | 비전 연산 ~45%↓, 이미지당 토큰 64→36. **정확도 비용의 대부분이 여기 있습니다**: smolvla_base 기준 512/10 대비 액션 MAE 0.202(액션 평균 크기의 53%, corr 0.838), 반면 512px+4스텝은 MAE 0.037. **이 해상도로 파인튜닝하지 않았다면 512px로 배포하고 num_steps만 낮추세요.** 변은 64의 배수 (config에서 검증) |

> ⚠️ **중요**: 위 두 기본값은 **새로 만드는 config에만** 적용됩니다. `from_pretrained("lerobot/smolvla_base")`로 로드하면 체크포인트의 config.json에 저장된 값(512, 10)이 우선합니다. 배포 시 `optimize_for_inference.py`로 오버라이드해서 저장하거나 런타임에 `--policy.num_steps=4` 식으로 지정하세요. 반대로 신규 플래그 4개(sdpa/캐싱/strip/비전배칭)는 구 체크포인트에도 기본 적용됩니다(전부 무손실·준무손실). `posmap_ref_resolution`은 기본 off입니다.

### posmap — 해상도 페널티 중 레이아웃 성분 되돌리기 (`posmap_ref_resolution`, 기본 off)

SmolVLA는 RoPE 위치를 `position_ids = cumsum(pad_masks) - 1`로 만듭니다. 따라서 `resize_imgs_with_padding`을
바꾸면 **두 가지가 동시에** 일어납니다 — 픽셀이 바뀌고, **모든 position id가 조용히 재번호매김**됩니다
(카메라당 token grid가 바뀌고, language/state token이 앞뒤로 밀림).

`posmap_ref_resolution`을 **체크포인트가 학습된 해상도**로 두면 두 번째만 되돌립니다: 각 image token이
기준 grid에서 차지했을 cell의 position id를 받고, language/state는 기준 슬롯에 남습니다.
추가 token 0 · 추가 FLOP 0 · 학습 0 · 지연 변화 0. 기준 해상도와 같으면 **비트 단위 no-op**입니다 (테스트로 강제).

**측정** (`experiments/runs/baseline/checkpoint.pt`, 384px에서 1200스텝 파인튜닝 → 네이티브 grid 6×6, held-out ep45-49 68윈도우,
시드 3개, GT 액션 대비 open-loop MAE, 4스텝):

| 해상도 | plain | +posmap(384) | 회복 | 지연 |
|---|---|---|---|---|
| 256 | 7.515° | **6.227°** | -17% | 87 ms |
| 320 | 8.064° | **7.544°** | -6% | 105 ms |
| **384 (네이티브)** | **5.679°** | 5.679° (no-op) | — | 130 ms |
| 448 | 6.862° | **5.787°** | -16% | 165 ms |
| 512 | 7.530° | **6.290°** | -16% | 208 ms |

두 가지가 확인됩니다:
1. **기준 grid는 체크포인트가 학습한 grid여야 합니다.** 320px에서 6×6 기준은 7.544°, 8×8 기준은 7.830° —
   틀린 기준을 쓰면 오히려 분포 이동을 주입합니다.
2. **512px는 384px보다 정보가 더 많은데도 33% 더 나쁩니다** (7.530° vs 5.679°). 정보 손실로는 설명할 수 없고,
   posmap이 그 격차의 2/3를 되돌립니다. 즉 해상도 페널티의 상당 부분은 픽셀이 아니라 **레이아웃**입니다.

> ⚠️ **배포 지침은 바뀌지 않습니다**: 체크포인트의 학습 해상도가 여전히 최적입니다. posmap은 더 싼 동작점을
> 만들어주지 못합니다 (320+posmap도 384보다 +1.87° 나쁨). **학습 해상도에서 벗어나야만 할 때만 켜세요.**
> 저장되지 않는 포크 전용 필드이므로 런타임에 지정해야 합니다.

### dtype 적응형 캐스팅 (bf16 배포 지원)

`modeling_smolvla.py`의 프로젝션 3곳(`state_proj`, `action_in_proj`, `action_out_proj`)이 가중치 dtype을 따라가도록 수정 → `policy.model.to(torch.bfloat16)` 한 줄로 전체 bf16 추론 가능. fp32 가중치에서는 기존과 완전 동일(no-op). Euler 적분(`x_t + dt*v_t`)은 타입 승격으로 fp32 유지, RoPE/softmax 내부도 fp32 유지.

## 새 파일

- `_local_smolvla.py` — 설치된 lerobot 위에 이 폴더의 수정 모듈을 주입 (설치본은 건드리지 않음)
- `optimize_for_inference.py` — 경량 설정(num_steps/해상도)을 구운 배포용 체크포인트 생성
- `benchmark.py` — chunk 지연 측정 (Jetson에서 실행 권장). 제어율 판정은 하지 않습니다 — 아래 참조

## 측정 결과 (이 Mac에서, 무작위 가중치 = 실제 가중치와 지연 동일)

| 구성 | 디바이스 | chunk 지연 | open-loop 재생률 |
|---|---|---|---|
| 원본 (eager, 512px, 10스텝, fp32) | CPU | 1157 ms | 43 Hz |
| **경량화 (sdpa, 384px, 4스텝, fp32)** | CPU | **496 ms (2.4×)** | **101 Hz** |
| 원본 (eager, 512px, 10스텝, fp32) | MPS(GPU) | 295 ms | 170 Hz |
| **경량화 + bf16 (sdpa, 384px, 4스텝)** | MPS(GPU) | **152 ms (1.9×)** | **328 Hz** |

> 재현성 주의: 동일 명령이 한 세션 안에서 ±35% 흔들립니다(509~742ms). 배율(2.4×)은 재현되지만 절대값 3자리는 재현되지 않습니다. 표는 `--cameras 2`로 측정 — `smolvla_base`는 카메라 3개이고 그 경우 +25%입니다.

수치 검증(전 항목 통과): eager↔SDPA 커널 5.4e-07 일치 · KV캐싱/lm_head 제거 비트 동일 · fp64 기준 SDPA 오차는 eager와 동수준 · 학습 forward 정상 · bf16 정상.

## Jetson Orin Nano 8GB 배포 가이드

**30 FPS는 chunk 지연만으로 결정되지 않습니다.** SmolVLA는 1회 추론으로 액션 50개를 생성하지만, lerobot의 기본 **동기** 제어 루프는 큐가 빌 때마다 추론이 끝날 때까지 멈춥니다 — 평균 제어율 = `50 / (49/30 + L)`이라 **어떤 유한한 L로도 30Hz 평균은 나오지 않습니다** (L=1667ms면 15.2Hz).

30Hz의 유일한 경로는 lerobot **async inference**이고, 거기서의 실제 예산은
`chunk_size_threshold × n_action_steps / target_hz` = `0.5 × 50 / 30` ≈ **833ms**,
관측 캡처와 카메라 프레임 gRPC 전송을 빼면 실효 **~600–700ms**입니다.
(측정된 128~160ms는 이 예산도 5배 여유로 통과하지만, **실제 제어율은 async 클라이언트에서 직접 재야 합니다.**)

```bash
# 0) 전원 모드 (기본 15W면 성능 절반 이하)
sudo nvpmodel -m 2 || sudo nvpmodel -m 0   # MAXN SUPER가 있으면 2
sudo nvpmodel -q && sudo jetson_clocks

# 1) 의존성. JetPack 기본 Python은 3.10인데 lerobot 0.5.x는 >=3.12를 요구합니다.
#    그냥 pip install 하면 에러 없이 lerobot 0.4.4로 백트래킹 설치되고,
#    나중에 `lerobot.policies.rtc` import에서 죽습니다. 반드시 확인하세요:
python3 -c "import sys; assert sys.version_info >= (3,12), sys.version"
python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "CUDA torch 필요 - NVIDIA jetson wheel 설치 후 재실행"; exit 1; }
pip3 install "lerobot[smolvla,async]"   # async 없으면 30Hz 이야기가 성립 안 함
pip3 install "transformers>=5.3,<5.4"   # 5.14는 lerobot.policies import를 깨뜨림

# 2) 스모크 (무작위 가중치, 다운로드 최소)
python3 benchmark.py --random --device cuda --dtype bfloat16 --iters 3 --warmup 2

# 3) 배포용 체크포인트 + 실측
#    --resolution 384는 그 해상도로 파인튜닝했을 때만 추가하세요 (위 트레이드오프 표 참조)
python3 optimize_for_inference.py --policy-path lerobot/smolvla_base \
    --output-dir ./smolvla_optimized --num-steps 4
python3 benchmark.py --policy-path ./smolvla_optimized --device cuda --dtype bfloat16

# 4) 진짜 판정: async inference를 띄우고 클라이언트 관측 FPS를 기록
#    (chunk 지연이 아니라 이 숫자가 30Hz의 근거입니다)
```

- 메모리: 가중치 ~0.8GB. 프로세스 peak ~2.0GB + CUDA context ~0.5–1.0GB + GUI ~1.0–1.4GB ≈ **3.5–4.4GB / 8GB**. 다만 fp32 생성 → bf16 캐스트 경로에 일시적으로 ~1.8GB가 뜹니다. swap/zram과 `systemctl set-default multi-user.target`을 권장합니다
- torch.compile (`--compile`): 워밍업 수십 초~수 분. **CUDA 가속 효과는 미측정**입니다 (CPU 0%, MPS는 크래시 → 자동으로 eager 폴백). graph break는 포크 코드 0개, 유일한 break가 transformers 쪽이며 `torch._dynamo.config.capture_dynamic_output_shape_ops = True` 한 줄로 없앨 수 있습니다

## 되돌리기

- 완전 원본으로: 각 파일을 lerobot 저장소 원본으로 교체 (`~/lerobot/src/lerobot/policies/smolvla/`)
- 개별 기능만 (런타임에만 — 저장되지 않음): config에서 `attention_implementation="eager"`, `cache_expert_kv_projections=False`, `strip_lm_head=False`, `batch_vision_encoder=False`, `posmap_ref_resolution=None`, `num_steps=10`, `resize_imgs_with_padding=(512, 512)`

## 남은 큰 옵션 (원하면 다음 단계)

1. **Flow matching 스텝 증류 (10→1~2스텝)** — 학습 필요하지만 문헌상 무손실 ~10× 가속 (SnapFlow 등)
2. **384px 파인튜닝** — 해상도 하락분 회복
3. **GGUF + vla.cpp** — PyTorch 자체를 제거하는 최경량 경로 (Jetson 실증 사례 있음)

## 적대적 리뷰에서 발견되어 수정된 사항

멀티에이전트 리뷰(발견→반박 검증)에서 확인된 이슈와 수정:

| 이슈 | 수정 |
|---|---|
| 신규 config 키 4개 때문에 **저장한 체크포인트가 stock lerobot에서 로드 불가** (draccus가 미지의 필드에 하드 에러) | `draccus.encode`에 SmolVLAConfig 인코더를 등록해 모든 dump에서 포크 전용 키 4개 제거 → config.json **과 train_config.json** 양쪽 커버 (`_save_pretrained` 오버라이드만으로는 후자가 새어 `lerobot-train --resume`이 깨졌음). stock 클래스 파싱 검증됨 |
| `_local_smolvla` 스텁이 `from lerobot.policies import ACTConfig...`를 쓰는 **async_inference를 ImportError로 깨뜨림** | 스텁에 lazy `__getattr__` 추가 — 요청된 이름만 그때 로드 (GROOT처럼 깨진 모듈은 건드리지 않는 한 무해) + 속성 체인(`lerobot.policies.smolvla.x`) 완성 |
| `optimize_for_inference.py`가 체크포인트의 num_steps/해상도를 **CLI 기본값으로 조용히 덮어씀** | `--num-steps`/`--resolution` 기본값 None = 체크포인트 값 유지, 변경 시 명시 출력 |
| **bf16 저장 후 재로드하면 조용히 fp32로 복원** (safetensors가 수신 모듈 dtype을 따름) | 스크립트/README에 명시 + 로드 후 `policy.model.to(torch.bfloat16)` 재적용 안내 (benchmark.py `--dtype bfloat16`이 자동 수행) |
| 로컬 체크포인트의 **전처리 파이프라인 파일(정규화 통계 등) 미복사** | `optimize_for_inference.py`가 부속 파일 통과 복사 (허브 smolvla_base는 원래 config+weights만 있음) |
| SDPA fully-masked row 처리의 `.any()` 분기가 **매 호출 GPU→CPU 동기화 + torch.compile graph break** 유발, 구버전 torch에서 backward NaN 미보호 | 분기 제거(branch-free): 더미 마스크 행 + 무조건 masked_fill — 검증 결과 수치 동일 |
| **bf16 모델로 학습 forward 호출 시 dtype 크래시** (fp32 하드코딩 1곳 잔존) | 학습 경로도 projection dtype을 따르도록 수정 (fp32 가중치에선 기존과 동일) — bf16 학습 forward 통과 검증 |
| `--resolution`이 64배수 검증을 **우회** (post_init은 생성 시에만 실행) | 두 스크립트 모두 argparse 단계에서 검증 |

### 이 Mac 환경 한정 참고
이 venv(lerobot 0.5.0 + transformers 5.14)에서는 **GROOT 정책 모듈이 깨져 있어** stock `lerobot.policies` 패키지 import 자체가 실패합니다 (우리 수정과 무관한 기존 문제). `_local_smolvla`의 스텁이 이를 우회하므로 이 폴더의 스크립트는 정상 동작합니다. Jetson에는 lerobot이 고정하는 transformers 버전을 쓰면 해당 없음.

## 후속 실험 결과 (2026-07-18~20, `experiments/` 참고)

- **스텝 수 탐침 (무학습)**: 파인튜닝된 단일 태스크에서 디노이징 1/2/4/10스텝의 open-loop MAE가 전부 동급 (5.7~6.1°, run 분산 ±0.3°). → **배포 시 `num_steps=2` 권장** (지연 추가 -11%). 멀티태스크/제로샷은 다중 모드 평균화 위험이 있어 기본값은 4 유지. 스텝 증류는 불필요해짐.
- **멀티카메라 비전 배칭** (`batch_vision_encoder=True` 기본): 모든 카메라를 SigLIP 1회 배치 패스로 인코딩. CPU fp32에서 **비트 단위 동일**, MPS fp32 1.4e-06 / MPS bf16 1.2e-02 (배치 차원이 바뀌면 GEMM reduction 순서가 달라짐 — CUDA 미검증). Mac(MPS)에서는 속도 중립 — CUDA(Jetson) 소배치에서 이득 기대, 실측 필요.
- **DCT 계수 공간 flow matching (K=16)**: 2회 A/B 후 기각 — `experiments/RESULTS.md` 참조.
- 검증 스위트는 `tests/`에 보존 (`test_lightweight.py` 21체크, `test_batched_vision.py`). Mac 절대경로/MPS가 하드코딩돼 있어 **Jetson에서는 그대로 실행되지 않습니다**.
- **ONNX 2-그래프 export** (`onnx_export/`): prefill.onnx + denoise.onnx + numpy 러너. PyTorch와 diff 7e-06 일치 검증. 사전 빌드 바이너리는 stale(512px)해서 삭제했습니다 — 배포 해상도로 직접 export하세요(1분). TensorRT 경로는 미실행.

## 주의

- 최종 판단은 **실제 태스크 성공률**로. 액션 MSE가 비슷해도 성공률이 다를 수 있습니다.
- 포크 전용 플래그 5개(`attention_implementation`/`cache_expert_kv_projections`/`strip_lm_head`/`batch_vision_encoder`/`posmap_ref_resolution`)는 **저장되지 않습니다.** 재로드하면 항상 기본값으로 돌아갑니다 (stock lerobot 호환을 위한 의도된 동작) — 아래 되돌리기는 런타임에만 유효합니다.
- lerobot CLI(`lerobot-record`/`lerobot-train`)는 `lerobot.policies`를 먼저 import하므로 `use_local_smolvla()`가 거부되고 **경고 한 줄만 남기고 조용히 stock 코드로 돌아갑니다.** 포크로 돌리려면 wrapper가 필요합니다:
  ```python
  import runpy, _local_smolvla; _local_smolvla.use_local_smolvla()
  runpy.run_module("lerobot.scripts.lerobot_record", run_name="__main__")
  ```
