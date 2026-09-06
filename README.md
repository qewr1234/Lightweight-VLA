# SmolVLA-Lite

**Jetson Orin Nano 8GB에서 SmolVLA를 돌리기 위한 경량화 포크.** 재학습 없이, 사전학습 체크포인트
호환을 깨지 않고 chunk 지연을 2.4× 줄입니다.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](http://www.apache.org/licenses/LICENSE-2.0)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![upstream](https://img.shields.io/badge/upstream-lerobot%20v0.5.1-orange.svg)](https://github.com/huggingface/lerobot)

원본: [`huggingface/lerobot`](https://github.com/huggingface/lerobot) v0.5.1 (`19c6adef`)
`src/lerobot/policies/smolvla/` · 논문: [arXiv:2506.01844](https://arxiv.org/abs/2506.01844)

---

## 설계 원칙

**사전학습 체크포인트(`lerobot/smolvla_base`)와의 가중치 호환을 깨는 변경은 하지 않습니다.**
레이어 절단·백본 교체는 재학습 없이 성능이 붕괴하므로(커뮤니티 실증: 256M 백본 스왑 시 성공률 7.5%)
후보에서 제외했습니다. 여기 있는 모든 변경은 가중치 shape을 건드리지 않으며, 저장한 체크포인트는
**stock lerobot으로 그대로 다시 로드됩니다.**

## 무엇이 바뀌었나

### 무손실 — 액션 출력이 원본과 비트 단위로 동일 (검증됨)

| 변경 | 플래그 | 효과 |
|---|---|---|
| **미사용 LM head 제거** | `strip_lm_head=True` | VLM에서 **47,308,800 파라미터 제거** (507.5M 중 ~9.3%). SmolVLA는 LM head를 호출하지 않고 임베딩도 untied(`tie_word_embeddings=False`)라 완전 무손실 |
| **Expert KV 프로젝션 캐싱** | `cache_expert_kv_projections=True` | cross-attn 레이어의 expert k/v 프로젝션은 타임스텝과 무관 → 첫 디노이징 스텝에만 계산하고 재사용 |

### 준-무손실 — 부동소수점 노이즈 수준

| 변경 | 플래그 | 효과 |
|---|---|---|
| **SDPA attention** | `attention_implementation="sdpa"` | eager 대비 attention 1.5~3× 가속 + L×L fp32 행렬 미실체화. 커널 단독 diff 5.4e-07. 패딩으로 완전히 마스킹된 row에서 SDPA가 내는 NaN을 branch-free로 0 치환 |
| **멀티카메라 비전 배칭** | `batch_vision_encoder=True` | 모든 카메라를 SigLIP 1회 배치 패스로 인코딩. CPU fp32 **비트 동일**, MPS fp32 1.4e-06 / MPS bf16 1.2e-02 (배치 차원이 바뀌면 GEMM reduction 순서가 달라짐). CUDA 미검증 |
| **dtype 적응형 캐스팅** | (플래그 없음) | 프로젝션 3곳(`state_proj`/`action_in_proj`/`action_out_proj`)이 가중치 dtype을 따라감 → `policy.model.to(torch.bfloat16)` 한 줄로 전체 bf16 추론. fp32 가중치에서는 완전한 no-op |

### 트레이드오프 — 정확도를 지연과 교환 (신규 config **기본값**만 변경)

| 변경 | 기본값 | 효과 / 리스크 |
|---|---|---|
| **디노이징 스텝 축소** | `num_steps: 10 → 4` | 지연 ~40%↓, 액션 MAE 0.037 (512px 기준) — **싼 편** |
| **이미지 해상도 축소** | `resize_imgs_with_padding: 512 → 384` | 비전 연산 ~45%↓, 이미지당 토큰 64→36. **정확도 비용의 대부분이 여기 있습니다**: `smolvla_base` 기준 512/10 대비 액션 MAE 0.202 (액션 평균 크기의 53%, corr 0.838). 변은 64의 배수여야 함 |

> ⚠️ **이 해상도로 파인튜닝하지 않았다면 512px로 배포하고 `num_steps`만 낮추세요.**
> 두 기본값은 **새로 만드는 config에만** 적용됩니다. `from_pretrained("lerobot/smolvla_base")`로 로드하면
> 체크포인트 `config.json`에 저장된 값(512, 10)이 우선합니다.

### 옵션 — `posmap_ref_resolution` (기본 `None` = off)

SmolVLA는 RoPE 위치를 `position_ids = cumsum(pad_masks) - 1`로 만듭니다. 따라서 해상도를 바꾸면
**두 가지가 동시에** 일어납니다 — 픽셀이 바뀌고, **모든 position id가 조용히 재번호매김**됩니다.
이 값을 **체크포인트가 학습된 해상도**로 두면 두 번째만 되돌립니다. 추가 토큰 0 · 추가 FLOP 0 ·
학습 0 · 지연 변화 0이고, 기준 해상도와 같으면 **비트 단위 no-op**입니다(테스트로 강제).

자세한 측정과 배포 지침은 [실험 결과](#실험-결과)를 보세요.

---

## 빠른 시작

```bash
# 요구사항: Python >= 3.12 (lerobot 0.5.x의 요구사항), lerobot 0.5.1
pip install "lerobot[smolvla,async]"
pip install "transformers>=5.3,<5.4"   # 5.14는 lerobot.policies import를 깨뜨립니다

git clone <this-repo> smolvla_copy && cd smolvla_copy

# 1) 스모크 테스트 — 무작위 가중치, 1GB 가중치 다운로드 없음
python benchmark.py --random --device cpu --iters 3

# 2) 배포용 체크포인트 만들기 (경량 설정을 config.json에 굽습니다)
python optimize_for_inference.py --policy-path lerobot/smolvla_base \
    --output-dir ./smolvla_optimized --num-steps 4

# 3) 실측
python benchmark.py --policy-path ./smolvla_optimized --device cuda --dtype bfloat16
```

`--num-steps` / `--resolution`은 **기본값이 `None`(체크포인트 값 유지)** 입니다. 명시할 때만 덮어씁니다.
`--resolution`은 64의 배수여야 하며 argparse 단계에서 검증합니다.

### 파이썬에서 쓰기

```python
from _local_smolvla import use_local_smolvla
use_local_smolvla()          # 반드시 lerobot import 전에. 이미 import됐으면 RuntimeError

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

policy = SmolVLAPolicy.from_pretrained("./smolvla_optimized")
policy.eval()
policy.to("cuda")                              # 정규화 버퍼까지 함께 이동
policy.model.to(torch.bfloat16)                # 가중치만 bf16 (버퍼는 fp32 유지)
chunk = policy.predict_action_chunk(batch)     # (B, 50, action_dim)
```

`_local_smolvla.py`는 설치된 lerobot 위에 이 폴더의 수정 모듈을 주입합니다 — **설치본은 건드리지 않습니다.**

---

## 측정 결과

Mac 기준, `--cameras 2`, 무작위 가중치(지연은 실제 가중치와 동일):

| 구성 | 디바이스 | chunk 지연 | open-loop 재생률 |
|---|---|---|---|
| 원본 (eager, 512px, 10스텝, fp32) | CPU | 1157 ms | 43 Hz |
| **경량화 (sdpa, 384px, 4스텝, fp32)** | CPU | **496 ms (2.4×)** | **101 Hz** |
| 원본 (eager, 512px, 10스텝, fp32) | MPS | 295 ms | 170 Hz |
| **경량화 + bf16 (sdpa, 384px, 4스텝)** | MPS | **152 ms (1.9×)** | **328 Hz** |

> **재현성**: 동일 명령이 한 세션 안에서 ±35% 흔들립니다(509~742ms). 배율(2.4×)은 재현되지만
> 절대값 3자리는 재현되지 않습니다. `smolvla_base`는 카메라 3개이고 그 경우 +25%입니다.
> **Jetson 실측 수치는 아직 없습니다** — 위 표는 전부 Mac입니다.

### ⚠️ 이 Hz는 제어율이 아닙니다

`benchmark.py`가 출력하는 것은 **open-loop 재생률**(`n_action_steps / 지연`)입니다. SmolVLA는 1회 추론으로
액션 50개를 만들지만, lerobot의 기본 **동기** 제어 루프는 큐가 빌 때마다 추론이 끝날 때까지 멈춥니다.
평균 제어율은 `50 / (49/30 + L)`이라 **어떤 유한한 L로도 30Hz 평균이 나오지 않습니다**(L=1667ms → 15.2Hz).

30Hz의 유일한 경로는 lerobot **async inference**이고, 거기서의 실제 예산은
`chunk_size_threshold × n_action_steps / target_hz` = `0.5 × 50 / 30` ≈ **833ms**,
관측 캡처와 카메라 프레임 gRPC 전송을 빼면 실효 **~600–700ms**입니다.
**진짜 판정은 async 클라이언트의 관측 FPS로 하세요.**

### 검증

```bash
python tests/test_lightweight.py       # 21체크: 등가성 + 스모크
python tests/test_batched_vision.py    # 비전 배칭 등가성 + 타이밍
```

eager↔SDPA 커널 5.4e-07 일치 · KV캐싱/lm_head 제거 비트 동일 · fp64 기준 SDPA 오차는 eager와 동수준 ·
posmap no-op 비트 동일 · config 직렬화에 포크 전용 키 미포함 · 학습 forward 정상 · bf16 정상.

> ⚠️ pytest 파일이 아니라 그냥 스크립트입니다 — `[PASS]`/`[FAIL]`을 출력하고 exit code로 판정합니다.
> `test_batched_vision.py`와 `experiments/`의 eval 스크립트들은 **디바이스가 `mps`로 하드코딩**돼 있어
> Jetson/CUDA에서는 그대로 실행되지 않습니다. 체크들은 하나의 policy 인스턴스를 공유하며 순서에
> 의존하므로 개별 실행이나 재정렬이 불가능합니다.

---

## Jetson Orin Nano 8GB 배포

```bash
# 0) 전원 모드 — 기본 15W면 성능이 절반 이하입니다
sudo nvpmodel -m 2 || sudo nvpmodel -m 0   # MAXN SUPER가 있으면 2
sudo nvpmodel -q && sudo jetson_clocks

# 1) 함정 확인. JetPack 기본 Python은 3.10인데 lerobot 0.5.x는 >=3.12를 요구합니다.
#    그냥 pip install 하면 에러 없이 lerobot 0.4.4로 백트래킹 설치되고,
#    나중에 `lerobot.policies.rtc` import에서 죽습니다.
python3 -c "import sys; assert sys.version_info >= (3,12), sys.version"
python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "CUDA torch 필요 - NVIDIA jetson wheel 설치 후 재실행"; exit 1; }
pip3 install "lerobot[smolvla,async]"   # async 없으면 30Hz 이야기가 성립 안 합니다
pip3 install "transformers>=5.3,<5.4"
```

- **메모리**: 가중치 ~0.8GB. 프로세스 peak ~2.0GB + CUDA context ~0.5–1.0GB + GUI ~1.0–1.4GB
  ≈ **3.5–4.4GB / 8GB**. fp32 생성 → bf16 캐스트 경로에 일시적으로 ~1.8GB가 뜹니다.
  swap/zram과 `systemctl set-default multi-user.target`을 권장합니다.
- **bf16은 저장되지 않습니다**: safetensors가 수신 모듈의 dtype을 따르므로, bf16으로 저장해도 재로드하면
  조용히 fp32로 돌아옵니다. 런타임에 `policy.model.to(torch.bfloat16)`을 다시 적용하세요
  (`benchmark.py --dtype bfloat16`이 자동으로 합니다).
- **torch.compile** (`--compile`): 워밍업 수십 초~수 분. **CUDA 가속 효과는 미측정**입니다
  (CPU 0%, MPS는 크래시 → 자동 eager 폴백). graph break는 포크 코드 0개.

---

## 실험 결과

전체 내용: [`experiments/RESULTS.md`](experiments/RESULTS.md) · 원시 수치: `experiments/runs/*.json`
데이터: `lerobot/svla_so100_pickplace` (train ep0-44, held-out ep45-49, 68 윈도우)

### posmap — 해상도 페널티 중 레이아웃 성분 되돌리기

`experiments/runs/baseline/checkpoint.pt`(384px에서 1200스텝 파인튜닝 → 네이티브 grid 6×6),
시드 3개 페어링, GT 대비 open-loop MAE(도), 4스텝:

| 해상도 (grid) | plain | +posmap(6×6) | 회복 | 지연 |
|---|---|---|---|---|
| 256 (4×4) | 7.515° | **6.227°** | -17% | 87 ms |
| 320 (5×5) | 8.064° | **7.544°** | -6% | 105 ms |
| **384 (6×6, 네이티브)** | **5.679°** | 5.679° (bit-exact no-op) | — | 130 ms |
| 448 (7×7) | 6.862° | **5.787°** | -16% | 165 ms |
| 512 (8×8) | 7.530° | **6.290°** | -16% | 208 ms |

1. **기준 grid는 체크포인트가 학습한 grid여야 합니다.** 320px에서 6×6 기준은 7.544°, 8×8 기준은 7.830° —
   틀린 기준을 쓰면 오히려 분포 이동을 *주입*합니다. 이 대조군이 메커니즘을 확증합니다.
2. **512px는 384px보다 정보가 엄격히 더 많은데도 33% 더 나쁩니다.** 정보 손실로는 설명할 수 없고,
   posmap이 그 격차의 2/3를 되돌립니다. **해상도 페널티의 상당 부분은 픽셀이 아니라 레이아웃입니다.**

> **배포 지침은 바뀌지 않습니다**: 체크포인트의 학습 해상도가 여전히 최적입니다. posmap은 더 싼 동작점을
> 만들어주지 못합니다(320+posmap도 384보다 +1.87° 나쁨). **학습 해상도에서 벗어나야만 할 때만 켜세요.**

### 디노이징 스텝 수 (무학습 탐침)

파인튜닝된 단일 태스크에서 1/2/4/10스텝의 open-loop MAE가 전부 동급입니다(5.7~6.1°, run 분산 ±0.3°).
→ **배포 시 `num_steps=2`도 고려할 만합니다.** 멀티태스크/제로샷은 다중 모드 평균화 위험이 있어
기본값은 **4**로 유지했습니다. 스텝 증류(distillation)는 이 태스크에서는 불필요해졌습니다.

### DCT-16 계수 공간 flow matching — **기각**

액션 50 토큰을 DCT 계수 16 토큰으로 대체해 디노이징 스텝당 비용을 줄이는 시도. 2회 A/B 후 기각:
정확도 **+53% 나쁨**, 지연 이득은 **4%뿐**(비전+prefill이 지배하는 구조에서 suffix 3.1× 압축은
wall-clock에 거의 안 보임). 에너지 가중 loss로 격차를 일부(-6%) 줄였지만 결론은 불변.

---

## ONNX (2-그래프 export)

optimum이 Idefics3/SmolVLM을 지원하지 않아 직접 만든 커스텀 export입니다. 상세: [`onnx_export/README.md`](onnx_export/README.md)

```
prefill.onnx : 이미지 2장(1,3,480,640) + 언어 토큰/마스크(1,48) + 정규화 상태(1,32)
               → 16레이어 프리픽스 KV 캐시(32텐서) + prefix_pad_masks   (chunk당 1회)
denoise.onnx : x_t(1,50,32) + timestep(1,) + KV 캐시 → v_t(1,50,32)     (num_steps회)
호스트 루프  : Euler 적분 — onnx_runner.py, numpy + onnxruntime만 필요
```

```bash
pip install onnxruntime          # lerobot[smolvla]에 포함되지 않습니다
# 저장소 루트에서 실행하세요 (--policy-path가 루트 기준 상대경로입니다)
python onnx_export/export_onnx.py --policy-path ./smolvla_optimized --verify --out ./onnx_out
python onnx_export/onnx_runner.py --model-dir ./onnx_out --num-steps 4
```

- **PyTorch ↔ ONNX 액션 max|diff| = 7.15e-06** (동일 노이즈, end-to-end PASS)
- **ORT CPU(Mac)에서는 PyTorch보다 1.13× 느립니다** (783ms vs 696ms). GPU/TensorRT가 본 목적입니다
- **TensorRT는 한 번도 실행된 적이 없습니다.** Jetson 처리량 수치는 이 저장소에 존재하지 않습니다
- **`--resolution` 플래그가 없습니다** — 해상도는 체크포인트 config에서 굳습니다. 배포 해상도로 구우려면
  `optimize_for_inference.py` 출력에서 export하세요
- **카메라 2개가 그래프에 하드코딩**돼 있습니다(`PrefillGraph.forward` 시그니처). `smolvla_base`는 카메라
  3개를 선언하므로 거기서 export하면 조용히 2-카메라 그래프가 나오고 **`--verify`는 이걸 잡지 못합니다**
- 사전 빌드 바이너리는 저장소에 없습니다 (배포 해상도로 직접 export, ~1분)
- `onnx_runner.py`는 **벤치 하네스**입니다 — 토크나이즈/상태 정규화/액션 역정규화는 호스트 책임이고
  그 코드는 여기 없습니다

---

## 구조

```
configuration_smolvla.py    포크 전용 플래그 5개 + 경량 기본값 + draccus 인코더
modeling_smolvla.py         SmolVLAPolicy / VLAFlowMatching — 비전 배칭, posmap, dtype 캐스팅
smolvlm_with_expert.py      VLM + action expert — SDPA, KV 프로젝션 캐싱, lm_head 제거
processor_smolvla.py        전/후처리 파이프라인 (upstream과 동일)
_local_smolvla.py           설치된 lerobot 위에 이 폴더를 주입하는 import shim
benchmark.py                chunk 지연 측정 (제어율 판정은 하지 않음)
optimize_for_inference.py   경량 설정을 구운 배포용 체크포인트 생성
tests/                      등가성 검증 21체크 + 비전 배칭 (⚠️ 일부 MPS 전용)
experiments/                posmap · 스텝 수 · DCT A/B 연구 + 원시 수치 (⚠️ MPS 전용)
onnx_export/                2-그래프 ONNX export + numpy 러너
```

---

## 알려진 제약

- **포크 전용 플래그 5개는 저장되지 않습니다.** `attention_implementation`,
  `cache_expert_kv_projections`, `strip_lm_head`, `batch_vision_encoder`, `posmap_ref_resolution`은
  `draccus.encode` 훅에서 모든 dump(`config.json`·`train_config.json`)에서 제거됩니다.
  **stock lerobot 호환을 위한 의도된 동작**입니다(draccus는 미지의 필드에 하드 에러를 냅니다).
  재로드하면 항상 포크 기본값으로 돌아가므로, `posmap_ref_resolution`처럼 off가 기본인 것은
  **런타임에 지정**해야 합니다.
- **lerobot CLI는 자동으로 포크를 쓰지 않습니다.** `lerobot-record`/`lerobot-train`은 `lerobot.policies`를
  먼저 import하므로 `use_local_smolvla()`가 거부되고 조용히 stock 코드로 돌아갑니다. wrapper가 필요합니다:
  ```python
  import runpy, _local_smolvla; _local_smolvla.use_local_smolvla()
  runpy.run_module("lerobot.scripts.lerobot_record", run_name="__main__")
  ```
- **폐루프 태스크 성공률은 미검증입니다.** 모든 수치는 open-loop 액션 MAE와 지연입니다.
  액션 MSE가 비슷해도 성공률은 다를 수 있습니다 — **최종 판단은 실제 태스크 성공률로 하세요.**
- 실험 결과는 **체크포인트 1개 · 태스크 1개 · 1200스텝** 기준입니다.
- `experiments/runs/*/checkpoint.pt`(각 775MB)는 `.gitignore`에 있습니다. 재현하려면
  `experiments/train_ab.py`로 다시 학습해야 합니다.

## 되돌리기

- 완전 원본으로: 각 파일을 lerobot 저장소 원본(`src/lerobot/policies/smolvla/`)으로 교체
- 개별 기능만 (런타임에만 — 저장되지 않음): `attention_implementation="eager"`,
  `cache_expert_kv_projections=False`, `strip_lm_head=False`, `batch_vision_encoder=False`,
  `posmap_ref_resolution=None`, `num_steps=10`, `resize_imgs_with_padding=(512, 512)`

## 남은 옵션

1. **384px 파인튜닝** — 해상도 하락분 회복 (가장 확실한 다음 수)
2. **TensorRT 실측** — ONNX 그래프는 준비돼 있고 한 번도 안 돌려봤습니다.
   레거시 export의 shape 연산 꼬리(prefill 11,349 노드 중 Shape 932개) 때문에
   `polygraphy surgeon sanitize --fold-constants` 선행을 권장합니다
3. **GGUF + vla.cpp** — PyTorch 자체를 제거하는 최경량 경로

---

## 라이선스

원본 lerobot과 동일하게 **Apache License 2.0**입니다. 각 소스 파일의 헤더를 보세요.

## 인용

```bibtex
@article{shukor2025smolvla,
  title={SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics},
  author={Shukor, Mustafa and Aubakirova, Dana and Capuano, Francesco and Kooijmans, Pepijn and Palma, Steven and Zouitine, Adil and Aractingi, Michel and Pascal, Caroline and Russi, Martino and Marafioti, Andres and Alibert, Simon and Cord, Matthieu and Wolf, Thomas and Cadene, Remi},
  journal={arXiv preprint arXiv:2506.01844},
  year={2025}
}
```
