# 빠른 비전 경로 (dual-rate SmolVLA)

비동기 파이프라인의 action 스레드에 **자기만의 싸고 새로운 시각 입력**을 주는 구조 변경입니다.
코드는 [`fast_vision.py`](../fast_vision.py), 검증은 [`tests/test_fast_vision.py`](../tests/test_fast_vision.py).

## 왜 필요한가

비동기 구조에서 action 스레드는 항상 **묵은 prefix KV cache**로 action을 만듭니다. Vision 스레드가
사이클 시작에 프레임을 찍고 260ms 뒤 cache를 공개하면, 그 cache는 다음 cache가 공개될 때까지
260ms 더 쓰입니다. 따라서 action 스레드가 보는 시각 정보의 나이는 **260~520ms**입니다
(프레임을 사이클 시작에 찍는다는 가정. 촬영 시점이 다르면 범위가 달라집니다).

측정상 chunk 시간의 87%가 SigLIP 비전 인코더라서, 비전을 더 자주 돌리는 방법으로는 해결이 안 됩니다.

## 구조

```
느린 경로 (변경 없음)  전방/상부 카메라 @512 → SigLIP → VLM prefill → PrefixCache      ~4 Hz
빠른 경로 (신규)       카메라 1대 @96x128 → ResNet-18 (layer3까지) → 48 토큰  ┐
                       현재 관절 상태 → 1 토큰                              ├─ gated cross-attn
                       noisy action 토큰 (timestep MLP 직후) ──────────────┘    → action expert
```

| 결정 | 이유 |
|---|---|
| 주입 위치는 suffix 임베딩 직후 | prefix/suffix attention 패턴이 그대로라 cached prefix, posmap, RTC, expert KV 캐시가 모두 그대로 동작 |
| Flamingo식 gated cross-attention, 게이트 0 초기화 | **붙인 직후 원본과 비트 단위 동일**. 사전학습 정책에서 출발해 도움이 되는 만큼만 학습. 토큰을 시퀀스에 그냥 붙이면 0 벡터라도 softmax 질량을 빼앗아 출력이 바뀜 |
| ResNet-18 layer3, 96x128 | 2.8M 파라미터, 480x640의 3:4 비율 유지, 6x8 토큰 격자. BatchNorm은 고정(배치 8의 통계는 노이즈) |
| 현재 관절 상태 토큰 1개 | prefix의 상태 토큰도 이미지만큼 묵어 있음 |
| chunk당 1회 인코딩 | 모든 denoise 스텝이 같은 빠른 토큰을 재사용 |

추가 파라미터는 9.25M(논문 모델 322M 대비 약 3%)이고, 빠른 인코딩은 CPU에서 4.5ms(느린 경로 999ms)입니다.

## 사용법

```python
from fast_vision import FastVisionConfig, attach_fast_vision

attach_fast_vision(policy, FastVisionConfig(camera_key="observation.images.front"))

prefix = policy.encode_slow(batch)          # vision 스레드, ~260ms마다
actions = policy.act_fast(prefix, batch)    # action 스레드, chunk마다
```

빠른 경로 카메라는 기존 카메라 중 하나를 재사용해도 됩니다(저해상도로 따로 읽음). 손목 카메라를
추가할 수 있다면 파지 마지막 구간에 더 유리합니다.

`encode_slow` / `act_fast`는 `VLAFlowMatching.encode_prefix` / `sample_from_prefix` 위에 있습니다.
이 둘은 기존 `sample_actions`를 나눈 것이고, 나눠 호출해도 결과가 비트 단위로 같습니다. 빠른 경로
없이도 비동기 파이프라인에 그대로 쓸 수 있습니다.

## 학습: staleness를 시뮬레이션해야 합니다

두 경로가 같은 프레임을 보면 빠른 경로에는 추가 정보가 없어서 모델이 그것을 무시하도록 학습됩니다.
`apply_staleness`는 샘플마다 **느린 경로(전 카메라 + 상태)를 t−d 프레임**에서, 빠른 경로와 정답
action은 t에서 가져옵니다. d는 배포 시 나이 범위(기본 8~16 프레임 @30fps)에서 뽑습니다.

```bash
python experiments/train_ab.py --variant baseline --steps 1000
python experiments/train_ab.py --variant fastvis --fast-camera observation.images.wrist --steps 1000
python experiments/eval_fastvis.py --baseline runs/baseline/checkpoint.pt --fastvis runs/fastvis/checkpoint.pt
```

`eval_fastvis.py`는 두 모델을 **같은 staleness 조건**(d = 0, 8, …, 16 프레임)에서 비교합니다.
d = 0은 동기식 상한이고, baseline이 d에 따라 얼마나 나빠지는지, fastvis가 그중 얼마를 되찾는지가
이 구조의 주장입니다.

학습 시 카메라마다 프레임을 6장씩(t + 과거 5개) 디코딩하므로 데이터 로딩이 느려집니다.
`--stale-offsets`로 개수를 줄일 수 있습니다.

## 검증 상태

| 항목 | 상태 |
|---|---|
| 분리 API = 기존 `sample_actions` (비트 동일), prefix 재사용 | ✅ `tests/test_fast_vision.py` |
| 붙인 직후 원본과 비트 동일 / 게이트가 열리면 빠른 이미지에 반응 | ✅ |
| 학습 시 게이트와 빠른 인코더까지 gradient 전달 | ✅ |
| staleness 변환 (과거 프레임 공유, 빠른 입력은 t) | ✅ |
| `train_ab --variant fastvis` → 저장 → `eval_fastvis` 재로드 전 과정 | ✅ 합성 데이터로 실제 `main()` 실행 |
| **실제 데이터 학습 / 정확도 개선 여부** | ❌ 아직 없음 — 이 환경에서 데이터셋 접근 불가 |
| Jetson 실측 | ❌ 아직 없음 |

## 알려진 제약

- `config.json`에는 빠른 경로가 기록되지 않습니다. 다시 로드하려면 정책을 만들고 같은
  `FastVisionConfig`로 `attach_fast_vision`을 호출한 뒤 state dict를 로드하세요
  (`eval_fastvis.load_fastvis`가 이 순서를 따릅니다).
- 한 모델 인스턴스에서 빠른 경로 조건부 호출을 동시에 두 개 실행하면 안 됩니다. 비동기 구조에서는
  action 스레드만 조건부 호출을 하고, vision 스레드는 `encode_prefix`만 호출합니다.
- `adapt_to_pi_aloha`와는 함께 쓸 수 없습니다(SO-101에는 해당 없음).
- 관련 연구: Helix / Fast-in-Slow(대형 dual-system), VLASH(미래 상태 conditioning), EagleVLA
  (Jetson 비동기 foresight). 이 구현의 차별점은 SmolVLA급 모델 + Orin Nano + KV 디커플링 설정입니다.
