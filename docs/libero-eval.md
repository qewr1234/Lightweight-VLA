# LIBERO 성공률 평가

[`experiments/eval_libero.py`](../experiments/eval_libero.py)는 모델을 LIBERO 시뮬레이터에서 돌려
**작업 성공률**을 냅니다. 비동기 파이프라인에서 생기는 "묵은 시각 정보"도 흉내 낼 수 있습니다.

## 준비

Linux가 필요합니다(Windows는 WSL2). lerobot의 LIBERO 연동은 Linux에서만 설치됩니다.

```bash
pip install "lerobot[smolvla,libero]" "transformers>=5.3,<5.4"   # Python >= 3.12
```

- 처음 실행할 때 장면 데이터(로봇, 물체 모델)를 Hugging Face에서 자동으로 받습니다.
- 렌더링은 기본이 `MUJOCO_GL=egl`(NVIDIA 드라이버 사용)입니다. EGL이 안 되는 환경(WSL2에서 흔함)은
  `MUJOCO_GL=osmesa`로 CPU 렌더링을 쓰세요. 느리지만 동작합니다.

## 실행 순서

**1. 연결 확인** (무작위 가중치, 1분 안쪽). 성공률은 0%가 정상이고, 끝까지 도는지만 봅니다.

```bash
python experiments/eval_libero.py --random --suites libero_spatial --task-ids 0 --episodes 1 --max-steps 20
```

**2. baseline 성공률.** 공개된 LIBERO 학습 체크포인트를 쓰므로 학습이 필요 없습니다.

```bash
python experiments/eval_libero.py --policy-path lerobot/smolvla_libero \
    --suites libero_spatial --episodes 50 --n-envs 5
```

**3. 묵은 정보의 영향.** 같은 체크포인트로 staleness만 바꿉니다.

```bash
python experiments/eval_libero.py --policy-path lerobot/smolvla_libero \
    --suites libero_spatial --stale-steps 0 5 10 --stale-align skip
```

## 옵션

| 옵션 | 의미 |
|---|---|
| `--stale-steps 0 5 10` | chunk를 계획할 때 쓰는 관측이 몇 스텝 전 것인지. LIBERO는 20Hz라 5스텝 = 250ms, 10스텝 = 500ms (Jetson 파이프라인의 vision 주기 1~2회분). 여러 값을 주면 차례로 평가 |
| `--stale-align none` | chunk를 첫 action부터 실행 (계획이 d스텝 늦게 실행됨) |
| `--stale-align skip` | 이미 지나간 d개 action을 버리고 실행 (lerobot 비동기 추론 방식) |
| `--num-steps` | denoise 스텝 수 (기본: 체크포인트 값) |
| `--n-action-steps` | chunk당 실행할 action 수 (기본: 체크포인트 값) |
| `--episodes`, `--n-envs` | 작업당 에피소드 수, 동시에 띄울 시뮬레이터 수 |
| `--suites` | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` |

결과는 표로 출력되고 `runs/eval_libero.json`에 저장됩니다(작업별 성공률, 95% 신뢰구간, chunk
계획 시간 중앙값 포함).

빠른 비전 경로(PR #4의 `fast_vision.py`)가 붙은 정책은 느린 입력만 묵은 것을 받고,
빠른 경로는 항상 현재 관측을 받습니다.

## 실제 파이프라인과 다른 점

여기서는 chunk를 `n_action_steps`마다 한 번, 정확히 d스텝 전 관측으로 계획합니다. 실제 로봇에서는
cache의 나이가 vision 주기 1~2회 사이에서 계속 변합니다. d를 여러 값으로 돌려 그 범위를 덮으세요.

## 8GB GPU (예: RTX 4060 Ti)

- **평가는 여유 있습니다.** 모델이 FP32로 약 1.6GB이고, 나머지는 시뮬레이터 몫입니다.
  메모리가 부족하면 `--n-envs`를 줄이세요.
- **학습은 설정에 따라 다릅니다.** 아래는 CPU에서 FP32로 잰 학습 1스텝의 메모리입니다(GPU에서는
  bf16 autocast를 쓰면 더 적게 듭니다).

| 설정 | 샘플당 메모리 | 고정 비용(가중치 + optimizer) | 8GB에서 |
|---|---|---|---|
| 논문 구성: 8층, expert만 학습 (51M) | 약 84MB | 약 1.7GB | 가능. batch 16부터 시작 |
| 16층, expert만 학습 (100M) | 약 118MB | 약 2.8GB | 가능. batch 16부터 시작 |
| 16층 전체 학습 (공개 체크포인트의 방식) | 더 큼 (미측정) | 약 6.4GB | 어려움 |

공개 체크포인트는 batch 32로 25,000스텝을 전체 학습한 것입니다. 그 재현은 8GB로 어렵지만,
체크포인트가 공개돼 있어서 다시 학습할 필요가 없습니다.

## 검증 상태

| 항목 | 상태 |
|---|---|
| 묵은 관측 전달, chunk 실행 순서, 성공 집계, 보고서 | ✅ `tests/test_eval_libero.py` (13개 항목, CI) |
| lerobot의 실제 전처리와 SmolVLA 정책으로 끝까지 실행 | ✅ 가짜 시뮬레이터로 |
| 실제 LIBERO 시뮬레이터에서 실행 | ❌ 아직 없음. 장면 데이터를 받을 수 없는 환경에서 작성함 |

실제 시뮬레이터 연결은 환경 생성 단계까지 확인했고, 장면 데이터를 읽는 지점에서 멈췄습니다.
위 1번 명령이 처음으로 끝까지 도는 것을 확인해 주세요.
