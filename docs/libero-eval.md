# LIBERO 성공률 평가

[`experiments/eval_libero.py`](../experiments/eval_libero.py)는 모델을 LIBERO 시뮬레이터에서 돌려
**작업 성공률**을 냅니다. 비동기 파이프라인에서 생기는 "묵은 시각 정보"도 흉내 낼 수 있습니다.

## 준비

Linux가 필요합니다(Windows는 WSL2). lerobot의 LIBERO 연동은 Linux에서만 설치됩니다.

```bash
pip install "lerobot[smolvla,libero]" "transformers>=5.3,<5.4"   # Python >= 3.12
pip install "mujoco==3.2.3"                                       # 아래 "시뮬레이터 버전" 참고
sudo apt install -y ffmpeg                                        # 학습 데이터 영상을 읽을 때 필요
```

- 처음 실행할 때 장면 데이터(로봇, 물체 모델)를 Hugging Face에서 자동으로 받습니다.
- 렌더링은 기본이 `MUJOCO_GL=egl`(NVIDIA 드라이버 사용)입니다. EGL이 안 되는 환경(WSL2에서 흔함)은
  `MUJOCO_GL=osmesa`로 CPU 렌더링을 쓰세요. 느리지만 동작합니다.

## 시뮬레이터 버전 (중요)

**mujoco는 3.2.3으로 맞추세요.** `pip install "lerobot[libero]"`가 받아 오는 최신 버전(확인한 것은
3.8.1)에서는 성공률이 실제보다 낮게 나옵니다.

LIBERO는 물체를 공중에서 떨어뜨려 장면을 만듭니다. 그래서 겹쳐 놓인 물체가 어떤 자세로 내려앉는지는
물리 엔진에 달려 있습니다. `libero_spatial` 작업 5(ramekin 위의 그릇)에서 확인한 차이는 이렇습니다.

| mujoco | 그릇이 내려앉은 위치 (ramekin 중심 기준) | 작업 5 성공 |
|---|---|---|
| 3.8.1 | 3.7cm 밀려 비스듬히 걸침. 학습 데이터에 없는 장면 | 0 / 120 |
| 3.2.3 | 1.4cm. 학습 데이터의 첫 장면과 같은 모습 | 3 / 3 (약 90스텝) |

3.8.1에서는 로봇이 그릇을 집어 접시까지 옮기지만, 접시 중심에서 4~5cm 벗어나게 놓아서 LIBERO의
성공 기준(3cm 이내)에 걸렸습니다. 두 버전 사이의 다른 버전은 시험하지 않았습니다.

평가 스크립트는 설치된 버전을 결과 파일의 `versions`에 기록하고, 3.2.3이 아니면 경고를 출력합니다.

## 실행 순서

**1. 연결 확인** (무작위 가중치, 1분 안쪽). 성공률은 0%가 정상이고, 끝까지 도는지만 봅니다.

```bash
python experiments/eval_libero.py --random --suites libero_spatial --task-ids 0 --episodes 1 --max-steps 20
```

**2. baseline 성공률.** 공개된 LIBERO 학습 체크포인트를 쓰므로 학습이 필요 없습니다.

```bash
python experiments/eval_libero.py --policy-path lerobot/smolvla_libero \
    --suites libero_spatial --episodes 50 --n-envs 5 --async-envs
```

**3. 묵은 정보의 영향.** 같은 체크포인트로 staleness만 바꿉니다.

```bash
python experiments/eval_libero.py --policy-path lerobot/smolvla_libero \
    --suites libero_spatial --stale-steps 0 5 10 --stale-align skip
```

**4. 해상도와 신선도의 맞교환.** 입력 해상도를 낮추면 그림은 거칠어지지만 vision이 빨리 끝나서
묵은 정도가 줄어듭니다. 어느 쪽이 이기는지 같은 체크포인트로 봅니다(학습 없음).

```bash
python experiments/eval_libero.py --policy-path lerobot/smolvla_libero \
    --suites libero_spatial --episodes 20 --n-envs 5 --async-envs --stale-align skip \
    --settings 512:off:0 512:off:5 384:off:0 384:on:0 384:on:3 320:off:0 320:on:0 320:on:2 \
    --out runs/libero_spatial_res.json
```

`해상도:posmap:묵은 스텝` 순서입니다. 묵은 스텝은 해상도별 vision 시간에 맞춘 값입니다
(Jetson에서 512px 260ms ≈ 5스텝, CPU에서 잰 비율로 384px ≈ 3스텝, 320px ≈ 2스텝).
`off:0` 줄들은 해상도만 낮췄을 때의 손실을, `on:0` 줄들은 posmap이 그 손실을 얼마나 되돌리는지를
보여 줍니다. 판정은 `512:off:5`와 `384:on:3`, `320:on:2`의 비교입니다.

## 옵션

| 옵션 | 의미 |
|---|---|
| `--stale-steps 0 5 10` | chunk를 계획할 때 쓰는 관측이 몇 스텝 전 것인지. LIBERO는 20Hz라 5스텝 = 250ms, 10스텝 = 500ms (Jetson 파이프라인의 vision 주기 1~2회분). 여러 값을 주면 차례로 평가 |
| `--settings 512:off:5 384:on:3` | `--stale-steps` 대신 구성을 하나씩 지정: 입력 해상도(64의 배수), posmap 켜기/끄기, 묵은 스텝. posmap은 낮은 해상도의 이미지 토큰을 학습 해상도의 위치 격자에 다시 놓습니다. 네 번째 칸으로 정렬 방식을 구성마다 줄 수 있습니다: `512:off:10:comp` |
| `--stale-align none` | chunk를 첫 action부터 실행 (계획이 d스텝 늦게 실행됨) |
| `--stale-align skip` | 이미 지나간 d개 action을 버리고 실행 (lerobot 비동기 추론 방식) |
| `--stale-align comp` | skip에 더해 drift 보정 (아래 "drift 보정") |
| `--stale-align fresh` | 카메라는 묵은 것, 로봇 state는 현재 값으로 계획하고 첫 action부터 실행 |
| `--comp-steps 5` | drift 보정을 몇 개 action에 나눠 넣을지 |
| `--settings 512:off:10/0:fresh` | 카메라마다 묵은 정도를 따로 지정 (LIBERO: 정면/손목 순서). 여기서는 정면 10스텝, 손목 0스텝. 로봇 state를 현재 값으로 쓰는 `fresh` 정렬에서만 가능 |
| `--num-steps` | denoise 스텝 수 (기본: 체크포인트 값) |
| `--n-action-steps` | chunk당 실행할 action 수 (기본: 체크포인트 값) |
| `--episodes`, `--n-envs` | 작업당 에피소드 수, 동시에 띄울 시뮬레이터 수 |
| `--async-envs` | 시뮬레이터마다 프로세스를 따로 띄워 CPU 코어를 하나씩 사용. `--n-envs`가 2 이상이면 켜는 것을 권장 |
| `--suites` | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` |

결과는 표로 출력되고 `runs/eval_libero.json`에 저장됩니다(작업별 성공률, 95% 신뢰구간, chunk
계획 시간 중앙값 포함). 파일은 작업 하나가 끝날 때마다 다시 쓰이므로, 중간에 멈춰도 그때까지의
결과가 남습니다.

여러 구성을 주면 작업마다 시뮬레이터를 한 번만 띄우고, 모든 구성이 **같은 초기 상태·같은 seed**로
같은 에피소드를 돕니다. 구성 사이의 차이가 에피소드 운에 덜 흔들립니다. 이 때문에 `--episodes`는
`--n-envs`의 배수여야 합니다.

빠른 비전 경로(PR #4의 `fast_vision.py`)가 붙은 정책은 느린 입력만 묵은 것을 받고,
빠른 경로는 항상 현재 관측을 받습니다.

## 왜 실패하는지 보기 (`--trace-dir`, `analyze_failures.py`)

성공률은 얼마나 자주 실패하는지만 알려 줍니다. 어떤 해법이 필요한지는 **어떻게** 실패하는지에
달려 있습니다. `--trace-dir`를 주면 평가 스크립트가 모든 에피소드의 매 스텝을 기록합니다:
손끝 위치, 보낸 action, 작업 물체들의 위치, 그 action 뒤의 관측이 얼마나 묵었는지, 그리고 묵은
chunk가 넘겨받는 순간의 위치 어긋남(drift).

```bash
python experiments/eval_libero.py --policy-path lerobot/smolvla_libero --suites libero_spatial \
    --episodes 20 --n-envs 5 --async-envs --n-action-steps 10 --stale-align skip \
    --settings 512:off:0 512:off:10 512:off:10:comp --trace-dir runs/traces --out runs/traced.json

python experiments/analyze_failures.py runs/traces --base "512px stale=0"
```

`analyze_failures.py`가 답하는 것은 세 가지입니다.

1. **구성마다 어떻게 끝나는가.** 에피소드를 위치 기록만으로 한 유형에 넣습니다.

   | 유형 | 기준 |
   |---|---|
   | never reached the target | 손끝이 목표 물체 7cm 안으로 들어간 적이 없음 |
   | reached, grasp failed | 다가갔지만 물체가 3cm 이상 들린 적이 없음 |
   | pushed the target away | 들리지 않은 채 옆으로 3cm 이상 밀림 |
   | dropped on the way | 들었지만 목표 지점 10cm 밖에서 끝남 |
   | still holding at timeout | 들고 있는 채로 시간이 끝남 |
   | placed off-centre | 목표 지점 10cm 안에 내려놓았지만 성공 판정(3cm)은 못 받음 |

2. **기준 대비 무엇을 잃는가.** 에피소드 i는 모든 구성에서 같은 초기 상태로 시작하므로, 기준은
   성공하고 이 구성은 실패한 에피소드가 곧 그 구성의 비용입니다. 그 에피소드들이 어느 유형으로
   실패했는지와 목록(suite, 작업, 에피소드 번호)을 출력합니다.
3. **묵은 chunk가 넘겨받을 때 계획에서 얼마나 벗어나 있는가.** drift를 명령된 이동 거리(cm)로
   환산합니다(action 1.0 = 제어 스텝당 5cm). `skip`처럼 보정하지 않는 구성에서도 잽니다.

유형 분류는 pick-and-place 작업(목표 물체가 첫 번째, 놓을 곳이 마지막 관심 물체)을 가정한
자동 1차 분류입니다. 논거로 쓰기 전에 목록의 에피소드를 `inspect_libero_task.py`의 사진과 대조하세요.

상태: 기록과 분류는 `tests/test_eval_libero.py`, `tests/test_analyze_failures.py`에서 확인했습니다.
실제 LIBERO 환경 클래스에 붙는 부분은 물리 엔진 자리에 대역을 넣고 lerobot의 실제 `LiberoEnv`
코드로 돌려 확인했고, 실제 시뮬레이터에서는 아직 안 돌렸습니다.

## 구성끼리 비교하기 (`compare_settings.py`)

같은 `--seed`로 돌린 구성들은 에피소드 i가 항상 같은 초기 상태에서 시작합니다. 그래서 두 구성을
성공률끼리가 아니라 **에피소드끼리** 비교할 수 있습니다. 두 구성의 결과가 갈린 에피소드만 세면
되므로, 같은 에피소드 수로 훨씬 작은 차이까지 가려낼 수 있습니다.

```bash
python experiments/compare_settings.py runs/spatial_mj323.json runs/spatial_mj323_cam.json \
    --base "512px stale=10"
```

구성마다 성공률과 95% 구간, 기준 대비 차이, 그리고 검정 결과가 나옵니다.

- 결과 파일에 에피소드별 성패(`episode_success`)가 있으면 McNemar 정확 검정을 씁니다.
  `lost`는 기준은 성공했는데 이 구성은 실패한 에피소드 수, `gained`는 그 반대입니다.
- 에피소드별 성패가 없는 예전 결과 파일은 작업별 성공률의 부호 뒤집기 검정으로 대신합니다.
  유효하지만 작업이 10개뿐이면 검출력이 약합니다.
- 정책, seed, 에피소드 수, `n_action_steps`, mujoco 버전이 다른 실행끼리는 같은 에피소드가 아니므로
  경고를 출력합니다.
- p값은 여러 구성을 한꺼번에 비교한 것에 대한 보정을 하지 않은 값입니다.

## 파이프라인 시간 재현 (`--pipelines`)

`--stale-steps`는 "항상 정확히 d스텝 묵은 관측"이라는 단순화입니다. 실제 비동기 파이프라인에서는
묵은 정도가 두 시간에서 나옵니다: 비전이 cache를 만드는 시간과, action 쪽이 chunk를 만드는 시간.
`--pipelines`는 이 두 시간을 받아 20Hz로 움직이는 로봇 옆에서 두 작업의 진행을 그대로 재현합니다.

```bash
python experiments/eval_libero.py --policy-path lerobot/smolvla_libero --suites libero_spatial \
    --episodes 20 --n-envs 5 --async-envs \
    --pipelines vision=260,action=430 vision=260,action=430,align=comp vision=260,action=86
```

| 옵션 | 의미 |
|---|---|
| `vision=MS` | 모든 카메라를 한 번 encode해서 prefix cache를 만드는 시간 |
| `action=MS` | cache 하나로 chunk 하나를 만드는 시간 (denoise 스텝 수 × 스텝당 시간) |
| `align=none\|skip\|comp` | chunk가 도착했을 때 이미 지나간 action의 처리 (기본 `skip`) |
| `sched=all\|rr` | 비전 작업 한 번에 모든 카메라(`all`)를 encode할지, 하나씩 번갈아(`rr`) 할지 |
| `state=cache\|job` | 로봇 state를 cache 시점 것으로 쓸지, action 작업을 시작할 때 읽을지 |

queue가 비면(에피소드 시작, chunk가 늦게 올 때) 로봇은 제자리에 멈춥니다. 결과 파일에는 구성마다
`action_age_ms_mean`(실행된 action 뒤의 관측이 평균 얼마나 묵었는지), `hold_fraction`(멈춰 있던
스텝의 비율), `expired_chunks`(도착했을 때 전부 지나가 버린 chunk 수)가 함께 기록됩니다.

단순화한 점: 두 작업이 GPU를 나눠 쓰며 서로 느려지는 것은 넣지 않았습니다. 두 시간은 **동시에 돌 때**
잰 값을 넣어야 합니다. 정책 계산 자체는 chunk가 준비되는 시점에, 그 작업이 썼을 프레임으로 합니다.

상태: 타이밍 계산은 `tests/test_eval_libero.py`에서 손으로 계산한 시나리오와 맞춰 봤습니다. 실제
시뮬레이터에서는 아직 안 돌렸습니다.

## drift 보정

묵은 관측으로 세운 계획은 "d스텝 전 로봇 위치에서 시작하는 경로"입니다. `skip`은 이미 지나간 앞의
d개 action을 버리는데, 그 d스텝 동안 로봇이 실제로 한 동작은 이전 계획에서 나온 것이라 새 계획이
가정한 위치와 실제 위치가 어긋납니다. action이 변위(delta)이므로 이 어긋남은 chunk가 끝날 때까지
그대로 남습니다.

그런데 두 동작 열은 모두 알고 있는 값입니다. 그래서 어긋난 양을 바로 계산할 수 있습니다.

```
drift = Σ_{i<d} (실제로 실행한 action_i − 새 계획의 action_i)      위치·회전 6차원만, 그리퍼 제외
보정된 action_k = 새 계획의 action_{d+k} − drift / m               k = 0 … m−1  (m = --comp-steps)
```

모델을 다시 부르지 않고, 학습도 필요 없습니다. 보정된 action은 action 범위 안으로 자르고, 다음 계획의
drift는 실제로 실행된 값으로 계산하므로 잘려서 못 넣은 보정이 누적되지 않습니다.

결과 파일의 `drift_mean`은 계획 한 번마다 보정한 위치 drift의 평균 크기(action 단위)입니다.

상태: `tests/test_eval_libero.py`에서 식대로 계산되는 것만 확인했습니다. 성공률을 되찾는지는 아직
재지 않았습니다.

## 한 작업만 계속 실패할 때

성공률만으로는 왜 실패하는지 알 수 없습니다. `inspect_libero_task.py`는 한 작업을 몇 에피소드만
돌리면서 두 가지를 기록합니다.

```bash
python experiments/inspect_libero_task.py --policy-path lerobot/smolvla_libero \
    --suite libero_spatial --task-id 5 --n-action-steps 10 --episodes 3 --out runs/inspect_task5
```

1. **시작 장면이 가만히 있는가.** 에피소드 시작 전의 대기 스텝 동안 1cm 넘게 움직인 물체를
   표시합니다. 쌓아 둔 물체가 미끄러지면 모델이 학습 때 본 적 없는 장면에서 시작하게 됩니다.
2. **에피소드에서 무슨 일이 일어나는가.** 그리퍼가 목표에 얼마나 다가갔는지, 무엇을 얼마나
   들어 올렸는지, 목표가 어디서 끝났는지를 숫자로, 에피소드 전체를 12장의 사진으로 남깁니다.

결과는 `--out` 폴더의 `report.txt`, `init_<k>.png`, `episode_<k>.png`입니다.

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
| 묵은 관측 전달, chunk 실행 순서, 성공 집계, 보고서, 해상도·posmap 구성 | ✅ `tests/test_eval_libero.py` (23개 항목, CI) |
| 실제 LIBERO 시뮬레이터에서 실행 | ✅ WSL2 Ubuntu 24.04, RTX 4060 Ti 8GB, `MUJOCO_GL=osmesa` |
| 공개 체크포인트의 성공률 재현 | ✅ `lerobot/smolvla_libero`, `libero_spatial` 전체 86.0% [81–90%] (mujoco 3.2.3, 10스텝마다 계획, n=200) |
| `--async-envs`를 실제 시뮬레이터에서 실행 | ✅ `--n-envs 5`로 같은 작업 8/10, 2분 24초 (첫 시도의 fork 방식은 시작 단계에서 멈춰서 spawn으로 교체) |
| `--settings`(해상도·posmap), 구성 간 같은 초기 상태 재사용 | ❌ 가짜 시뮬레이터로만 확인. 실제 시뮬레이터에서는 아직 안 돌림 |
| staleness에 따른 성공률 변화 | ✅ 아래 "측정 결과" |
| 해상도에 따른 성공률 변화 (학습 없이) | ✅ 아래 "측정 결과". 낮은 해상도의 손실이 신선도로 얻는 것보다 훨씬 큼 |

## 측정 결과

`lerobot/smolvla_libero`, `libero_spatial` 10개 작업 × 20 에피소드, mujoco 3.2.3, `--n-action-steps 10`,
`--stale-align skip`. 모든 구성이 같은 초기 상태와 seed를 씁니다. 괄호는 95% 신뢰구간입니다.

| 입력 | 묵은 정도 | 성공률 |
|---|---|---|
| 512px (학습 해상도) | 0 | 86.0% [81–90] |
| 512px | 5스텝 (250ms) | 78.5% [72–84] |
| 512px | 10스텝 (500ms) | 74.0% [68–80] |
| 384px + posmap | 0 | 58.0% [51–65] |
| 384px + posmap | 5스텝 | 52.5% [46–59] |
| 320px + posmap | 0 | 35.5% [29–42] |
| 320px + posmap | 4스텝 | 15.5% [11–21] |

- 묵은 관측은 성공률을 깎습니다: 250ms에 -7.5%p, 500ms에 -12%p.
- 학습 없이 해상도를 낮추면 그보다 훨씬 많이 잃습니다: 384px에서 -28%p, 320px에서 -50%p.
  "해상도를 낮춰 더 신선한 화면을 쓴다"는 방식은 이 체크포인트 그대로는 성립하지 않습니다.

mujoco 3.8.1에서 잰 이전 값(같은 조건에서 78.0 / 75.5 / 69.0%)은 작업 5가 항상 실패하는 환경의
값이라 쓰지 않습니다.

## 걸리는 시간

위 9/10 실행은 `--n-envs 2`(프로세스 하나에서 차례로 실행), CPU 렌더링으로 10 에피소드에
6분 23초가 걸렸습니다. 같은 속도라면 `libero_spatial` 전체(10개 작업 × 50 에피소드)는 약 5시간입니다.
시간의 대부분은 시뮬레이터와 렌더링이라서, `--async-envs`로 코어를 나눠 쓰면 줄어듭니다.
`--async-envs --n-envs 5`에서는 10 에피소드에 2분 24초(시뮬레이터 시작 포함)였습니다.

## 해상도 실험을 읽을 때 주의

LIBERO 카메라는 256×256으로 렌더링되고, 모델이 그것을 512로 키워서 봅니다. 그래서 384나 320으로
낮춰도 원본 화소는 버려지지 않고 토큰 수만 줄어듭니다. 실제 로봇의 640×480 카메라에서는 화소도
함께 줄어드니, 여기서 손실이 없다고 실기에서도 없다는 뜻은 아닙니다.
