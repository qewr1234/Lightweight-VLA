# 여러 해상도로 학습하기

[`experiments/train_multires.py`](../experiments/train_multires.py)는 학습된 체크포인트를 미세조정해서
**한 벌의 가중치가 여러 입력 해상도에서 동작**하게 만듭니다.

## 왜 필요한가

연산 시간의 대부분은 비전 encoder가 씁니다(논문 구성, CPU 기준 chunk 시간의 87%). 입력 해상도를
낮추면 그 시간이 크게 줄지만(이미지당 512px 476ms → 384px 226ms → 320px 158ms), 체크포인트는
학습한 해상도에서 가장 잘 동작합니다. 해상도를 그냥 낮추면 두 가지가 한꺼번에 바뀝니다.

1. 이미지 특징이 거칠어집니다.
2. 이미지 토큰 수가 줄면서 뒤따르는 모든 토큰의 위치 번호가 밀립니다.

2번은 posmap(`posmap_ref_resolution`)이 학습 없이 되돌립니다. 이 스크립트는 남은 1번을 학습으로
메웁니다.

## 방법

- 학습 스텝마다 해상도를 하나 뽑습니다. 해상도들이 같은 몫을 갖도록 묶음 단위로 섞습니다.
- 체크포인트의 해상도보다 낮을 때는 posmap을 켜서 위치 배치를 원래대로 둡니다.
- 비전 encoder와 VLM은 고정하고 action expert만 학습합니다(16층 기준 100M / 403M 파라미터).
  8GB GPU에 맞추기 위한 선택입니다.
- 결과는 `eval_libero.py --policy-path`로 바로 평가할 수 있는 체크포인트 폴더입니다.

## 실행 순서

LIBERO 데이터셋(`lerobot/libero`, 1.9GB)은 처음 실행할 때 자동으로 받습니다.

**1. 메모리와 속도 확인** (20스텝, 저장 안 함). `gpu_peak_gb`와 `sec_per_step`을 봅니다.

```bash
python experiments/train_multires.py --resolutions 512 384 320 --steps 20 --no-save
```

**2. 여러 해상도 학습과 대조군.** 대조군은 같은 데이터 순서, 같은 스텝 수로 원래 해상도에서만
학습합니다. 대조군이 없으면 "더 학습해서 좋아진 것"과 "여러 해상도로 학습해서 좋아진 것"을
구분할 수 없습니다.

```bash
python experiments/train_multires.py --resolutions 512 384 320 --steps 2000 --out runs/multires
python experiments/train_multires.py --resolutions 512 --steps 2000 --out runs/control512
```

**3. 평가.** 세 체크포인트(원본, 대조군, 여러 해상도)를 같은 구성으로 비교합니다.

```bash
for ckpt in lerobot/smolvla_libero runs/control512 runs/multires; do
  python experiments/eval_libero.py --policy-path $ckpt \
      --suites libero_spatial --episodes 20 --n-envs 5 --async-envs \
      --settings 512:off:0 384:on:0 320:on:0 --out runs/res_$(basename $ckpt).json
done
```

기대하는 모양: `runs/multires`가 384·320에서 원본과 대조군보다 높고, 512에서는 대조군과 비슷합니다.

## 옵션

| 옵션 | 의미 |
|---|---|
| `--resolutions 512 384 320` | 학습할 해상도(64의 배수, 체크포인트 해상도 이하) |
| `--steps`, `--batch` | optimizer 스텝 수, 배치 크기 (기본 2000, 8) |
| `--accum N` | N번 역전파한 뒤 한 번 갱신. 메모리는 그대로 두고 유효 배치를 N배로 |
| `--lr` | 최대 학습률 (기본 2e-5). 100스텝 warmup 후 cosine으로 1/10까지 감소 |
| `--save-every N` | N스텝마다 중간 저장 |
| `--video-backend pyav` | 기본 디코더(torchcodec)가 안 될 때 |
| `--policy-path` | 시작 체크포인트 (기본 `lerobot/smolvla_libero`) |

메모리가 부족하면 `--batch 4 --accum 2`처럼 배치를 줄이고 누적을 늘리세요.

## 검증 상태

| 항목 | 상태 |
|---|---|
| 고정 범위, 해상도 순서, 학습 스텝, 누적, 저장 후 다시 불러와 같은 action | ✅ `tests/test_train_multires.py` (17개 항목, CI). 무작위 가중치와 합성 데이터 |
| 실제 `lerobot/libero` 데이터셋 읽기 (영상 디코딩 포함) | ❌ 아직 안 돌림 |
| 8GB GPU에서의 메모리와 속도 | ❌ 아직 안 잼. 위 1번으로 확인 |
| 학습 후 낮은 해상도에서 성공률이 실제로 오르는지 | ❌ 아직 모름. 이 실험의 질문 자체 |

기본 학습률과 스텝 수는 근거 있는 값이 아니라 출발점입니다. 대조군이 512에서 원본보다 떨어지면
학습률이 높다는 뜻이니 낮춰서 다시 돌려야 합니다.
