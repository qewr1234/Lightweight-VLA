# 경량 VLA 선행 연구는 모델을 줄였고 실행 계층을 남겼다

2026-11-16 기준으로 경량·비동기 VLA의 "모델 쪽"은 이미 포화 상태다. 작은 백본, 토큰 프루닝·캐싱, 양자화, flow 스텝 축소, 런타임 컴파일, dual-rate 구조가 150편 안팎의 논문으로 덮여 있고, SmolVLA가 Jetson Orin Nano 8GB에서 몇 ms에 도는지와 "Nano에서는 비전 인코더가 가장 큰 단계"라는 관찰도 vla.cpp와 커뮤니티 TensorRT 파이프라인에 이미 공개돼 있다. 여섯 후보 아이디어 가운데 **(5) MuJoCo 버전 민감도는 메커니즘·경계 버전(3.4.0)·3.72 cm 수치·수정안까지 GitHub 이슈에 공개돼 "이미 됨"**이고, (1) 카메라별 변화 기반 갱신, (2) 측정 지연 재생 평가, (4) 기기별 freshness-vs-fidelity 설정은 **조각만 존재하고 전체는 비어 있으며**, **(3) stale delta-action 청크의 실행 측 닫힌 형태 드리프트 보정만이 논문·라이브러리 어디에서도 일치 사례를 찾지 못한 항목**이다. 시뮬레이션 결과와 Jetson 지연 실측은 있으나 실로봇 실험이 없는 논문이 방어할 수 있는 빈칸은 세 가지로 좁혀진다: 실측 지연 트레이스를 비전/청크 생성 단계로 나눠 2-스레드 타임라인으로 LIBERO에 재생하는 평가, per-step delta 청크를 "관측 시점 기준 상대 궤적"처럼 동작하게 만드는 training-free 보정, 그리고 frozen policy에 대한 카메라별 staleness dose-response와 변화 기반 카메라별 갱신. 세 가지 모두 "재학습 없이 frozen 체크포인트에 적용"이라는 축에서 기존 비동기 캐시 연구(전부 학습 필요)와 갈린다. 다만 LIBERO가 quasi-static이라 효과 크기가 작고, 11월 11~12일에만 직접 관련 논문 두 편이 올라올 만큼 분야가 주 단위로 움직이므로, 아래에 명시한 논문들을 전문으로 읽기 전에는 "최초" 주장을 써서는 안 된다. 이 판정은 웹 검색과 arXiv 페이지 조회에 기반하며, 검색에서 나오지 않았다는 것이 존재하지 않는다는 증명은 아니다.

> **검증 표기.** 표기가 없는 주장은 조사 노트가 논문 전문(arXiv HTML), README, 원본 파일·소스코드에서 읽은 것이다. 다음 세 표기는 모두 **미검증**을 뜻한다: 〔초록〕 = 초록·학회 포스터 페이지만 읽음, 〔2차〕 = 검색 결과 요약이나 2차 자료에서만 확인, 〔요약〕 = 1차 페이지를 가져왔으나 자동 요약 도구를 거친 내용만 확보. 표기가 없는 수치도 자동 추출 도구를 거쳤으므로 논문에 인용하기 전에 PDF와 대조해야 한다. "프로젝트(미발표)"로 표시한 수치는 요청 프로젝트의 자체 측정값이며 문헌이 아니다.

## 경량화는 아홉 계열, 비동기 실행은 일곱 유형으로 갈리고 SmolVLA급은 그 틈에 놓여 있다

### 경량화 아홉 계열: 무엇을 줄이고, 학습이 필요한가

전용 서베이 두 편(2025-10)과 엣지 배치 서베이(2026-03), 생애주기 서베이(2026-11)를 1차 논문들과 합치면 경량화는 아홉 계열로 정리된다. 서베이들은 기법 유형으로만 분류하고 "어느 파이프라인 단계(비전 인코더 / prefix / action expert)가 실제로 빨라지는가"로는 분류하지 않으며, 자체 Jetson 측정이 없고, 전용 서베이 두 편은 Jetson을 언급조차 하지 않는다 ([arXiv 2510.24795](https://arxiv.org/abs/2510.24795), [arXiv 2510.17111](https://arxiv.org/abs/2510.17111), [arXiv 2603.16952](https://arxiv.org/abs/2603.16952), [arXiv 2611.18372](https://arxiv.org/abs/2611.18372)).

| 계열 | 줄이는 대상 | 대표 논문 | 보고된 이득 | 재학습 | 측정 하드웨어 |
|---|---|---|---|---|---|
| 1. 소형 백본·정적 레이어 제거 | 백본 크기, LLM 깊이 | SmolVLA ([2506.01844](https://arxiv.org/abs/2506.01844)), Evo-1 ([2511.04555](https://arxiv.org/abs/2511.04555)), NanoVLA ([2510.25122](https://arxiv.org/abs/2510.25122)), VLA-Adapter ([2509.09372](https://arxiv.org/abs/2509.09372)), Shallow-pi ([2601.20262](https://arxiv.org/abs/2601.20262)) | Shallow-pi: Jetson Orin에서 pi0 364 → 110 ms; VLA-Adapter 219.2 Hz | 필요 | 대부분 RTX 4090·H100; NanoVLA만 Orin Nano Super |
| 2. 시각 토큰 프루닝·병합·캐싱 | LLM에 들어가는 토큰·KV | VLA-Cache ([2502.02175](https://arxiv.org/abs/2502.02175)), SpecPrune-VLA ([2509.05614](https://arxiv.org/abs/2509.05614))〔요약〕, VLA-Pruner ([2511.16449](https://arxiv.org/abs/2511.16449))〔요약〕, LightVLA ([2509.12594](https://arxiv.org/abs/2509.12594))〔요약〕, TRIAGE ([2610.34519](https://arxiv.org/abs/2610.34519)) | 7B급에서 종단 1.35~1.99× | 대부분 불필요(LightVLA·CogVLA는 필요) | 대부분 데스크톱 GPU; TRIAGE는 Jetson AGX Orin |
| 3. 동적 깊이·early exit | 입력별 LLM 깊이 | DeeR-VLA ([2411.02359](https://arxiv.org/abs/2411.02359))〔초록〕, MoLe-VLA ([2503.20384](https://arxiv.org/abs/2503.20384))〔초록〕, DySL-VLA ([2602.22896](https://arxiv.org/abs/2602.22896))〔2차〕 | LLM 연산 5.2~6.5× 감소(LLM 부분만) | 전부 필요 | 초록에 미기재 |
| 4. 양자화·이진화 | 가중치·활성 비트폭 | QVLA ([2602.03782](https://arxiv.org/abs/2602.03782))〔초록〕, BitVLA ([2506.07530](https://arxiv.org/abs/2506.07530))〔초록〕, FoldQuantVLA ([2609.24433](https://arxiv.org/abs/2609.24433))〔초록〕, LoopQ ([2611.05093](https://arxiv.org/abs/2611.05093))〔초록〕 | 메모리 −43~−74%; 속도는 1.2~1.9× 또는 0·음수 | PTQ 불필요, QAT·복구 필요 | H100, AGX Orin, AGX Thor |
| 5. 행동 생성 가속 | 디코딩 방식, flow 스텝 수 | OpenVLA-OFT ([2502.19645](https://arxiv.org/abs/2502.19645))〔초록〕, SnapFlow ([2604.05656](https://arxiv.org/abs/2604.05656))〔초록〕, AdaVLA ([2608.29208](https://arxiv.org/abs/2608.29208))〔초록〕 | 10 → 1~2 스텝으로 3.3~3.7×; AdaVLA는 training-free 1.87~2.24× | 증류는 필요, 적응형 솔버는 불필요 | AdaVLA: AGX Orin |
| 6. dual-system | 느린 VLM과 빠른 정책의 주기 분리 | HiRT ([2410.05273](https://arxiv.org/abs/2410.05273))〔초록〕, Helix ([figure.ai](https://www.figure.ai/news/helix)), FiS-VLA ([2506.01953](https://arxiv.org/abs/2506.01953))〔초록〕 | 모델 가속이 아닌 제어율 분리(Helix 7~9 Hz / 200 Hz) | 필요 | 로봇 탑재 임베디드 GPU(기종 미공개) |
| 7. 증류 | 깊이·스텝 | Shallow-pi, SnapFlow〔초록〕 | 2× 이상 | 필요 | H100, Orin, Thor |
| 8. 런타임·컴파일 | eager PyTorch 오버헤드 | vla.cpp ([2606.08094](https://arxiv.org/abs/2606.08094)), realtime-vla ([2510.26742](https://arxiv.org/abs/2510.26742)), VLA-HWBench ([2610.34717](https://arxiv.org/abs/2610.34717)), VLA-Triage ([2611.06710](https://arxiv.org/abs/2611.06710)) | SmolVLA eager → compile 2.9~3.1× | 불필요 | Thor, AGX Orin, Orin Nano 8GB |
| 9. 비동기·지연 인지 실행 | 모델이 아니라 로봇의 대기 시간 | SmolVLA async, RTC ([2506.07339](https://arxiv.org/abs/2506.07339)), VLASH ([2512.01031](https://arxiv.org/abs/2512.01031)), EagleVLA ([2607.12659](https://arxiv.org/abs/2607.12659)) | 과제 완료 약 30% 단축, reaction latency 감소 | SmolVLA async·RTC는 불필요, 나머지는 fine-tune 또는 소형 모듈 | SO100, Orin·Thor(EagleVLA) |

2026년 중반부터는 방법이 아닌 "교정" 흐름이 추가됐다. 측정·평가 타당성 연구들이 보고된 가속이 하드웨어·실행 모드·폐루프 평가를 넘어 전이되지 않음을 보인다. 예컨대 컴파일은 SmolVLA와 pi0.5의 디노이징 스텝을 2배 이상 빠르게 만들어 "스텝 수를 줄여 얻는 절감"을 줄이고, 32개 flow 모델 비교 중 29개에서 누적 지연이 단독 측정으로 예측한 값보다 5.3% 넘게(최대 227%) 컸다 ([arXiv 2611.06710](https://arxiv.org/abs/2611.06710)).

SmolVLA급 모델에 중요한 사실은, 계열 2의 가속이 거의 전부 **비전 인코더 이후**에서 일어난다는 점이다. VLA-Cache는 LLM 디코더의 KV를 재사용할 뿐 인코더를 건너뛴다고 쓰지 않고, TTF-VLA는 매 프레임 인코더를 호출하며, SpecPrune-VLA·VLA-Pruner·LightVLA·DepthCache는 모두 인코더 출력 이후에서 토큰을 고른다 ([arXiv 2502.02175](https://arxiv.org/html/2502.02175v2), [arXiv 2508.19257](https://arxiv.org/html/2508.19257v3)〔요약〕, [arXiv 2603.10469](https://arxiv.org/html/2603.10469)〔요약〕). SmolVLA는 이미 프레임당 시각 토큰이 64개이고 LLM을 16층으로 잘랐으므로([arXiv 2506.01844](https://arxiv.org/abs/2506.01844)), 7B 모델에서 보고된 1.3~2× 이득이 그대로 옮겨오지 않는다는 것이 노트의 추론이다(어느 논문도 직접 주장하지 않음).

### 비동기 실행 일곱 유형: 지연을 어디서 다루는가

| 유형 | 지연 처리 위치 | 대표 | 학습 | 평가에서의 지연 모델 |
|---|---|---|---|---|
| A. 시스템 수준 큐 | 청크 예측과 실행을 분리, 큐 임계값으로 재요청 | SmolVLA async stack ([2506.01844](https://arxiv.org/abs/2506.01844)) | 불필요 | 실로봇 SO100 |
| B. 생성 측 연속성(학습 없음) | 디노이징 루프 안에서 이전 청크 잔여분에 맞춤 | RTC ([2506.07339](https://arxiv.org/abs/2506.07339)), SimpleRTC ([2602.12062](https://arxiv.org/abs/2602.12062)), FreeBar ([2610.14997](https://arxiv.org/abs/2610.14997)), Leap-VLA soft-nudge ([2611.01011](https://arxiv.org/abs/2611.01011)), PACT-VLA ([2610.28019](https://arxiv.org/abs/2610.28019)) | 불필요 | 고정 정수 d = 0~4 |
| C. 생성 측 조건화(학습) | 지연·실행된 행동을 모델 입력으로 | Training-time RTC ([2512.05964](https://arxiv.org/abs/2512.05964)), VLASH ([2512.01031](https://arxiv.org/abs/2512.01031)), REMAC ([2601.20130](https://arxiv.org/abs/2601.20130)), DEFLECT ([2605.19294](https://arxiv.org/abs/2605.19294)), Action ControlNet ([2606.25985](https://arxiv.org/abs/2606.25985)), FutureRTC ([2607.24008](https://arxiv.org/abs/2607.24008)) | 필요 | 고정 d |
| D. 실행 측 후처리 | 생성된 청크를 제어 루프에서 재색인·보정 | prefix drop(UMI [2402.10329](https://arxiv.org/abs/2402.10329), lerobot), spline 블렌딩(VLA-RAIL [2512.24673](https://arxiv.org/abs/2512.24673)), 색인 재매칭(Mobile UMI [2605.20894](https://arxiv.org/abs/2605.20894)), 학습된 residual(A2C2 [2509.23224](https://arxiv.org/abs/2509.23224)) | A2C2만 필요 | 실로봇 실측 또는 고정 d |
| E. dual-rate 캐시 분리 | 느린 비전·VLM 캐시 + 빠른 action expert | DAM-VLA ([2606.12105](https://arxiv.org/abs/2606.12105)), Leap-VLA, D2-VLA ([2609.34792](https://arxiv.org/abs/2609.34792)), DuoCore-FS ([2512.20188](https://arxiv.org/abs/2512.20188)), Acting While Understanding ([2606.15285](https://arxiv.org/abs/2606.15285)) | 전부 필요 | 선택한 갱신 주기 |
| F. 이벤트 트리거(관측 전체) | 변화 점수로 "지금 전체 추론을 돌릴지" 결정 | React When You Need To ([2609.22587](https://arxiv.org/abs/2609.22587)), StateMem ([2609.22684](https://arxiv.org/abs/2609.22684)), ElegantVLA ([2605.29438](https://arxiv.org/abs/2605.29438)) | 필요(소형 모듈·스케줄러) | 동기 실행 |
| G. 토큰 단위 재사용(동기) | 한 번의 forward 안에서 정적 토큰 KV 재사용 | VLA-Cache, TRIAGE ([2610.34519](https://arxiv.org/abs/2610.34519)), gated VLA-Cache ([2608.10824](https://arxiv.org/abs/2608.10824)) | 불필요 | 동기 실행 |

이 표에서 두 가지 빈 칸이 드러난다. 첫째, **"비동기 + 캐시된 prefix + training-free"는 조사 범위에서 빈 셀**이다. 유형 E의 모든 설계는 stale 캐시를 학습(지연 샘플링, age 조건화, stale-frame 커리큘럼, 어댑터)으로 견디게 만들고, training-free 캐시 방법(유형 G)은 전부 동기식이며 OpenVLA 계열에서만 검증됐다. 둘째, 유형 B의 gradient-free·training-free 방법들은 모두 **청크 경계 연속성**을 겨냥할 뿐 "청크를 계산하는 동안 누적된 변위"를 보정하지 않는다. Leap-VLA 부록 B는 잔여분과 새 계획이 실제로 충돌할 때 hard inpainting, guidance, soft-nudge 어느 것도 실행을 잔여분 쪽으로 끌어오지 못한다(2~3%를 넘지 못함)고 보고한다 ([arXiv 2611.01011](https://arxiv.org/html/2611.01011)).

### SmolVLA의 Orin급 지연 수치는 서로 다른 것을 재고 있다

노트 간에 SmolVLA-on-Orin 수치가 달라 보이지만, 런타임·지표 정의·카메라 수·전원 모드가 다른 측정이다. 아래 표가 그 정리이며, 같은 줄에 놓고 비교할 수 있는 수치는 없다.

| 출처 | 기기 | 런타임 | 지표 | 값 | 검증 |
|---|---|---|---|---|---|
| vla.cpp 논문 Table III ([2606.08094](https://arxiv.org/abs/2606.08094)) | Orin Nano 8GB | vla.cpp (ggml/CUDA) | 실행된 action당 시간(S=4), LIBERO-Object 성공 | 141.81 ms, 176/200 | 전문 |
| vla.cpp 논문 execution-horizon sweep | Orin Nano | vla.cpp | 호출당 median | 353~457 ms | 〔요약〕 |
| vla.cpp 저장소 보고서 ([jetson-orin-nano.md](https://raw.githubusercontent.com/VinRobotics/vla.cpp/main/docs/benchmark/jetson-orin-nano.md)) | Orin Nano Super, 2×512 px, 10 스텝 | vla.cpp 기본 / 최속 플래그 | 호출당 지연, 그중 비전 | 기본 p50 463.9 ms(비전 298.6) / 최속 플래그 286.7 ms(비전 173.1) | 〔요약〕 |
| 커뮤니티 HF model card ([eetmie/smolvla-base-onnx](https://huggingface.co/eetmie/smolvla-base-onnx)) | Orin Nano 8GB, MAXN_SUPER | ONNX Runtime + TensorRT FP16 | 호출당 p50, 비전, 디노이즈 10 스텝 | 189.89 ms, 91.2 ms, 59.3 ms | 〔요약〕 |
| 같은 저자의 GitHub README ([eetmie/jetson-orin-nano-vla](https://github.com/eetmie/jetson-orin-nano-vla)) | Orin Nano Super, JetPack 7.2.1 | 분할 TensorRT 엔진을 CUDA graph 하나로 재생 + Triton 커널 | 호출당 p50, 합성 관측 | 107.55 ms(9.30 Hz), RAM 1.78 GB | README 전문 |
| Shallow-pi ([2601.20262](https://arxiv.org/abs/2601.20262)) | Jetson Orin(Nano 아님) | 미기재 | 호출당 | 230 ms | 전문 |
| VLA-HWBench ([2610.34717](https://arxiv.org/abs/2610.34717)) | AGX Thor | PyTorch eager → torch.compile | 호출당 p50 | 191 → 66.4 ms | 전문 |
| 프로젝트(미발표) | Orin Nano 8GB, 자체 8-layer 모델 | 프로젝트 자체 스택 | 비전 시간 / 디코드 스텝 | 260 ms / 43 ms(VLM 16 → 8층 이전 85 ms) | 프로젝트 측정 |

"비전 인코더가 병목"이라는 전제도 기기와 스택에 따라 답이 달라진다. AGX Thor의 PyTorch eager에서는 action 생성이 187 ms 중 146.6 ms(78%)이고 비전은 20.9 ms다 ([arXiv 2610.34717](https://arxiv.org/abs/2610.34717)). 반면 Orin Nano에서는 vla.cpp 기본 설정에서 비전이 463.9 ms 중 298.6 ms, TensorRT에서도 189.89 ms 중 91.2 ms로 가장 큰 단일 단계다〔요약〕. 따라서 프로젝트의 260 ms 비전 시간은 공개 수치(173~299 ms)와 같은 범위에 있고, **"Nano에서 비전이 지배적"이라는 관찰 자체는 새 주장으로 내세울 수 없다.** 더 불편한 점은, 커뮤니티 TensorRT 파이프라인이 2-카메라 SmolVLA 전체를 약 105~190 ms에 돌린다는 것이다. 260 ms 중 일부는 아키텍처가 아니라 런타임 문제일 수 있으며, 리뷰어가 "TensorRT를 쓰면 되지 않느냐"고 물을 여지가 있다. 107.55 ms 파이프라인의 비전 시간과 해상도는 README에 적혀 있지 않아 이 추론은 확인이 필요하다.

## 문헌이 스스로 꼽는 미해결 문제는 "속도 수치와 성공률을 믿을 수 없다"로 모인다

2025년 서베이의 문제 진술은 "모델이 너무 크다"였고 구조적 간결함과 표현력 사이의 긴장, 고정 임계값에 의존하는 토큰 프루닝, 계층적·비동기 프레임워크의 불안정한 제어 신호를 꼽았다 ([arXiv 2510.24795](https://arxiv.org/abs/2510.24795), [arXiv 2510.17111](https://arxiv.org/abs/2510.17111)). 2026년 문헌은 무게중심을 측정과 평가의 신뢰성으로 옮겼다. 서로 독립된 출처에서 반복되는 미해결 문제는 여섯 가지다.

**첫째, 지연 보고 규약이 없다.** 감사 대상 25편 중 baseline과 방법 양쪽의 실행 모드를 밝힌 논문은 1편뿐이고, 가속의 크기·부호·순위가 GPU마다 바뀌며, 커널을 교체하는 비교 21건 중 6건이 플랫폼에 따라 가속에서 감속으로 뒤집혔다 ([arXiv 2611.06710](https://arxiv.org/abs/2611.06710)). 생애주기 서베이는 속도가 throughput, 청크당 지연, FPS, 유효 제어율처럼 서로 호환되지 않는 단위로 보고된다고 지적한다 ([arXiv 2611.18372](https://arxiv.org/abs/2611.18372)).

**둘째, 가속된 정책의 폐루프 검증이 빠져 있다.** 8-layer Octo 증류 모델이 모든 오프라인 검사를 통과하고도 폐루프에서 0/72(teacher 40/72)를 기록했고 ([arXiv 2609.23048](https://arxiv.org/abs/2609.23048))〔초록〕, 표준 LIBERO에서 손실이 없어 보이는 양자화 설정이 LIBERO-plus 섭동에서는 1.5~4.3점을 잃으며 ([arXiv 2611.05093](https://arxiv.org/abs/2611.05093))〔초록〕, 일곱 벤치마크 감사에서 버그 22개가 나와 "버그 수정이 방법 순위를 뒤집을 수 있다"고 보고됐다 ([arXiv 2609.37771](https://arxiv.org/abs/2609.37771))〔초록〕. 배치 지향 논문은 과제 성공을 아예 생략하는 일이 잦다. VLA-HWBench는 폐루프 성공을 측정하지 않으며 컴파일된 pi0.5 실행 하나가 그리퍼 명령을 뒤집었다고 적는다 ([arXiv 2610.34717](https://arxiv.org/html/2610.34717v2)).

**셋째, 가장 덜 압축된 부품이 최적화 이후의 병목이다.** 대부분의 가속이 VLM 추론 비용을 겨냥하고 반복 ODE 풀이를 다루지 못하며 ([arXiv 2608.29208](https://arxiv.org/abs/2608.29208))〔초록〕, 7개 VLA·4개 플랫폼 측정 연구가 다룬 양자화조차 언어 모델 선형층만 포함해 비전 인코더와 action head는 양자화되지 않으며 ([arXiv 2611.06710](https://arxiv.org/abs/2611.06710)), 최적화 뒤에는 CPU 전처리가 GR00T N1.7 호출의 39%까지 커진다 ([arXiv 2610.34717](https://arxiv.org/abs/2610.34717)).

**넷째, 비동기 실행은 속도를 얻고 정확도를 잃는다.** SmolVLA 자체 비동기 스택이 완료 시간을 약 30% 줄이면서 평균 성공률은 78.3%에서 73.3%로 낮추고 ([arXiv 2506.01844](https://arxiv.org/abs/2506.01844)), Jetson Orin에서의 naive async는 실제 접기 과제의 하위 과제에서 0/10으로 무너진다 ([arXiv 2607.12659](https://arxiv.org/html/2607.12659)). 비동기 방법 벤치마크는 스스로 "d를 고정 정수로 취급한다. 실제 배치에서 d는 GPU 스케줄링, 네트워크 지터, 입력 복잡도에 따라 변한다"는 한계를 적었다 ([arXiv 2605.08168](https://arxiv.org/html/2605.08168)).

**다섯째, 캐시 갱신이 고정 타이머에 묶여 있다.** DAM-VLA는 "카메라가 장면 변화가 아니라 고정 타이머로 갱신된다. 변화에 VLM을 트리거하는 검출기가 이 간극을 메울 것"이라고 쓰고 ([arXiv 2606.12105](https://arxiv.org/html/2606.12105)), Acting While Understanding은 적응형 semantic refresh를, Fast-VLA는 이벤트 트리거 메모리 갱신을 향후 과제로 든다 ([arXiv 2606.15285](https://arxiv.org/html/2606.15285v1), [arXiv 2610.21389](https://arxiv.org/html/2610.21389)). Leap-VLA는 트리거 경로를 설계에 넣고도 "보고된 어느 셀에서도 실행하지 않았다" ([arXiv 2611.01011](https://arxiv.org/html/2611.01011)).

**여섯째, 8GB급 기기는 메모리가 먼저 걸린다.** 3B급 모델은 vla.cpp에서 Orin Nano 메모리를 넘기고 ([arXiv 2606.08094](https://arxiv.org/abs/2606.08094)), 커뮤니티 저장소는 모델을 여러 TensorRT 엔진으로 쪼개야 8GB에서 빌드가 된다고 밝힌다 ([GitHub](https://github.com/eetmie/jetson-orin-nano-vla)). 엣지 배치 서베이는 이를 "모델 압축만의 문제가 아니라 시스템 문제"로 규정한다 ([arXiv 2603.16952](https://arxiv.org/abs/2603.16952)).

이 여섯 가지 가운데 넷째와 다섯째, 그리고 첫째·둘째의 교집합("지연을 실측하고 그것을 폐루프 성공에 연결")이 아래 후보 아이디어들이 겨냥하는 지점이다.

## 여섯 후보 중 하나는 이미 공개됐고, 셋은 조각만 있으며, 하나만 일치 사례가 없다

| # | 후보 아이디어 | 판정 | 가장 가까운 선행 | 남은 빈칸 한 줄 요약 |
|---|---|---|---|---|
| 1 | 카메라별 변화 기반(AoII) 비전 캐시 갱신 | **부분적으로 됨**(조각만 존재, 전체는 열림) | DAM-VLA, Leap-VLA, TRIAGE, StateMem, LIBERO-VPro | 카메라 단위 갱신, N-way 중재, AoII 정식화, training-free 비동기 |
| 2 | 실측 엣지 지연 재생 폐루프 평가 + training-free 수정 조합 | **부분적으로 됨** | arXiv 2605.08168, EagleVLA, TIDAL, RoboPAD 워크숍 논문 | 단계별·가변 트레이스 재생, 2-스레드 타임라인, 실행 측 수정의 조합 |
| 3 | stale delta-action 청크의 실행 측 드리프트 보정 | **열림**(일치 사례 미발견, 인접 연구는 다수) | UMI, Mobile UMI, VLA-RAIL, PPC, A2C2 | per-step delta 청크용 닫힌 형태 (executed − planned) 오프셋 |
| 4 | 기기별 freshness-vs-fidelity 설정 | **부분적으로 됨** | vla.cpp, FASTER, "Same Scene, Different Answer", DASH | 기기별 최적점 이동의 실증, 해상도 축, 선택 규칙 |
| 5 | LIBERO의 MuJoCo 버전 민감도 | **이미 됨**(회색 문헌) | lerobot #4390, LIBERO #141, drift atlas | 3.2.3 데이터 포인트, 데이터셋 대비 비교 정도만 남음 |
| 6 | 비전 인코더가 병목인 소형 VLA의 남은 방향 | **방향별로 열림~부분** | arXiv 2607.12287, ActPrune, STC-Cacher, EdgeVLA-Base | 아래 다섯 방향 |

### (1) 카메라별 변화 기반 갱신: 조각 넷이 따로 있고 합친 논문은 없다

찾은 모든 비동기·dual-rate VLA는 시각-의미 캐시를 **모든 카메라에 대해 한꺼번에**, 고정 주기 또는 "직전 느린 pass가 끝나는 즉시" 갱신한다. 카메라가 언급되는 경우에도(DAM-VLA, D2-VLA, DuoCore-FS, Leap-VLA) 느린 pass 전에 이어 붙인다.

| 논문 | 실제로 하는 일 | 학습 | 하드웨어·벤치마크 | 빠진 것 |
|---|---|---|---|---|
| DAM-VLA ([2606.12105](https://arxiv.org/html/2606.12105)) | 모달리티별 latent 버퍼; VLM 인코더는 추론 4 스텝마다, proprio·F/T는 매 제어 스텝 | 필요(X-VLA 기반 fine-tune) | RTX 4060 Ti 추론; Franka 실제 과제 7종, 95.2% 대 40.95% | 비전은 단일 버퍼·고정 타이머; 변화 트리거는 저자가 직접 향후 과제로 지목 |
| Leap-VLA ([2611.01011](https://arxiv.org/html/2611.01011)) | VLM prefix 캐시(약 1 Hz) + cache age로 조건화한 flow expert + 약 23M 고속 시각 인코더(카메라당 196 토큰) | 필요(age 조건화, stale-frame 커리큘럼) | RTX 4090D; LIBERO 4종 평균 1 s 갱신 95.2%, 2 s 94.5% | 트리거 경로는 설계만 있고 미실행; 느린 캐시의 카메라별 갱신 없음; 스케줄러 없음 |
| TRIAGE ([2610.34519](https://arxiv.org/html/2610.34519)) | 토큰별 drift(< 0.3)·age(≤ 10 쿼리)·attention 중요도로 재계산/KV 재사용/프루닝 선택, 스텝당 예산 | 불필요 | OpenVLA·OpenVLA-OFT; LIBERO 73.7% 대 76.5%; AGX Orin prefill 약 0.52× | 토큰 단위·동기식; 비용 항과 카메라 간 중재 없음; flow-matching 모델 미검증 |
| StateMem ([2609.22684](https://arxiv.org/html/2609.22684)) | prefix KV를 anchor로 캐시, 예측 오차가 EMA 임계값을 넘으면 prefix 갱신 | 필요(A6000 4장, 약 5일) | pi0.5 초기화; LIBERO 97.6%, 갱신률 79.75% | RGB 3뷰를 평균 풀링해 전부-아니면-전무 결정 |
| React When You Need To ([2609.22587](https://arxiv.org/html/2609.22587v1)) | 이벤트 디코더가 anchor 특징과 현재 특징을 비교해 전체 추론을 트리거 | 필요(pi0.5 LoRA + 디코더) | Franka 실제 과제 2종, 평균 95% | 카메라 3대지만 카메라별 트리거 없음; KV 캐시 없음 |
| LIBERO-VPro ([2609.24350](https://arxiv.org/html/2609.24350)), Blackout vs. Freeze ([2609.39145](https://arxiv.org/html/2609.39145)) | 한 카메라만 stale·freeze, 다른 카메라는 최신 | 해당 없음(벤치마크) | LIBERO 40과제; pi0.5 정상 97.2 / third-person stale 21.35 / wrist stale 19.70 | 방법이 아님; 가혹한 조건만(지연 크기 미확보), 작은 age의 dose-response 없음 |
| AoI/AoII 이론 ([2404.16281](https://arxiv.org/abs/2404.16281), [2509.01926](https://arxiv.org/abs/2509.01926), [2606.30623](https://arxiv.org/abs/2606.30623))〔초록〕 | 다중 소스 원격 추론의 Whittle index, 상관 소스 스케줄링, "쉬거나 센서 하나를 pull" | 해당 없음 | 일반 센서 모델 | 학습된 정책의 캐시나 로봇 카메라에 적용한 사례 없음 |

정확한 잔여 빈칸은 다섯 가지다. 카메라 A의 캐시를 유지하면서 카메라 B만 재인코딩하는 **카메라 단위 갱신**, "어느 카메라를, 또는 아무것도 안 할지"를 변화량·중요도·비용으로 정하는 **N-way 중재**(기존 트리거는 전체 관측에 대한 1비트 결정), 정책의 지각 캐시에 대한 **AoI/AoII 정식화**, frozen prefix-cache VLA에서의 **training-free 비동기 동작**, 그리고 작은 age 구간(1~10 제어 스텝)에서의 **카메라별 dose-response 데이터**다.

이 아이디어의 위험은 숨기지 말아야 한다. Frozen policy는 단일 카메라 staleness에 취약하다. LIBERO-VPro에서 6개 모델 중 5개가 wrist 뷰 섭동에 더 민감했고, 카메라를 고정(freeze)하면 pi0.5 성공률이 93.1%에서 5.8%(scene)·3.6%(wrist)로 떨어진다 ([arXiv 2609.39145](https://arxiv.org/html/2609.39145)). pi0.5에서 prefix KV 전체를 재사용한 유일한 training-free 시도는 0/1, 0/2로 실패했는데 표본이 1~2 에피소드라 일화 수준이다 ([arXiv 2605.21854](https://arxiv.org/html/2605.21854)). 450M flow-matching VLA에서 비전 prefix 캐싱이 24개 움직임 조건 전부에서 INT8 양자화보다 큰 action 오차를 냈다는 결과도 있다(개루프·합성 자극) ([NeurIPS 2026 포스터](https://neurips.cc/virtual/2026/170035))〔초록〕. 고정 임계값이 로봇 suite 간에 잘 전이되지 않는다는 보고(ElegantVLA)와 백본별 보정이 필요하다는 보고(TRIAGE)도 training-free 지수의 실용 위험이다. 또한 flow-matching VLA에서는 디노이징 루프가 지연을 지배하므로, 이득은 "더 빠른 action 생성"이 아니라 "비전 연산 단위당 더 낮은 캐시 age"로 표현해야 한다. 세 그룹이 6~11월 사이에 변화 트리거 갱신을 향후 과제로 지목했으므로 전체 관측 단위 버전은 곧 나올 가능성이 높고, 선행이 가장 얇은 부분은 카메라별 중재와 AoII 지수 프레이밍이다.

### (2) 지연 충실 폐루프 평가: 고정 정수 d가 표준이고, 단계별 트레이스 재생은 없다

문헌의 사실성은 세 수준에 머문다. 거의 모든 연구가 Kinetix나 LIBERO에 **고정 정수 d**를 주입하고, 일부가 기기 사양이나 측정에서 얻은 **상수 지연**을 스케줄로 바꾸며, 워크숍 논문 한 편이 이벤트 기반 executor에서 **요청별 측정 지연**을 쓴다. 비전 인코딩 시간과 청크 생성 시간을 분리하고 호출마다 변하는 엣지 기기 트레이스를 2-스레드 타임라인으로 재생한 연구는 찾지 못했다.

| 논문 | 실제로 하는 일 | 학습 | 하드웨어·벤치마크 | 빠진 것 |
|---|---|---|---|---|
| "Understanding Asynchronous Inference Methods for VLA Models" ([2605.08168](https://arxiv.org/html/2605.08168)) | IT-RTC·TT-RTC·VLASH·A2C2를 통일 코드로 비교; d 스텝 전 관측을 넣어 지연 모사 | IT-RTC만 불필요 | SmolVLA, LIBERO 4종, H=50·s=50, d ≤ 20; 지연은 RTX 3090에서만 측정(naive 405.2 ms) | 고정 정수 d; 측정 지연을 평가에 되먹이지 않음; 실행 측 training-free 방법·조합 없음 |
| EagleVLA ([2607.12659](https://arxiv.org/html/2607.12659)) | Jetson Orin·Thor에서 ViT/LLM/action expert 지연 실측, LIBERO Δ=1~9 sweep, Δ_ae = ceil(Δ/3) | 필요(약 40M future-correction 모듈) | pi0.5; Orin-50W 152.3/631.0/536.8 ms; 실로봇 XR-1 | 정수 고정 지연과 3:1 고정 비율; 경량 VLA 아님; training-free 수정 아님 |
| TIDAL ([2601.14945](https://arxiv.org/html/2601.14945)) | 추론 중 물리가 진행되는 "Non-Paused" 평가 | 필요 | GR00T-N1.5; AGX Orin 지연은 공개 수치(413 ms 주기) 차용; RoboCasa 기반 자체 과제 | 재생 트레이스가 아닌 상수; LIBERO 아님 |
| "When Delay Meets Distribution Shift" ([NeurIPS 2026 RoboPAD](https://neurips.cc/virtual/2026/162917))〔초록〕 | discrete-event executor에서 요청별 측정 지연 + 논리 지연 200 ms | 불필요(naive async, RTC) | pi0.5, LIBERO / LIBERO-Plus | 엣지 기기·단계 분리·가변 트레이스 재생 여부가 초록에 없음(전문 미확인) |
| DASH ([2611.11854](https://arxiv.org/html/2611.11854))〔요약〕 | 지연을 시뮬레이션 시간에 부과 | 필요 | RoboTwin 50과제; RTX 4090 지연만 | 아키텍처 정당화용; 기기 축·training-free 수정 없음 |
| PACT-VLA ([2610.28019](https://arxiv.org/html/2610.28019)) | 단계별 결정 주기 실측(prefill 23.2 ms + flow 24.9 ms, pre-commit prefill 병행)과 기존 방법 위 조합 | 불필요 | RTX 5090; Kinetix·LIBERO | 시뮬레이션은 고정 d; 데스크톱 GPU |

남은 빈칸은 네 단계로 쓸 수 있다. (i) 실제 엣지 기기에서 호출별 지연 트레이스를 비전 인코딩과 청크 생성으로 나눠 기록하고, (ii) 상수 d가 아닌 가변 트레이스를 LIBERO 안의 discrete-event 2-스레드 executor로 재생하며, (iii) 경량 VLA의 성공률을 그 타임라인의 함수로 보고하고, (iv) training-free 수정을 단독·조합으로 평가한다. (i)+(ii)의 단계 분리·지터와 (iv)의 실행 측 수정은 선행을 찾지 못했고, (iii)은 EagleVLA의 pi0.5 고정 비율 버전만 있다. arXiv 2605.08168 자체가 SmolVLA에서 비전 prefill이 청크 연산의 89.8%(FLOPs)이고 "원칙적으로 action 실행과 파이프라인화할 수 있다"고 적어, 아무도 폐루프로 평가하지 않은 2-스레드 타임라인을 직접 가리킨다.

신규성 위험은 포지셔닝이다. 리뷰어는 arXiv 2605.08168, EagleVLA, RoboPAD 워크숍 논문을 들이댈 수 있다. 방어 가능한 주장은 "최초의 LIBERO 지연 벤치마크"가 아니라 **"실측·가변·단계 분해 타임라인이 고정 d 순위와 무엇을 다르게 만드는가"**다. 또 LIBERO 수치는 논문 간 비교가 되지 않는다. H, execution horizon, 체크포인트가 달라 같은 SmolVLA의 naive async가 d=20에서 약 10~12%(H=50, s=50)로도, 56.2%(FutureRTC 설정)로도 보고된다 ([arXiv 2607.24008](https://arxiv.org/html/2607.24008)).

### (3) 실행 측 드리프트 보정: 일치하는 선행 사례를 찾지 못한 유일한 항목

"stale delta-action 청크의 앞 d개 action을 버리고, 그 구간의 (실행된 − 계획된) action 차의 합을 이후 action들에 나눠 빼는" 방법을 기술한 논문·저장소·문서는 찾지 못했다. 가장 가까운 선행은 세 갈래의 부분 겹침이다.

| 선행 | 실제로 하는 일 | 학습 | 하드웨어·벤치마크 | 후보와 다른 점 |
|---|---|---|---|---|
| UMI ([2402.10329](https://arxiv.org/html/2402.10329)) | 지연으로 낡은 앞부분 action을 버림; 청크는 관측 시점 EE 자세 기준의 relative trajectory | 불필요 | 실로봇; latency matching 제거 시 던지기 87.5% → 57.5% | 절대 기준이 있는 표현이라 암묵적으로 재고정됨; per-step delta에는 적용 안 됨 |
| openpi / lerobot ([openpi config](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py), [lerobot action_queue.py](https://github.com/huggingface/lerobot/blob/fe1c54594f6ba275fccf210b9f7b11f0e2324b89/src/lerobot/policies/rtc/action_queue.py)) | state-relative 청크를 요청 시점 state로 절대값 변환; RTC 큐는 `real_delay`개를 버리기만 함 | 불필요 | 라이브러리 | openpi 설정은 "LIBERO의 원시 action은 이미 delta"라 변환을 생략한다고 명시 — LIBERO 청크에는 절대 기준이 없음 |
| Mobile UMI ([2605.20894](https://arxiv.org/html/2605.20894)) | increment 청크를 관측 시점 state에서 전방 적분, 현재 측정 state와 가장 가까운 색인 i*를 골라 그 앞을 폐기 | 불필요 | 실로봇 10 Hz, 지연 142 ± 18 ms; 보정 제거 시 83.8% → 60.0% | 측정 state와 수작업 가중 metric 필요; 오프셋이 아니라 **색인**을 바꿈; 잔여 오프셋은 미보정 |
| VLA-RAIL ([2512.24673](https://arxiv.org/html/2512.24673)) | 지연 보상 색인 계산 후 dual-quintic spline으로 블렌딩 | 불필요 | AgiBot G1, 노트북 RTX 4080 | 연속성만; executed-vs-planned 항 없음 |
| PPC ([2605.11459](https://arxiv.org/html/2605.11459)) | delta EE 청크에 닫힌 형태 보정을 실행 구간에 점진적으로 분산 | 불필요 | 자체 MoveBench(ManiSkill, 20 Hz) | 입력이 외부 물체 속도 신호; 추론 지연 보정이 아님 |
| A2C2 ([2509.23224](https://arxiv.org/html/2509.23224)) | 매 제어 스텝 최신 관측으로 stale action에 residual을 더하는 correction head | 필요 | SmolVLA, LIBERO-Spatial 64.4% → 84.2%(e=40, d=10) | 학습된 head; 최신 관측 필요 |
| VLASH·FutureRTC·Action ControlNet·RAPAC-DP ([2512.01031](https://arxiv.org/html/2512.01031), [2607.24008](https://arxiv.org/html/2607.24008), [2606.25985](https://arxiv.org/html/2606.25985), [2608.15924](https://arxiv.org/html/2608.15924)) | 추론 중 실행된 action의 합을 **모델 입력**으로 사용 | 필요 | Kinetix, LIBERO, 실로봇 | 같은 양을 출력단에서 닫힌 형태로 쓰지 않음 |

후보의 수학적 위치는 분명하다. 절대 또는 state-relative 목표에서는 앞 d개를 버려도 남은 목표가 의도한 절대 경로 위에 있지만, per-step delta 청크에서는 앞 d개를 버리면 남은 경로 전체가 그 d 스텝 동안의 Σ(executed − planned)만큼 밀리고, 조사한 어떤 시스템도 그 밀림을 제거하지 않는다. 이 방법은 action 로그만 쓰므로 state와 action 차원이 맞지 않는 LIBERO(9-D state, 7-D action — arXiv 2605.08168이 VLASH roll-forward가 "불가능"해 oracle 미래 state를 썼다고 밝힌 바로 그 문제)에서도 동작한다.

위험도 분명하다. 메커니즘이 단순해 "알려진 엔지니어링 요령"이라는 반응이 나올 수 있고, action 합이 변위와 같다는 가정은 추종이 좋은 position형 delta 제어에서만 성립한다(VLASH와 arXiv 2605.08168 모두 velocity·torque 제어에서 roll-forward가 근사라고 적음). 고전 원격조작·궤적 추종 문헌과 산업용 배치 스택은 검색하지 않았으며, LIBERO 평가 하네스 가운데 delta 청크를 실행 전에 절대 목표로 조용히 변환하는 것이 있는지도 확인하지 않았다. "일치 사례 미발견"은 그 범위 안에서의 판정이다.

### (4) 기기별 freshness-vs-fidelity 설정: 재료는 대부분 2026년에 나왔고, 기기별 최적점 이동만 없다

| 논문 | 실제로 하는 일 | 학습 | 하드웨어·벤치마크 | 빠진 것 |
|---|---|---|---|---|
| vla.cpp ([2606.08094](https://arxiv.org/html/2606.08094v2)) | Orin Nano에서 SmolVLA execution-horizon sweep: S=1 82%, S=16 98%, S=50 38%; GR00T N1.7 flow 스텝 sweep(성공률 평탄)〔요약〕 | 불필요 | Orin Nano·RTX 3060; LIBERO-Object, 설정당 50 에피소드 | 해상도 미변경; 시뮬레이터에 지연을 부과하지 않음; 기기 의존성은 시험하지 않았다고 명시 |
| "Same Scene, Different Answer" ([NeurIPS 2026 포스터](https://neurips.cc/virtual/2026/178329))〔초록〕 | 디노이징 스텝 × execution horizon 4×3 요인 설계 + 측정 지연; pi0.5는 1 스텝이 10 스텝과 구분되지 않고 지연은 2.7× 낮음 | 불필요 | pi0.5·pi0, LIBERO; 하드웨어 미기재 | 기기 축·해상도 없음 |
| FASTER ([2603.19199](https://arxiv.org/html/2603.19199v3)) | 반응 시간 모델, 기기 의존 최소 실행 지평 s_min = ⌈Δt_infer/Δt_ctrl⌉(RTX 4090에서 3, RTX 4060에서 10)〔요약〕 | 초록상 불필요(실로봇 체크포인트의 fine-tune 여부 미확인) | 실로봇 탁구; 4090 대 4060 점수 0.800 대 0.467〔요약〕 | 설정 선택이 아니라 샘플링 스케줄; 스텝·지평 대 성공률 sweep 없음 |
| EagleVLA ([2607.12659](https://arxiv.org/html/2607.12659)) | 기기 실측 지연으로 L = H − Δ 결정 | 필요 | pi0.5; Orin 대 RTX 4090 실로봇 | 스텝·해상도·청크 고정 |
| "What Makes a VLA Reactive?" ([2611.15491](https://arxiv.org/abs/2611.15491))〔요약〕 | 예측·실행 지평과 sync/async의 체계적 비교; "지연이 낮다는 것만으로는 도움이 되지 않았다" | 해당 없음(실행 전략 비교 연구) | pi0.5; A100·RTX 5090 | 지연이 통제 변수가 아님 |
| VLA-HWBench ([2610.34717](https://arxiv.org/abs/2610.34717)), VLA-Perf ([2602.18397](https://arxiv.org/html/2602.18397)) | 기기별 지연 모델(스텝에 선형, 청크 크기는 거의 무료) | 불필요 | Thor·AMD APU / 해석 모델 | 성공률 없음; VLA-Perf는 staleness와 성공의 연결을 향후 과제로 남김〔요약〕 |
| HF 블로그 ([Zuo, 2026-06-28](https://huggingface.co/blog/zuoxingdong/smolvla-libero-spatial-eval))〔요약〕 | SmolVLA flow 스텝 수 대 성공률: 1 스텝 80.5%, 2 스텝 74.0%, 4 스텝 66.0%, 10 스텝 65.5%(n=200, nas=10) | 불필요 | `HuggingFaceVLA/smolvla_libero`, LIBERO-Spatial | 동료 평가 없음; 프로젝트가 쓰는 `lerobot/smolvla_libero`와 다른 체크포인트 |

찾지 못한 것은 (a) 각 (스텝, 해상도, 청크, 재계획) 설정의 지연을 기기에서 재고, (b) 그 지연을 폐루프에 부과하고, (c) 성공률 최대 설정이 기기마다 다름을 보인 논문이다. 문헌이 주는 것은 최적점이 아니라 실행 가능성 한계뿐이다: d ≤ s ≤ H − d(RTC), g ≥ (E[ℓ]/Δt)/n(SmolVLA), s_min(FASTER). **해상도 축은 폐루프 증거가 전혀 없다.** OpenVLA의 224 대 384 px 비교는 해상도별로 따로 학습한 것이고 ([arXiv 2406.09246](https://arxiv.org/html/2406.09246))〔요약〕, 학습 해상도보다 낮춰 추론했을 때의 성공률 하락을 보고한 VLA 논문은 찾지 못했다.

세 가지는 주장할 수 없다. 첫째, "느리고 정확한 설정은 더 낡은 관측에 작용하므로 개루프가 아니라 폐루프로 평가하라"는 정성적 진술은 OpenVLA(int8이 1.2 Hz로 느려져 58.1% 대 71.3%)〔요약〕, RTC, VLASH, FASTER, DASH에 이미 있다. 둘째, 스텝 축에서는 "fidelity" 쪽이 사라지거나 뒤집힐 수 있다. pi0.5와 GR00T N1.7에서 스텝 수는 성공률을 거의 움직이지 않고, 위 HF 블로그에서는 SmolVLA가 스텝이 적을수록 오히려 높다. 그렇다면 결과는 "스텝을 줄여라"로 환원되며 이는 VLA-HWBench가 이미 권고한다. 셋째, 기기별 최적점을 믿게 하려면 큰 rollout 예산이 필요하다. flow 샘플링 시드만 바꿔도 500 에피소드 LIBERO 프로토콜에서 2.2점이 벌어진다〔초록〕.

### (5) MuJoCo 버전 민감도: 메커니즘까지 이미 공개됐다

프로젝트의 관찰은 huggingface/lerobot 이슈 #4390(2026-08-09 개설, 미해결)의 재발견이다. 해당 이슈는 libero_spatial task 5의 저장된 초기 상태가 bowl을 ramekin 위 약 11 cm 공중에 두고 `env.reset()` 안의 settle에 의존하며, MuJoCo 3.4.0 이상에서는 bowl이 ramekin 가장자리에 기울어 걸치고 XY 드리프트가 3.72 cm(3.3.7 이하에서는 1.45 cm, 안착)가 된다고 보고한다 ([lerobot #4390](https://github.com/huggingface/lerobot/issues/4390))〔요약〕.

| 출처 | 공개된 내용 | 스택·표본 | 검증 |
|---|---|---|---|
| [lerobot #4390](https://github.com/huggingface/lerobot/issues/4390) | 경계 버전 3.4.0, 원인 후보(box-box 거리 계산 버그 수정), task 5 성공률 SmolVLA 80% → 28%, OpenVLA-OFT+GRPO 98% → 12%; 같은 MuJoCo 3.8.1에서 초기 상태 파일만 재안착시키면 SmolVLA 28% → 84% | LeRobot 0.6.0, robosuite 1.4.1, MuJoCo 3.2.7~3.8.1, 초기 상태 50개 | 〔요약〕 |
| [LIBERO #141](https://github.com/Lifelong-Robot-Learning/LIBERO/issues/141) | 최초 증상 보고(2026-04-16); settle이 프레임 렌더링 전에 일어나 영상에서는 보이지 않음 | robosuite 1.4.0/1.4.1, MuJoCo 3.6.0 | 〔요약〕 |
| [lerobot #4614](https://github.com/huggingface/lerobot/issues/4614) | 같은 공개 체크포인트 `lerobot/smolvla_libero`가 MuJoCo 3.8.1에서 task 5 0/15; suite 76.7%, task 5 제외 85.2%; 기본값 `n_action_steps=50`이 약 20 pp 손실 | lerobot 0.6.2, 과제당 15 에피소드 | 〔요약〕 |
| [libero-mujoco-drift-atlas](https://github.com/clavis-systems/libero-mujoco-drift-atlas) | 40과제 2,000개 상태 중 209개가 1 cm 이상 이동; libero_10의 task 0·1·7은 50/50 전부 | robosuite 1.4.0, hf-libero 0.1.4, 정책 없음 | 〔요약〕 |
| [arXiv 2609.37771](https://arxiv.org/html/2609.37771) | "LIBERO가 MuJoCo를 고정하지 않는다"를 벤치마크 버그로 등재; 3.3.2로 고정하면 pi0.5 "pick up bowl" 62% → 88% | 일곱 벤치마크 감사 | 〔요약〕 |
| MuJoCo changelog·패키지 메타데이터 ([changelog](https://mujoco.readthedocs.io/en/stable/changelog.html), [PyPI hf-libero](https://pypi.org/pypi/hf-libero/json), [openpi requirements](https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/examples/libero/requirements.txt)) | 3.4.0 접촉 관련 항목은 "box-box distance computation 버그 수정" 하나; hf-libero 0.1.4는 `mujoco>=3.0.0,<3.9.0`(3.8.1로 해석됨); openpi LIBERO 예제만 `mujoco==3.2.3` 고정 | 원본 파일 | 원문 |

프로젝트 수치(3.8.1에서 78.0%·task 5 0/120, 3.2.3에서 86.0%)는 공개된 76.7% 대 85.2%와 같은 그림이다. 메커니즘, 3.4.0 경계, 3.7 cm 수치, 사전 안착 초기 상태 파일이라는 수정안을 발견으로 주장할 수 없으며 #4390과 LIBERO #141을 인용해야 한다. 아직 보고되지 않은 것은 작다: lerobot 0.5.1 + hf-libero 0.1.4 + robosuite 1.4.0 스택에서 정책을 넣은 결과, openpi가 고정하는 3.2.3 데이터 포인트(공개 settle 실험의 최저 버전은 3.2.7), 0/15보다 큰 0/120 표본, 안착 후 평가 장면과 **학습 데이터셋 프레임**의 직접 비교, drift만 문서화되고 성공률이 없는 libero_10 task 0·1·7의 정책 수준 영향이다. 이 항목은 기여가 아니라 **재현성 각주**로 쓰는 것이 맞다. 다만 실용적 가치는 있다. arXiv 2609.37771은 같은 버전 차이를 물리가 아닌 렌더링 탓으로 돌렸는데, 3.3.2와 3.8.1 사이에는 3.3.3의 렌더링 변경과 3.4.0의 settle 변경이 모두 있어 그 귀속은 교란돼 있다(노트의 추론).

### (6) 비전 인코더가 병목일 때 남은 방향 다섯 가지

VLA에서 인코더 자체를 줄이는 논문은 둘뿐이다. "Reducing Temporal Redundancy for Efficient VLA Inference"는 pi0·pi0.5(224 px, A100)에서 ViT 내부 attention만 건너뛰어 ViT 40.1 → 28.5 ms를 얻지만 종단 지연은 293.2 → 282.9 ms로 3.5%만 줄고, 재사용이 training-free인지 밝히지 않으며, update-ratio 표가 본문 주장과 반대 방향이다 ([arXiv 2607.12287](https://arxiv.org/pdf/2607.12287)). ActPrune은 인코더 앞에서 패치를 버리지만 보정된 카메라와 wrist depth가 필요하고 인코더 단독 시간을 보고하지 않으며 RGB-only는 향후 과제로 남겼다 ([arXiv 2611.08827](https://arxiv.org/html/2611.08827)).

| 방향 | 가장 가까운 선행 | 같지 않은 이유 | 재학습 없이 시험 가능 | 주된 위험 |
|---|---|---|---|---|
| ① SigLIP 내부 training-free 희소 시간적 토큰 갱신(카메라별, Jetson wall-clock) | arXiv 2607.12287; STC-Cacher ([2512.00891](https://arxiv.org/html/2512.00891))〔요약〕; Eventful Transformers ([2308.13494](https://arxiv.org/abs/2308.13494))〔초록〕 | 224 px·256 토큰·데스크톱 GPU; 비디오 이해 모델에만 평가; pixel-shuffle 커넥터와 카메라당 1024 패치 미취급 | 가능 | FLOP 절감이 TensorRT wall-clock으로 안 옮겨짐; 갱신 사이 feature drift(STC-Cacher에서 정적 캐싱은 43.7 → 33.5); wrist 카메라의 ego-motion |
| ② 카메라별 혼합 주기 인코더 갱신 | ElegantVLA(비전+LLM 통째 생략, PPO 스케줄러); DepthCache의 wrist 게이트([2603.10469](https://arxiv.org/html/2603.10469))〔요약〕; BFA++의 단계별 뷰 중요도([2602.20566](https://arxiv.org/html/2602.20566v1))〔요약〕 | 전부 캐시하거나 전부 갱신; 두 카메라를 이미 인코딩한 뒤 토큰만 재가중 | 가능 | 동기화된 뷰로 학습된 정책의 교차 뷰 시간 불일치 |
| ③ 512/384/320 px 다해상도 경량 fine-tune + 런타임 해상도 선택 | FlexiViT ([2212.08013](https://arxiv.org/abs/2212.08013))〔2차〕, SigLIP 2 NaFlex ([2502.14786](https://arxiv.org/html/2502.14786v1))〔2차〕 | 로봇 정책 아님; VLA는 해상도별 별도 학습만 존재 | 불가(training-free 버전이 실패한 것) | pixel shuffle 때문에 LLM 쪽 토큰 격자가 64 → 36 → 25로 바뀜(노트의 산술 추론); SigLIP 2 저자들은 단순 저학습률 fine-tune이 불충분했다고 보고〔2차〕 |
| ④ RGB-only·proprioception 기반 사전 인코더 패치 드롭 / 연산 절감형 foveation | ActPrune; Look, Focus, Act ([2507.15833](https://arxiv.org/html/2507.15833v2))〔요약〕 | depth·보정 필요; ViT를 처음부터 학습하고 시선 데이터 필요, 언어 없음 | 부분적 | 인코딩 전에 버린 패치는 전역 문맥을 잃음(pi0.5 VLABench 42.0 → 26.3〔요약〕); position ID 보존 필수(T3R [2611.08413](https://arxiv.org/html/2611.08413))〔요약〕 |
| ⑤ frozen policy용 drop-in 증류 비전 타워 + parity-gated FP16/INT8 TensorRT 연구 | EdgeVLA-Base model card ([HF](https://huggingface.co/enfuse/edgevla-base-fmb))〔요약〕 | 정책 전체를 재학습(H200에서 150K 스텝), 오프라인 지표만 | 불가(증류 필요) | 높은 cosine 유사도에서도 폐루프 실패 가능; BitsAndBytes INT8은 Orin에서 SigLIP을 FP16보다 2.4~4.7× **느리게** 만듦([2607.08029](https://arxiv.org/html/2607.08029))〔요약〕 |

이미 붐벼서 열린 방향으로 꼽지 않은 것은 LLM 쪽 토큰 프루닝·KV 캐싱, 비전+LLM의 스텝 단위 통째 생략, 느린 VLA 옆의 경량 correction head(A2C2), wrist 카메라를 학습된 crop으로 대체(FoveaVLA), 강건성 목적의 이벤트 카메라(E-VLA [2604.04834](https://arxiv.org/html/2604.04834v1)〔요약〕)다. 방향 ②는 후보 (1)과 같은 것이고, ③은 후보 (4)의 해상도 축을 여는 전제 조건이다. 450M VLA에서 prefill 지연이 "약 23 ms + 464 ms/메가픽셀"로 픽셀 작업량을 따른다는 워크숍 초록의 법칙〔초록〕에 프로젝트 설정(2 × 512² = 0.524 MP)을 넣으면 약 266 ms로 프로젝트의 260 ms와 가깝지만, 그 법칙의 기준 하드웨어가 확인되지 않았으므로 시사점 이상으로 쓰면 안 된다.

## 실로봇 없이 방어할 수 있는 빈칸은 재생 평가, 드리프트 보정, 카메라별 신선도다

### 프로젝트의 미발표 결과가 문헌 위에서 놓이는 자리

아래는 **요청 프로젝트의 미발표 자체 결과이며 문헌이 아니다.** 공개 체크포인트 `lerobot/smolvla_libero`, LIBERO-Spatial, MuJoCo 3.2.3, 10 스텝마다 재계획 조건에서 성공률은 신선한 관측 86.0%, 250 ms 낡은 관측 78.5%, 500 ms 낡은 관측 74.0%다(각 n=200, skip alignment). 재학습 없이 입력 해상도를 낮추면 position-id 재매핑을 해도 384 px에서 58.0%, 320 px에서 35.5%다. MuJoCo 3.8.1에서는 같은 설정이 78.0%이고 task 5가 0/120이다. Jetson Orin Nano 8GB에서 자체 8-layer 모델의 비전 시간은 260 ms, 디코드 스텝은 43 ms(VLM 층을 16에서 8로 줄이기 전 85 ms)다. 평가 하네스는 카메라별 staleness 주입, 비전·action 시간으로부터의 2-스레드 파이프라인 재생, 드리프트 보정을 지원하고, 8 GB GPU에서 action expert를 다해상도로 fine-tune하는 스크립트가 있다. 한 가지 서술상 주의점이 있다. 디코드 스텝 43 ms가 flow 스텝 1회당 값이라면 10 스텝 기준 action 생성(약 430 ms)이 비전(260 ms)보다 길어 "비전이 병목"이라는 문장이 그 스택에서는 성립하지 않는다(본 보고서의 산술이며 프로젝트 정의를 확인하지 못함). 논문은 스텝 수, 호출당 합계, 런타임(eager·compile·TensorRT 여부)을 함께 적어야 한다.

이 수치들을 문헌에 대면 세 가지가 보인다. 첫째, **staleness 손실 자체는 새 현상이 아니다.** LIBERO 제어 주기를 Leap-VLA가 쓰는 0.05 s로 보면 250 ms와 500 ms는 약 5·10 스텝이고, FutureRTC는 SmolVLA-450M naive async를 d=5에서 70.3%, d=10에서 64.5%로 보고한다(설정이 달라 절대값 비교는 불가) ([arXiv 2607.24008](https://arxiv.org/html/2607.24008)). 둘째, **표본이 경계선이다.** 본 보고서의 단순 이항 근사로 n=200에서 86.0% 대 78.5%는 z≈2.0, 86.0% 대 74.0%는 z≈3.0, 78.5% 대 74.0%는 z≈1.1이다. 250 ms와 500 ms 조건은 통계적으로 구분되지 않으므로, 초기 상태를 짝지은 paired 설계(McNemar)나 셀당 500 에피소드 이상이 필요하다. 셋째, **해상도 결과(−28.0, −50.5점)는 공개된 대응 수치를 찾지 못한 데이터 포인트**다. 다만 HF 블로그의 "observation size sweep"(256 → 83.0%, 512 → 66.5%)〔요약〕은 렌더 크기를 바꾼 것으로 모델 입력 해상도와 다른 축이니 혼동해서는 안 된다. 그리고 프로젝트의 86.0%는 MuJoCo를 3.4 미만으로 고정했기 때문에 얻은 값이므로 버전을 논문에 명시해야 한다.

### 빈칸 1 — 실측·가변·단계 분해 지연의 2-스레드 재생 평가 (후보 2, 가장 방어하기 쉬움)

주장은 "고정 정수 d 평가가 엣지 기기의 실제 타임라인에서 무엇을 놓치는가"로 좁힌다. 필요한 실험은 다음과 같다. Orin Nano에서 비전 인코딩 시간과 청크 생성 시간을 **호출별 트레이스**로 따로 기록하고(전원 모드, 런타임, 카메라 수, 해상도, 스텝 수, 분포와 지터 보고), 이를 LIBERO 안의 2-스레드 discrete-event 타임라인으로 재생한다. 핵심 비교는 같은 평균 지연에 대해 (i) 고정 정수 d, (ii) 상수 지연, (iii) 트레이스 재생 세 조건에서 **수정 방법들의 순위가 바뀌는지**다. 순위가 바뀌지 않으면 기여는 "고정 d로 충분하다"는 음성 결과가 되고, 그것도 정직하게 보고할 가치가 있다. 수정 방법은 drop-only(skip alignment), IT-RTC, 드리프트 보정, 블렌딩을 단독과 조합으로 넣고 동기식 상한을 함께 둔다. 네 suite 전부, MuJoCo 고정, `n_action_steps` 명시가 필요하다. 기기가 하나뿐이라는 약점은 공개된 단계별 타이밍(TensorRT의 91.2 ms / 59.3 ms, vla.cpp의 298.6 ms, Thor의 20.9 ms / 146.6 ms)을 "차용한 가상 기기 프로파일"로 명시해 재생하는 것으로 일부 보완할 수 있다(본 보고서의 제안이며, 차용 수치는 미검증임을 표기해야 한다). LIBERO가 quasi-static이라는 반론에 대비해, Action ControlNet이 d=0~15로 쓴 Meta-World MT50 같은 두 번째 벤치마크를 하나 추가하는 것이 안전하다 ([arXiv 2606.25985](https://arxiv.org/html/2606.25985)).

신규성을 주장하기 전에 **전문으로 읽어야 할 논문**: arXiv 2605.08168(코드 포함), EagleVLA arXiv 2607.12659(버전 번호 확인, v7 기준), "When Delay Meets Distribution Shift"(OpenReview id 5ji7QDSgJK — 이 항목의 가장 큰 불확실성), TIDAL arXiv 2601.14945, DASH arXiv 2611.11854, Leap-VLA arXiv 2611.01011(두 루프의 GPU 경합 처리), PACT-VLA arXiv 2610.28019 부록 C, vla.cpp arXiv 2606.08094, "Where Edge VLA Latency Actually Goes"(Misra & Haque, OpenReview만 존재, "staleness admissibility"의 정의 미확인), FASTER arXiv 2603.19199, arXiv 2611.15491, 그리고 launch offset d를 고정 execution horizon에서 sweep한다는 검색 요약만 있는 FlashVLA arXiv 2608.27384.

### 빈칸 2 — per-step delta 청크의 닫힌 형태 드리프트 보정 (후보 3, 선행이 가장 얇음)

프레이밍은 "delta-action 청크를 지연 하에서 관측 기준 상대 궤적처럼 동작하게 만든다"이고, 표현 수준의 해법은 UMI·openpi에, 재색인 대안은 Mobile UMI에 귀속시킨다. 필요한 실험은 비교군이 핵심이다: (a) drop-only, (b) drop + 일괄 오프셋, (c) drop + k 스텝 분산(k sweep), (d) Mobile UMI식 state 색인 매칭, (e) 시뮬레이터 state로 절대 목표에 재고정하는 oracle, (f) 학습된 참조점으로서 A2C2, (g) IT-RTC와의 조합. 고정 d(0~20)와 빈칸 1의 재생 타임라인 양쪽에서 돌린다. 성공률만으로는 "요령"이라는 비판을 못 막으므로 **의도한 절대 경로로부터의 EE 편차**를 메커니즘 증거로 함께 보고한다. 조사 노트가 짚은 미보고 사실 — 같은 벤치마크에서 delta-action 정책과 절대/state-relative 정책의 지연 손실을 분리한 논문이 없다 — 을 직접 메우면 기여가 커진다. LeRobot LIBERO 환경이 `control_mode` relative/absolute를 제공하므로〔요약〕 지연 손실 가운데 delta 드리프트 몫과 낡은 지각 몫을 나눌 수 있다. 한계 사례(컨트롤러 포화·클리핑, 회전·그리퍼 차원, 추종 오차가 큰 접촉 구간)에서 가정이 깨지는 지점도 보여야 한다.

**전문으로 읽어야 할 자료**: UMI arXiv 2402.10329, Mobile UMI arXiv 2605.20894, VLA-RAIL arXiv 2512.24673(시뮬레이션 부분은 읽지 않음), PPC arXiv 2605.11459, A2C2 arXiv 2509.23224, VLASH arXiv 2512.01031, FutureRTC arXiv 2607.24008의 state-correction 모듈, Action ControlNet arXiv 2606.25985, RAPAC-DP arXiv 2608.15924, BICPO-VLA arXiv 2608.13924, Leap-VLA 부록 B, lerobot의 `rtc/action_queue.py`·`rtc/relative.py`와 openpi의 `transforms.py`·`config.py`. 추가로 널리 쓰이는 LIBERO 평가 하네스의 코드 검색(delta를 절대 목표로 바꾸는지)과 원격조작·궤적 추종 고전 문헌 확인이 남아 있다.

### 빈칸 3 — 카메라별 staleness dose-response와 변화 기반 카메라별 갱신 (후보 1, 신규성은 높고 위험도 높음)

두 단계로 나눠야 한다. **1단계는 측정**이다. wrist만, third-person만, 둘 다 낡게 한 조건에서 age 1~10 제어 스텝(50~500 ms)에 대한 성공률 곡선을 suite별·과제 단계별로 그린다. 이 작은-age dose-response는 어떤 논문에도 없고(LIBERO-VPro는 지연 수준 3개의 크기와 수준별 수치를 확보하지 못함), 하네스가 이미 지원하므로 가장 싸게 얻는 새 데이터다. **2단계는 스케줄러**다. 같은 비전 연산 예산에서 round-robin, wrist 우선 고정 비율, 변화 트리거(픽셀 차이·그리퍼 상태), 지수(변화 × 중요도 / 비용) 정책을 비교하고 성공률, 카메라별 평균 age, 초당 인코딩 횟수를 보고한다. 한 카메라만 다시 인코딩할 때 드는 실제 비용(카메라당 인코딩 시간, 배치 대 순차, prefix 재계산 시간)을 Orin Nano에서 재야 하며, Jetson에서 2-카메라 SigLIP의 배치 대 순차 비교는 공개 측정이 없다.

솔직한 상한 계산이 필요하다. 프로젝트 수치로 관측 전체가 250 ms 낡았을 때의 손실이 7.5점이므로, 완벽한 스케줄링의 최대 이득은 이 지연대에서 대략 그 정도이고 n=200으로는 검출하기 어렵다. 1단계 곡선이 "한 카메라는 낡아도 거의 무해"를 보이면 2단계가 성립하고, 두 카메라 모두 민감하면 이 빈칸은 **음성 결과 논문**이 된다. AoII라는 이름은 지수를 실제로 유도하거나 최소한 pull 기반 AoII 정식화(arXiv 2606.30623)에 대응시킬 때만 써야 한다. 증명 없는 휴리스틱에 AoII를 붙이면 네트워킹 리뷰어에게 과장으로 읽힌다. 또 arXiv 2404.16281의 결과(추론 오차가 age에 대해 비단조일 수 있음)는 순수 age 규칙이 사전적으로 정당화되지 않음을 뜻하므로, 변화 기반 척도를 쓰는 근거로 인용할 수 있다〔초록〕.

1단계 곡선이 여지를 보이지 않을 경우의 대안은 방향 ③이다. 이미 있는 다해상도 fine-tune 스크립트로 512/384/320 px에서 동작하는 단일 체크포인트를 만들면, 프로젝트의 −28.0/−50.5점 실패가 baseline이 되고 "해상도(fidelity) 대 관측 age(freshness)"의 폐루프 sweep이 가능해지며, 이는 후보 (4)에서 유일하게 증거가 전무한 축이다. 단, 학습이 필요하므로 training-free라는 논문의 일관된 축에서는 벗어난다.

**전문으로 읽어야 할 논문**: Leap-VLA arXiv 2611.01011(트리거 경로와 부록), DAM-VLA arXiv 2606.12105, TRIAGE arXiv 2610.34519, StateMem arXiv 2609.22684, React When You Need To arXiv 2609.22587, ElegantVLA arXiv 2605.29438, LIBERO-VPro arXiv 2609.24350(지연 수준별 수치), Blackout vs. Freeze arXiv 2609.39145 부록 E, Acting While Understanding arXiv 2606.15285, Fast-VLA arXiv 2610.21389, arXiv 2607.12287, Misra & Haque 워크숍 논문, AoI 3편(arXiv 2404.16281, 2509.01926, 2606.30623 — 모두 초록만 읽음). 열어보지 못해 평가되지 않은 것들도 확인해야 한다: CMU 석사 논문 "Beyond Vision-Language-Action Models"(느린 비동기 vision-language 채널과 latency-adaptive schedule을 다룬다는 검색 요약만 있음), NebulaVLA arXiv 2608.16503, "Visual Cache Reuse for Efficient Multimodal Inference" arXiv 2610.25136, FreqCache arXiv 2604.24391, Embodied.cpp arXiv 2607.02501, "Decoupled Early Exits" arXiv 2609.29382.

### 새 주장으로 쓸 수 없는 것

프로젝트가 가진 결과 가운데 다음은 인용 대상이지 기여가 아니다. Orin Nano에서 SmolVLA의 비전 인코더가 최대 단계라는 것(vla.cpp 저장소 보고서, 커뮤니티 model card), VLM 층을 16에서 8로 줄이는 것(SmolVLA 논문의 층 수 ablation이 N=8 75.0〔요약〕, N=16 78.5를 이미 보고), 낡은 관측이 성공률을 깎는다는 것, 청크가 추론 지연보다 짧으면 비동기 파이프라인이 깨진다는 것(RTC·SmolVLA·FASTER의 실행 가능성 한계에서 직접 따라 나옴), 생성 측의 "gradient-free·training-free"(SimpleRTC, FreeBar, Leap-VLA soft-nudge), MuJoCo 버전 효과, `n_action_steps`의 영향이다. 반대로 소형 action expert의 eager 지연이 Orin Nano급 CPU에서 "층 수 × 스텝 수에는 비례하고 폭·토큰 수·배치에는 비례하지 않는다"는 구체적 스케일링 시험은 공개된 것이 없다. 메커니즘(커널 실행이 CPU launch보다 짧음)은 큰 모델에서 문서화돼 있으므로 "문서화된 메커니즘의 새 증거"로 부록에 넣을 수 있다.

## 노트 간 모순 아홉 건: 둘은 해소됐고 나머지는 원문 재확인이 필요하다

**1. Leap-VLA(arXiv 2611.01011)의 존재 — 해소.** 비전 인코더 노트는 "약 23M 인코더를 가진 Leap-VLA"를 찾지 못했고 MERL의 ECCV 2026 논문 LEAP-VLA(action-latent prototyping, 고속 시각 스트림 없음)만 찾았다고 적었다 ([MERL TR2026-127](https://shadow.merl.com/publications/TR2026-127))〔2차〕. 반면 비동기 노트와 지연 노트는 서로 독립적으로 arXiv 2611.01011의 HTML 전문을 읽고 같은 내용(Qwen2.5-VL-2B 느린 루프, 62개 로봇 데이터셋으로 사전학습한 약 23M 고속 인코더, measured-age 조건화, soft-nudge, 미실행 트리거 경로)을 인용한다 ([arXiv 2611.01011](https://arxiv.org/html/2611.01011)). 판단: **이름이 같은 두 논문**이며 2611.01011은 존재한다. 한쪽의 "미발견"은 검색어가 MERL 논문에 걸린 검색 실패다. 다만 전문 독해도 앞 100k자와 부록 끝부분을 자동 도구로 읽은 것이므로 인용 전 직접 확인이 필요하고, 인용 시 arXiv ID로 구분해야 한다. 같은 종류의 이름 충돌이 FlashVLA(2505.21200 대 2608.27384), AsyncVLA(2602.13476 대 2511.14148), EcoVLA, Reflex에도 있다.

**2. SmolVLA-on-Orin-Nano 지연 수치 — 스택·지표 차이로 해소, 잔여 의문 하나.** 141.81 ms(실행된 action당, 분할 상환), 약 460 ms와 287 ms(vla.cpp 호출당), 189.89 ms(ONNX Runtime + TensorRT), 107.55 ms(분할 TensorRT 엔진 + 단일 CUDA graph)는 앞 절의 표대로 서로 다른 것을 잰 값이다. 한 노트는 189.89 ms와 107.55 ms를 "충돌"로 표시했으나 같은 저자의 두 산출물(HF model card와 GitHub 저장소)로 보는 것이 타당하다. 해소되지 않은 것은 107.55 ms 파이프라인의 해상도와 비전 시간이 README에 없다는 점과, PyTorch FP32 1167.93 ms라는 수치가 검색 스니펫에만 있다는 점이다〔2차〕. "FP16 약 25 ms, INT8 약 20 ms"라는 블로그 주장은 방법론이 없고 저장소 실측과 모순되므로 신뢰하지 않는다 ([blog](https://jared-hpc.com/posts/vla-physical-ai-tensorrt-tether/))〔요약〕.

**3. "비전이 병목인가" — 기기·스택 의존으로 정리.** Thor eager에서는 action expert가 78%, 450M VLA 워크숍 초록에서는 action expert가 67%〔초록〕, Orin Nano의 vla.cpp·TensorRT에서는 비전이 최대다. 모순이 아니라 조건 차이이며, 이 대비를 두 기기 등급에 걸쳐 한 논문이 진술한 사례는 없다.

**4. SmolVLA의 flow 스텝 대 성공률 — 부분 해소.** 엣지 프로파일링 노트는 "공개된 SmolVLA 스텝 수 ablation이 없다"고 했고, 재현성 노트는 HF 블로그의 sweep(1 스텝 80.5% ~ 10 스텝 65.5%)을 인용한다. 판단: 동료 평가 논문에는 없고 블로그(회색 문헌, 다른 체크포인트)에 있다〔요약〕.

**5. 버전에 따라 달라지는 헤드라인 수치.** EagleVLA는 초기 버전 요약에서 naive PyTorch 대비 8.66×·vla.cpp 대비 5.41×, v7에서 14.71×·9.20×다. VLASH는 초록 기준 reaction latency 최대 11.8× 감소, 다른 노트의 2차 요약에서는 17.4×·속도 2.03×다. 인용 시 버전 번호를 적어야 한다.

**6. A2C2의 크기와 주장.** 한 노트는 correction head를 "학습 파라미터 20.8M", 다른 노트는 "약 32M"으로 적는다(학습 가능 대 전체 파라미터 차이일 가능성, 미확인). 초록의 "LIBERO에서 RTC 대비 +7"을 뒷받침하는 LIBERO RTC 결과가 본문에 없다는 지적도 있다.

**7. MuJoCo 효과의 귀속.** arXiv 2609.37771은 렌더링 변화로, lerobot #4390은 초기 상태 settle로 설명한다. #4390의 동일 물리 A/B가 settle 쪽을 지지하지만, 이슈 본문 자체가 자동 요약으로만 확보됐다〔요약〕.

**8. 같은 논문의 독해 깊이 차이.** TRIAGE, FlashVLA(2608.27384), FutureRTC, TIDAL, Action ControlNet은 한 노트에서는 초록·미열람, 다른 노트에서는 전문 독해다. 본 보고서는 더 깊은 독해를 따랐다.

**9. 인용 전 주의할 내부 불일치와 사소한 어긋남.** arXiv 2607.12287의 update-ratio 표는 본문 주장과 반대 방향이고, ElegantVLA는 한 표에서 CogACT baseline 89.60% 대 자체 72.54%로 "성공률 보존"과 어긋나며, RhinoVLA·EdgeVLA(7×/6×/약 4×)·LiteVLA-Edge(AGX Orin 대 Orin NX)에 내부 수치 불일치가 있다. LIBERO-PRO의 pi0 position 섭동 결과는 2차 요약끼리 0.0과 5.9로 갈리고, openpi 재계획 스텝은 스크립트 5 대 LeRobot 문서 10이며, arXiv 2611.09935는 ID 접두(2611)와 표시된 제출일(10월 13일)이 맞지 않는다.

## 결론

조사가 바꾼 그림은 이렇다. 처음의 질문은 "경량 비동기 VLA에서 무엇이 아직 안 됐는가"였지만, 답은 특정 기법이 아니라 **층위**에 있다. 가중치를 건드리는 모든 것(작게, 얕게, 낮은 비트로, 적은 스텝으로, 지연을 견디도록 재학습)은 붐비고, 가중치를 건드리지 않은 채 제어 루프와 스케줄링에서 할 수 있는 일 — 실측 타임라인을 평가에 넣고, 낡은 청크를 닫힌 형태로 고치고, 어느 카메라를 언제 다시 볼지 정하는 일 — 은 비어 있다. 실로봇이 없는 프로젝트에 이것은 약점이 아니라 맞는 자리다. frozen 공개 체크포인트, 공개 벤치마크, 한 대의 Orin Nano로 주장 가능한 범위가 정확히 그 층위이기 때문이다.

동시에 이 층위의 주장은 측정의 엄밀함으로만 방어된다. 프로젝트의 250 ms와 500 ms 조건은 현재 표본으로 구분되지 않고, MuJoCo 한 줄의 버전 차이가 8점을 움직이며, 시드만 바꿔도 2점이 흔들린다는 보고가 있다. 따라서 논문의 설득력은 새 방법의 개수보다 paired 설계, 버전 고정, 단계별 지연 보고 규약, 그리고 "고정 d로 충분했다"나 "두 카메라 모두 민감했다" 같은 음성 결과를 그대로 싣는 태도에서 나온다. 권하는 구성은 빈칸 1을 틀로, 빈칸 2를 방법으로, 빈칸 3의 1단계 dose-response를 새 데이터로 두고, MuJoCo 관찰과 Nano 지연 분해는 인용을 붙인 재현성 부록으로 내리는 것이다. 11월 둘째 주에만 직접 관련 논문 두 편이 올라왔으므로, 제출 직전에 위 필독 목록과 OpenReview 워크숍 논문 두 편을 다시 확인해야 한다.
