# SmolVLA ONNX Export (2-그래프)

optimum이 Idefics3/SmolVLM을 지원하지 않아 직접 만든 커스텀 export입니다.
런타임 구조를 그대로 반영해 두 그래프로 분할합니다:

```
prefill.onnx : 원본 이미지 2장(1,3,480,640, [0,1]) + 언어 토큰/마스크(1,48) + 정규화 상태(1,32)
               → 16레이어 프리픽스 KV 캐시 (32텐서) + prefix_pad_masks
denoise.onnx : x_t(1,50,32) + timestep(1,) + KV 캐시 → v_t(1,50,32)
호스트 루프  : prefill 1회 → denoise × num_steps + Euler 적분 (onnx_runner.py, numpy만 필요)
```

## 사용법

```bash
pip install onnxruntime   # lerobot[smolvla]에 포함되지 않습니다

# export (~1분). 사전 빌드 바이너리는 512px stale이라 삭제했습니다.
# 아래 명령은 저장소 루트에서 실행하세요 (--policy-path/--model-dir가 루트 기준 상대경로).
# 주의: export_onnx.py에는 --resolution이 없습니다 — 체크포인트 config의 값을 그대로 굽습니다.
# 배포 해상도로 구우려면 optimize_for_inference.py 출력에서 export하세요.
python onnx_export/export_onnx.py --policy-path ./smolvla_optimized --verify --out ./onnx_out

# torch 없이 추론/벤치마크 (numpy + onnxruntime만)
python onnx_export/onnx_runner.py --model-dir ./onnx_out --num-steps 4

# Jetson에서 TensorRT로 (onnxruntime-gpu 또는 trtexec)
python onnx_export/onnx_runner.py --model-dir ./onnx_out \
    --providers TensorrtExecutionProvider CUDAExecutionProvider
# 또는 개별 엔진 빌드:
#   trtexec --onnx=prefill.onnx --fp16 --saveEngine=prefill.trt
#   trtexec --onnx=denoise.onnx --fp16 --saveEngine=denoise.trt
```

## 검증 상태

- ONNX(ORT CPU) ↔ PyTorch 액션 max|diff| = **7.15e-06** (동일 노이즈, end-to-end PASS)
- ORT CPU(Mac): 재측정 결과 **PyTorch보다 1.13× 느림** (ORT 783ms vs PyTorch 696ms, 6쌍 전부). 이전 1.4× 주장은 철회 — GPU/TensorRT가 본 목적
- TensorRT는 **한 번도 실행된 적 없습니다.** 리스크: 레거시 export가 남긴 shape 연산 꼬리(prefill 11,349 노드 중 Shape 932개) → `polygraphy surgeon sanitize --fold-constants` 선행 권장. `LayerNormalization`/`IsNaN`은 TRT ≥8.6/≥8.5 필요
- `onnx_runner.py`는 **벤치 하네스**입니다 — 토크나이즈/상태 정규화/액션 역정규화가 호스트 책임이고 그 코드는 여기 없습니다

## Export 시 적용되는 패치 (수치 영향 없음/무시 가능)

1. **비전 위치 임베딩 정적화**: 가변 해상도용 bool-scatter(`position_ids[mask]=...`)가
   ScatterND 타입 오류를 유발 → 우리 입력은 항상 고정 해상도 풀 이미지라 위치 id가 상수 →
   원본 공식으로 미리 계산해 상수로 굽는다 (완전 동일)
2. **타임스텝 사인 임베딩 f64→f32**: TensorRT가 f64 거부 (오차 ~1e-7)
3. **bool cumsum/mul 회피**: ONNX CumSum/Mul은 bool 미지원 → int 캐스팅 버전 사용 (동일)
4. **fp32 export**: bf16은 ORT CPU 커널 부재. 정밀도는 TensorRT 빌드에서 `--fp16`으로

## 고정된 것 (정적 shape — TensorRT 친화)

배치 1 · 카메라 2 (480×640 입력, in-graph 리사이즈 — **해상도는 export 시점 체크포인트 config에서 굳습니다**) · 언어 48토큰(마스크로 가변)
· chunk 50 · num_steps는 호스트 루프 파라미터 (그래프 무관, 1~10 자유)

raw 입력 크기(`RAW_HW`)와 언어 길이(`LANG_LEN`)는 export_onnx.py 상단 상수입니다.
**카메라 수는 상수만으로 안 바뀝니다** — `CAMERAS=2`는 `--random` 입력 생성에만 쓰이고, 그래프 자체는
`PrefillGraph.forward` 시그니처와 `input_names`에 이미지 2장이 하드코딩돼 있습니다.
해상도는 CLI가 아니라 체크포인트 config에서 굳으므로 `optimize_for_inference.py --resolution`으로
먼저 바꾼 뒤 export하세요.

## 호스트 책임 (그래프 밖)

- 태스크 문자열 토크나이즈 (끝에 `\n` 필수, 48로 패딩) — `tokenizers` 라이브러리면 충분
- 상태 MEAN_STD 정규화 + 32차원 제로패딩 / 출력 액션 역정규화 + 실제 차원 슬라이스
- 노이즈 샘플링 (러너가 기본 처리)
