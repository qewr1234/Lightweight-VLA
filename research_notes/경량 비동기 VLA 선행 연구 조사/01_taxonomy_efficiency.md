# Taxonomy of VLA Efficiency Work (state of the literature as of 2026-11-16)

Scope: how Vision-Language-Action (VLA) models have been made lightweight / fast; feeds a prior-art survey for SmolVLA-class models (~450M params: SigLIP encoder + small VLM prefix with KV cache + flow-matching action expert emitting action chunks) on a Jetson Orin Nano 8GB.

**Verification legend used on every row / bullet**

- **[F]** = taken from the paper's full text (arXiv HTML / README fetched in this session).
- **[A]** = taken from the arXiv abstract / metadata only (arXiv API, fetched in this session). Numbers are the authors' self-reported headline numbers.
- **[S]** = secondary source (a survey table or a search-result snippet quoting the paper); not checked against the primary text.
- **[U]** = unverified (could not be fetched; stated from the researcher's background knowledge or flagged as uncertain).

All arXiv IDs below were returned by the arXiv API or arXiv HTML in this session; none are invented. Dates are first-submission dates (YYYY-MM-DD). Many Sept-Nov 2026 items are v1 preprints without peer review; treat their numbers as self-reported.

**Name collisions to watch for when citing** (all confirmed via arXiv API):
- "FlashVLA" = (a) *Think Twice, Act Once: Token-Aware Compression and Action Reuse...* [2505.21200](https://arxiv.org/abs/2505.21200) (2025-05-27, token pruning + action reuse) and (b) *FlashVLA: Streaming Action Decoding for Fast and Asynchronous VLA Inference* [2608.27384](https://arxiv.org/abs/2608.27384) (2026-08-27).
- "EcoVLA" = (a) *Environment-Aware Adaptive Pruning with Interleaved Inference Orchestration* [2602.00780](https://arxiv.org/abs/2602.00780) and (b) *EcoVLA: Energy-Efficient Device-Edge Co-Inference* [2608.15502](https://arxiv.org/abs/2608.15502).
- "Reflex" = [2607.14695](https://arxiv.org/abs/2607.14695) (streaming inference, ICML 2026) and 2608.14379 (reaction-critical manipulation; title only seen).
- "FastVLA" [2610.22146](https://arxiv.org/abs/2610.22146) vs "Fast-VLA: You Only Think Once" (2610.21389; title only seen).
- "LiteVLA-Edge" [2603.03380](https://arxiv.org/abs/2603.03380) vs "LiteVLA-H" [2605.00884](https://arxiv.org/abs/2605.00884) vs "LightVLA" [2509.12594](https://arxiv.org/abs/2509.12594).
- "EdgeVLA / EVLA" [2507.14049](https://arxiv.org/abs/2507.14049) (K-Scale) is the paper; "EagleVLA" [2607.12659](https://arxiv.org/abs/2607.12659) was formerly "Jetson-PI".

---

## Q1. What are the main families of VLA efficiency work, and which papers represent each?

### Takeaway
The literature (two dedicated 2025 surveys plus ~150 primary papers through Nov 2026) clusters into nine families: (1) small backbones / static layer removal, (2) visual-token pruning / merging / caching, (3) dynamic depth / early exit, (4) quantization / binarization, (5) faster action generation, (6) dual-system fast/slow designs, (7) distillation, (8) inference runtimes / compilation, and (9) asynchronous / latency-aware execution. From mid-2026 a tenth strand appears that is not a method family but a corrective: measurement and evaluation-validity studies showing that many reported speedups do not transfer across hardware, execution modes or closed-loop evaluation.

### Cited Findings

#### Family 1 - Small backbones and static layer skipping / removal

| Method | arXiv / date / venue | Backbone and size | What is made small | Retraining | Benchmark and headline numbers | Hardware reported | Ver. |
|---|---|---|---|---|---|---|---|
| SmolVLA | [2506.01844](https://arxiv.org/abs/2506.01844), 2025-06-02 | SmolVLM-2 (SigLIP + SmolLM2); 450M total, ~100M action expert; 0.24B and 2.25B variants | No image tiling + pixel shuffle -> 64 visual tokens/frame; only first 16 of 32 LLM layers (N = L/2); action expert width 0.75x with interleaved cross-/causal self-attention; 10 flow steps; chunk 50; VLM frozen | Yes (expert trained; pretraining ~30k GPU-h total project, 4 GPUs) | LIBERO avg 87.3 (0.45B) vs pi0 86.0 / OpenVLA 76.5; Meta-World 57.3 vs pi0 47.9; real SO100 78.3 vs pi0 61.7 / ACT 48.3. Ablation: N=16 -> 78.5, N=24 -> 79.5, N=32 -> 80.3 | Paper gives **no latency numbers and no GPU model**; claims consumer GPU / CPU deployability; "~40% faster to train, 6x less memory" than pi0 | [F] |
| TinyVLA | [2409.12514](https://arxiv.org/abs/2409.12514), 2024-09-19 | Small multimodal backbone (Pythia-1.3B per survey [2510.17111](https://arxiv.org/abs/2510.17111)) + diffusion policy decoder; no robot pre-training | Backbone size; LoRA | Yes | "significantly outperforms OpenVLA in speed and data efficiency" [A]; 14 ms per action prediction vs OpenVLA-7B 292 ms on one A6000 per [2611.18372](https://arxiv.org/abs/2611.18372) Table 2 [S] | A6000 [S] | [A]/[S] |
| MiniVLA | No arXiv ID found; Stanford blog (ai.stanford.edu/blog/minivla, returned HTTP 503 in this session); cited as ref [84] "MiniVLA: a better VLA with a smaller footprint" in [2510.24795](https://arxiv.org/abs/2510.24795) | Survey says it pairs SigLIP with DINOv2 [S]; Qwen2.5-0.5B LLM, ~1B params, VQ action chunking [U] | Backbone size | Yes | LIBERO-90 improvement over OpenVLA [U] | [U] | [S]/[U] |
| EdgeVLA (EVLA) | [2507.14049](https://arxiv.org/abs/2507.14049), 2025-07-18 | Small language model backbone | Removes autoregressive end-effector prediction ("7x speedup in inference") + SLM | Yes | "early results": training characteristics comparable to OpenVLA; abstract gives no task-success or device latency numbers | Abstract says "edge devices", names none | [A] |
| NORA | [2504.19854](https://arxiv.org/abs/2504.19854), 2025-04-28 | Qwen-2.5-VL-3B + FAST+ tokenizer, 3B | Backbone size | Yes (970k demos) | Outperforms larger VLAs (abstract; no numbers) | not in abstract | [A] |
| RoboMamba | [2406.04339](https://arxiv.org/abs/2406.04339), 2024-06-06, NeurIPS 2024 | Mamba SSM in place of transformer LLM | Linear-complexity sequence model; tunes 0.1% params | Yes | "inference speeds 3 times faster than existing VLA models" | not in abstract | [A] |
| VLA-Adapter | [2509.09372](https://arxiv.org/abs/2509.09372), 2025-09-11 | Prismatic VLM on Qwen2.5-0.5B; 97M policy with Bridge Attention; no robot pre-training | Backbone size | Yes (8 h on one consumer GPU) | LIBERO avg 97.3 (Pro 98.5); CALVIN ABC->D 4.42; 219.2 Hz throughput, 0.0365 s latency vs OpenVLA-OFT 71.4 Hz, OpenVLA 4.2 Hz | Experiments on 4x H100 | [F] |
| Evo-1 | [2511.04555](https://arxiv.org/abs/2511.04555), 2025-11-06 | InternVL3-1B (InternViT-300M + Qwen2.5-0.5B, only first 14 language layers); cross-attention-only DiT with flow matching; 0.77B | Backbone size + layer truncation; 448x448 with pixel-unshuffle (4x fewer tokens) | Yes (two-stage; no robot pretraining) | Meta-World 80.6, LIBERO 94.8, real xArm6 78%; 16.4 Hz / 2.3 GB vs SmolVLA 12.7 Hz / 2.0 GB, pi0 11.5 Hz / 17.9 GB, OpenVLA 7.9 Hz / 15.1 GB | RTX 4090d only; **no Jetson evaluation** | [F] |
| NanoVLA | [2510.25122](https://arxiv.org/abs/2510.25122), 2025-10-29 | ResNet18 image encoder + BERT-base (S, 161M) or Qwen2.5-0.5B (L, 520M); routed variant ~296M avg; 52M trainable | Late vision-language fusion so the instruction embedding is computed once and cached; long-short action chunking; dynamic small/large routing | Yes (fine-tuning) | LIBERO avg 84.1 (R), 80.4 (L), 78.7 (S); "52x higher FPS and +13.8% SR" vs OpenVLA; vs SmolVLA: 43.8% higher FPS at 50 AC steps, 17.1% slower at 10 AC steps | **Jetson Orin Nano Super 8GB** (exact FPS only in figures) | [F] |
| FLOWER | [2509.04996](https://arxiv.org/abs/2509.04996), 2025-09-05, CoRL 2025 | 950M; Florence-2-style VLM with intermediate-modality fusion | Prunes up to 50% of LLM layers, reallocating capacity to the flow head; Global-AdaLN cuts 20% params | Yes (200 H100 GPU-hours pretraining) | CALVIN ABC 4.53 (SoTA claim); 190 tasks / 10 benchmarks | not in abstract | [A] |
| X-VLA | [2510.10274](https://arxiv.org/abs/2510.10274), 2025-10-11 | 0.9B soft-prompted transformer, flow matching | Size | Yes | SOTA claims across 6 sims / 3 robots | not in abstract | [A] |
| TurboVLA | [2607.27205](https://arxiv.org/abs/2607.27205), 2026-07-29 | 0.2B, direct V+L->A (no LLM-centric path) | Removes LLM from the execution path | Yes | LIBERO 97.6%; 31.2 ms; 0.9 GiB VRAM | RTX 4090 | [A] |
| CoTinyVLA | [2607.25487](https://arxiv.org/abs/2607.25487), 2026-07-28 | 0.9B on Qwen3.5-0.8B; CoT distilled from 35B teacher | Size (robustness via supervision) | Yes | LIBERO-Plus 90.8/87.3/86.6/80.7; peak 2.25 GiB GPU memory | not in abstract | [A] |
| EffVLA ("What Makes an Efficient VLA?") | [2609.13984](https://arxiv.org/abs/2609.13984), 2026-09-12 | SigLIP2 + Qwen2.5, controlled sweep | Finds action-head *initialization* (copying last LLM layers) matters more than head architecture/loss/inference budget; "extra inference passes give no measurable benefit"; returns diminish near pi-series size | Yes | Matches strongest open VLAs on LIBERO at lower latency; SO-ARM101 transfer | "measured on-device latency" (device not in abstract) | [A] |
| Shallow-pi (static depth reduction by KD) | [2601.20262](https://arxiv.org/abs/2601.20262), 2026-01-28 | pi0 / pi0.5, 18 -> 6 transformer layers in both VLM and action head | Depth | Yes (KD; teacher+student in memory) | LIBERO: pi0 95 -> 94, pi0.5 96 -> 95; H100 CUDA time 22.6 -> 10.5 ms (pi0), 25.5 -> 11.3 ms (pi0.5); FLOPs 2.93T -> 1.18T | **Jetson Orin**: pi0 364 ms -> 110 ms (SmolVLA measured at 230 ms on same Orin); **Jetson Thor**: pi0.5 130 -> 78 ms | [F] |

- The EVLA survey classifies SmolVLA's layer skipping, RLRC, FLOWER (and EfficientVLA) as *training-free* layer pruning and MoLe-VLA / LightDP as *training-based*, and describes SmolVLA's approach as "a pragmatic, naive strategy" - [2510.24795](https://arxiv.org/abs/2510.24795) [F].
- Shallow-pi states that "systematic transformer layer reduction has received limited attention and ... has not been explored for flow-based VLA models under knowledge distillation" - [2601.20262](https://arxiv.org/abs/2601.20262) [A].

#### Family 2 - Visual-token pruning / merging / caching (and feature reuse)

| Method | arXiv / date / venue | Base model | Mechanism | Retraining | Headline numbers | Hardware | Ver. |
|---|---|---|---|---|---|---|---|
| VLA-Cache | [2502.02175](https://arxiv.org/abs/2502.02175), 2025-02-04, NeurIPS 2025 | OpenVLA-family | Reuses cached **KV representations** of static visual tokens across frames; recomputes task-relevant tokens; layer-adaptive reuse ratio | No (training-free) | up to 1.7x CUDA-latency speedup, +15% control frequency; LIBERO, SIMPLER, real robot | not in abstract | [A] |
| EfficientVLA | [2506.10100](https://arxiv.org/abs/2506.10100), 2025-06-11 | CogACT | (1) prune redundant LLM layers, (2) task-aware visual token selection, (3) cache intermediate features in diffusion head | No | 1.93x speedup, FLOPs to 28.9%, -0.6% SR on SIMPLER | not in abstract | [A] |
| FlashVLA (2505.21200) | [2505.21200](https://arxiv.org/abs/2505.21200), 2025-05-27 | OpenVLA-family | Token-aware action reuse across stable steps + visual token selection | No | FLOPs -55.7%, latency -36.0%, SR -0.7% on LIBERO | not in abstract | [A] |
| SP-VLA | [2506.12723](https://arxiv.org/abs/2506.12723), 2025-06-15 | 7B VLA + lightweight generator (ridge regression per [2510.17111](https://arxiv.org/abs/2510.17111)) | Action-aware model scheduling (deliberative vs intuitive actions) + spatio-semantic token pruning | No | 1.5x lossless on LIBERO, 2.4x on SimplerEnv | not in abstract | [A] |
| SpecPrune-VLA | [2509.05614](https://arxiv.org/abs/2509.05614), 2025-09-06, ICML 2026 | OpenVLA-OFT-class | Action-level static + layer-level dynamic pruning + coarse/fine action-aware controller | No | up to 1.57x (LIBERO), 1.70x real | not in abstract | [A] |
| LightVLA | [2509.12594](https://arxiv.org/abs/2509.12594), 2025-09-16 | OpenVLA-OFT-class | Differentiable (Gumbel-softmax) query-based token selection learned by fine-tuning; no extra params | **Yes** (fine-tuning) | FLOPs -59.1%, latency -38.2%, SR +2.6% on LIBERO | not in abstract | [A] |
| CogVLA | [2508.21046](https://arxiv.org/abs/2508.21046), 2025-08-28, NeurIPS 2025 | OpenVLA (SigLIP + DINOv2), LoRA | EFA-Routing: instruction FiLM inside the vision encoders, 64 aggregation tokens per branch (visual tokens to 25%); LFP-Routing: ~50% pruning in LLM; coupled attention with parallel decoding | **Yes** | LIBERO 97.4%; latency 0.091 s / 87.9 Hz / 2.72T FLOPs vs OpenVLA 0.254 s / 3.9 Hz / 8.48T and OpenVLA-OFT 0.132 s / 60.6 Hz; training cost 2.5x lower | trained on 4x A800; inference hardware not specified | [F] |
| VLA-Pruner | [2511.16449](https://arxiv.org/abs/2511.16449), 2025-11-20 | multiple VLAs | Dual-level importance (semantic prefill + temporally smoothed action-decode attention) | No | up to 1.99x speedup | not in abstract | [A] |
| TTF-VLA | [2508.19257](https://arxiv.org/abs/2508.19257), 2025-08-15, AAAI 2026 | OpenVLA, VLA-Cache | Temporal token fusion of historical and current visual tokens (pixel diff + attention) | No | +4.0 pp LIBERO (72.4 vs 68.4). **Primary claim is quality, not speed**; notes query-matrix reuse as a route to acceleration | not in abstract | [A] |
| ADP | [2509.22093](https://arxiv.org/abs/2509.22093), 2025-09-26 | OpenVLA, OpenVLA-OFT | Text-driven pruning gated by recent action trajectory | No | 1.35x on OpenVLA-OFT | not in abstract | [A] |
| KV-Efficient VLA | [2509.21354](https://arxiv.org/abs/2509.21354), 2025-09-20 | generic | RNN-gated chunked KV cache for history tokens | Yes (gate) | 24.6% FLOPs saving, 1.34x speedup, 1.87x KV-memory reduction | not in abstract | [A] |
| TEAM-VLA | [2512.09927](https://arxiv.org/abs/2512.09927), 2025-12-10 | LIBERO VLAs | Token expand-then-merge | No | faster at same/higher SR (no number in abstract) | - | [A] |
| Learned token caching | [2602.00686](https://arxiv.org/abs/2602.00686), 2026-01-31 | OpenVLA-class | Learned cached-token selector + cache-ratio predictor | **Yes** | 1.76x wall-clock, +1.9 pp LIBERO | - | [A] |
| ST-Merge | [2606.29350](https://arxiv.org/abs/2606.29350), 2026-06-28 | Qwen2.5-VL, pi0.5 | Spatio-temporal token merging **during the visual encoding phase** with positional correction | No | 8.3x on pi0.5 at 1024x1024; small accuracy drop at lower resolutions | - | [A] |
| Temporal-redundancy reduction | [2607.12287](https://arxiv.org/abs/2607.12287), 2026-07-14 | LIBERO / RoboTwin VLAs | "incrementally update only tokens corresponding to dynamic scene regions instead of re-encoding entire frames" + 2-step diffusion schedule | Yes (for 2-step) | >2x speedup, up to 98% SR | - | [A] |
| Gated VLA-Cache | [2608.10824](https://arxiv.org/abs/2608.10824), 2026-08-11, IROS 2026 | OpenVLA, OpenVLA-OFT | Invalidates KV cache when action-logit margin is low | No | recovers >100% of lost accuracy, keeps 80% of compute savings | - | [A] |
| GeoScaffold | [2609.36967](https://arxiv.org/abs/2609.36967), 2026-09-29 | pi0.5 | Region budgets + farthest-point sampling to keep a "spatial scaffold" | No | 20% tokens kept, 93.2% LIBERO, 1.78x **prefill** speedup | - | [A] |
| TVCache | [2609.34319](https://arxiv.org/abs/2609.34319), 2026-09-28 | 4 VLAs | Text-vision synergistic caching | No | +14.5 pp over VLA-Cache at 12.5% retention; FLOPs /2.45 | - | [A] |
| SAPrune | [2610.05273](https://arxiv.org/abs/2610.05273), 2026-10-04, NeurIPS 2026 | LIBERO / SIMPLER VLAs | Calibrated pruning layers + dual-path rule | No (small calibration set) | 87.5% tokens pruned, up to 1.718x | - | [A] |
| FastVLA | [2610.22146](https://arxiv.org/abs/2610.22146), 2026-10-15, WACV 2027 | OpenVLA, OpenVLA-OFT, CogACT | Task-conditioned, temporally adaptive compression | No | LLM FLOPs -35-59%, action latency -9-32% (1.09-1.46x); real robot 1.25x | Franka FR3 (GPU not in abstract) | [A] |
| TRIAGE | [2610.34519](https://arxiv.org/abs/2610.34519), 2026-10-25 | two backbones | Per-token choice among recompute / KV reuse / prune based on shallow-layer drift | No | within 3 pp of full recompute; prefill latency **-48% on Jetson AGX Orin** | **Jetson AGX Orin** | [A] |
| T3R | [2611.08413](https://arxiv.org/abs/2611.08413), 2026-11-05 | 3 backbones | Two-stage token refinement; pruning costs 7.8 ms/call; up to 3.3x LLM-forward speedup at high resolution | No | benefits strongest in multi-camera settings | - | [A] |
| ActPrune | [2611.08827](https://arxiv.org/abs/2611.08827), 2026-11-06 | LIBERO / VLABench VLAs | View-specific action-grounded pruning (end-effector trajectory in third-person view; ego-motion-compensated static patches in wrist view) | No | up to 1.61x, FLOPs -59.5% | - | [A] |
| APS-VLA | [2611.13030](https://arxiv.org/abs/2611.13030), 2026-11-09 | OpenVLA-OFT, pi0.5, XR-1 | Action-guided future-aware pruning | aux predictor removed at inference | real dual-arm: 137.25 -> 93.25 ms (1.47x) | - | [A] |
| rMuscle | [2609.19104](https://arxiv.org/abs/2609.19104), 2026-09-16 | flow VLAs | Cross-execution context cache (visual-token outputs) + action cache (activation patterns) | No | 1.29-1.42x | RTX 4090 and **Jetson Thor** | [A] |
| ActionCache | [2607.06370](https://arxiv.org/abs/2607.06370), 2026-07-07 | pi0.5, GR00T-N1.6 | Warm-start flow sampling from retrieved past intermediate actions | No | action-head acceleration up to 10.44x / 40.17x | - | [A] |
| Block-wise Adaptive Caching (BAC) | [2506.13456](https://arxiv.org/abs/2506.13456), 2025-06-16 | Diffusion Policy, VLAs | Block-wise feature caching across denoising steps | No | up to 3x | - | [A] |

- The survey [2510.17111](https://arxiv.org/abs/2510.17111) [F] splits this family into "selective feature processing" (FastV, EfficientVLA, SP-VLA, FlashVLA, LightVLA, ADP, FastDriveVLA, SpecPrune-VLA, SQAP-VLA; OTTER; UniVLA) and "temporal sharing and reuse" (VLA-Cache, TTF-VLA, FlashVLA, EfficientVLA, Fast ECoT).
- The survey [2510.24795](https://arxiv.org/abs/2510.24795) [F] splits "token optimization" into compression (SmolVLA, CogVLA, Oat-VLA, FAST, VOTE), pruning (FlashVLA, EfficientVLA, SP-VLA, CogVLA, SpecPrune-VLA, SQAP-VLA, LightVLA, KV-Efficient VLA, ADP) and caching (VLA-Cache, HybridVLA, FlashVLA, Fast ECoT, EfficientVLA, CronusVLA, AMS).

#### Family 3 - Dynamic depth / early exit / dynamic layer skipping

| Method | arXiv / date / venue | Base | Mechanism | Retraining | Headline numbers | Ver. |
|---|---|---|---|---|---|---|
| DeeR-VLA | [2411.02359](https://arxiv.org/abs/2411.02359), 2024-11-04, NeurIPS 2024 | RoboFlamingo-style MLLM | Multi-exit architecture; termination criteria under average-compute, peak-compute and GPU-memory budgets | Yes | CALVIN: **LLM** compute -5.2-6.5x, **LLM** GPU memory -2-6x without performance loss (numbers refer to the LLM only) | [A] |
| MoLe-VLA | [2503.20384](https://arxiv.org/abs/2503.20384), 2025-03-26 | MLLM-based VLA | Layers as experts; Spatial-Temporal Aware Router; cognition self-KD | Yes | RLBench + real: +8% mean SR; compute "up to x5.6" lower than standard LLMs | [A] |
| CEED-VLA | [2506.13725](https://arxiv.org/abs/2506.13725), 2025-06-16 | Jacobi/parallel-decoding VLAs (PD-VLA line) | Consistency distillation + early-exit **decoding** (exit of the iterative decoder, not of LLM depth) | Yes | >4x inference acceleration | [A] |
| DySL-VLA | [2602.22896](https://arxiv.org/abs/2602.22896), 2026-02-26, DAC 2026 | RoboFlamingo-class on CALVIN | Informative (always run) vs incremental (skippable) layers; prior-post skipping guidance; skip-aware two-stage KD | Yes (but light) | 3.75x latency reduction vs RoboFlamingo; +2.1% avg successful length vs DeeR-VLA with 85.7x fewer trainable params and 13.7x fewer training steps | [S] (search summary of arXiv listing) |
| Scene-Adaptive VLA (driving) | 2610.24590, 2026-10-17 (title only seen) | driving VLA | Dynamic layer routing | - | - | [U] |
| AutoPruner | 2610.23880 (title only seen) | - | Agentic depth pruning | - | - | [U] |

- Survey [2510.17111](https://arxiv.org/abs/2510.17111) [F] groups SmolVLA (fixed removal), FLOWER (upper-layer pruning), DeeR-VLA (early exit), MoLe-VLA (MoE gating + self-distillation) and EfficientVLA (skip layers with high input/output cosine similarity) under "dynamic computation pathways".

#### Family 4 - Quantization and binarization

| Method | arXiv / date / venue | Type | What is quantized | Retraining | Headline numbers | Hardware | Ver. |
|---|---|---|---|---|---|---|---|
| OpenVLA (serving) | [2406.09246](https://arxiv.org/abs/2406.09246), 2024-06-13 | PTQ (4-bit) | LLM backbone | No | "served efficiently via quantization without a hit to downstream success rate"; survey: 4-bit "halves GPU memory" [S, [2510.24795](https://arxiv.org/abs/2510.24795)] | - | [A] |
| QAIL | [2412.01034](https://arxiv.org/abs/2412.01034), 2024-12-02 | QAT | 4-bit weights (manipulation); W4A4 (driving) | Yes | up to 2.5x speedup and 2.5x energy saving (4-bit weights) "on a real edge GPU"; 3.7x / 3.1x on a low-end GPU for driving | "edge GPU" - **device not named in abstract** | [A] |
| SQIL | [2505.15304](https://arxiv.org/abs/2505.15304), 2025-05-21 | QAT + saliency loss weighting | 4-bit weights | Yes | up to 2.5x speedup / 2.5x energy saving on an edge GPU | device not named in abstract | [A] |
| BitVLA | [2506.07530](https://arxiv.org/abs/2506.07530), 2025-06-09 (v2 2026-03-01) | Native 1.58-bit (ternary) everywhere; "Quantize-then-Distill" compresses the **vision encoder** to 1.58-bit with a full-precision teacher | LLM (BitNet b1.58 2B4T) **and vision encoder** | Yes | Matches OpenVLA-OFT; memory -11.0x, end-to-end latency -4.4x | not in abstract; see vla.cpp rows in Q4 for Jetson numbers | [A] |
| RLRC | [2506.17639](https://arxiv.org/abs/2506.17639), 2025-06-21, RA-L 2026 | Structured pruning -> SFT+RL recovery -> quantization | LLM | Yes | up to 8x memory reduction, 2.3x speedup at original SR | - | [A] |
| SQAP-VLA | [2509.09090](https://arxiv.org/abs/2509.09090), 2025-09-11 | PTQ co-designed with token pruning | LLM | No | 1.93x speedup, up to +4.5% SR | - | [A] |
| QVLA | [2602.03782](https://arxiv.org/abs/2602.03782), 2026-02-03, ICLR 2026 | Channel-wise mixed-precision PTQ guided by action-space sensitivity (0-bit = pruning) | OpenVLA-OFT | No | 29.2% of VRAM, 98.9% of performance, 1.49x speedup; +22.6% over SmoothQuant | - | [A] |
| QuantVLA | [2602.20309](https://arxiv.org/abs/2602.20309), 2026-02-23, CVPR 2026 | Scale-calibrated PTQ (attention temperature matching, output-head balancing) | Linear layers in the language backbone **and the DiT action head** | No (small unlabeled calibration buffer) | LIBERO SR above FP baselines; ~70% relative memory saving on quantized components; 1.22x end-to-end speedup | not stated in search summary | [S] |
| HBVLA | [2602.13710](https://arxiv.org/abs/2602.13710), 2026-02-14 | 1-bit PTQ (policy-aware Hessian + Haar domain) | OpenVLA-OFT, CogACT | No | retains 92.2% (LIBERO) / 93.6% (SimplerEnv) of FP performance | - | [A] |
| DyQ-VLA | [2603.07904](https://arxiv.org/abs/2603.07904), 2026-03-09 | Runtime-error-aware dynamic activation precision | - | No | 1.88-1.93x speedup | - | [A] |
| DA-PTQ | [2604.11572](https://arxiv.org/abs/2604.11572), 2026-04-13 | Drift-aware PTQ + motion-driven mixed precision | - | No | no numbers in abstract | - | [A] |
| ActQuant | [2605.24011](https://arxiv.org/abs/2605.24011), 2026-05-19 | Sub-4-bit action-guided weight-only PTQ + OmniModel.cpp C/C++ runtime | Backbone only | No | <=3 bpw: 95.0% (OpenVLA-OFT), 94.8% (pi0.5); 2.5 bpw compresses backbone 14.3 GB -> 2.7 GB | UR3 real arm | [A] |
| HoloQ-VLA | [2605.28803](https://arxiv.org/abs/2605.28803), 2026-05-27 | Uniform W4A4 PTQ | LLM **and entire diffusion action head** | No | pi0.5 98.0% / GR00T-N1.5 87.8% at W4A4; static memory -74.2% | - | [A] |
| Mix-QVLA | [2606.19565](https://arxiv.org/abs/2606.19565), 2026-06-17 | Mixed-precision PTQ | OpenVLA-OFT | No | 15.4 -> 4.1 GB, 96.3 vs 97.1 SR, 1.52x | - | [A] |
| FoldQuantVLA | [2609.24433](https://arxiv.org/abs/2609.24433), 2026-09-21 | PTQ W4A4 with **native integer TensorRT plugins** | Projections in the language backbone and iterative action expert | No | W4A4: 1.20-1.33x over floating-point TensorRT on Orin, 1.25-1.52x on desktop; GR00T N1.7 real-robot SR 80.0% (uniform W4A4) vs 92.5% (keeping some projections at W8A8, +1 ms on Orin) | **Jetson AGX Orin**, Ada GPUs | [A] |
| CHASE-VLA | [2610.02666](https://arxiv.org/abs/2610.02666), 2026-10-02, ACCV 2026 | Chunk-aware PTQ for the action expert | AE MLP + attention projections at W4A4 | No | pi0.5 97.3% LIBERO; AE linear-layer storage -73.4% | - | [A] |
| QCRL | [2610.15675](https://arxiv.org/abs/2610.15675), 2026-10-11 | Quantization-conditioned RL recovery | pi0.5, GR00T N1.5 (INT4 W4A16; NVFP4 W4A4) | Yes | 99.3% of FP with INT4 and 42.9% memory saving; NVFP4: 1.73x speedup vs FP16 | **Jetson Thor** | [A] |
| Benchmarks: LoopQ, VLAQuantBench, QbD-VLA, PIVOT-Q | [2611.05093](https://arxiv.org/abs/2611.05093), [2609.25376](https://arxiv.org/abs/2609.25376), [2610.38661](https://arxiv.org/abs/2610.38661), [2610.05745](https://arxiv.org/abs/2610.05745) | evaluation / recovery | see Q5 | - | see Q5 | H100, L4, Jetson AGX Thor (LoopQ) | [A] |

#### Family 5 - Faster action generation

5a. Decoding scheme for token-based (autoregressive) VLAs

| Method | arXiv / date / venue | Mechanism | Retraining | Headline numbers | Hardware | Ver. |
|---|---|---|---|---|---|---|
| OpenVLA-OFT | [2502.19645](https://arxiv.org/abs/2502.19645), 2025-02-27, RSS 2025 | Parallel decoding + action chunking + continuous actions + L1 regression | Yes (fine-tuning) | LIBERO 76.5% -> 97.1%; 26x action-generation throughput; 71.4 Hz / 0.11 s latency on one A100 per [2611.18372](https://arxiv.org/abs/2611.18372) [S] | A100 [S] | [A] |
| FAST / FAST+ | [2501.09747](https://arxiv.org/abs/2501.09747), 2025-01-16 | DCT + BPE action tokenizer | Yes | Enables AR VLAs on dexterous high-frequency tasks; training time up to 5x lower than diffusion pi0 (an **efficient-training** result, not an inference-latency result) | - | [A] |
| PD-VLA | [2503.02310](https://arxiv.org/abs/2503.02310), 2025-03-04, IROS 2025 | Jacobi parallel fixed-point decoding of chunked AR actions | No (training-free) | 2.52x execution frequency | - | [A] |
| Spec-VLA | [2507.22424](https://arxiv.org/abs/2507.22424), 2025-07-30, EMNLP 2025 | Speculative decoding with relaxed acceptance | draft model | 1.42x vs OpenVLA | - | [A] |
| VOTE | [2507.05116](https://arxiv.org/abs/2507.05116), 2025-07-07 | Single `<ACT>` token decoded by MLP + trajectory-ensemble voting | Yes | 39x faster than OpenVLA, 46 Hz "on edge platforms" | Jetson AGX Orin per [2509.11480](https://arxiv.org/abs/2509.11480) (same group; 55.57 Hz for VOTE-MLP4 in MAX mode) [F] | [A]/[F] |
| EdgeVLA | [2507.14049](https://arxiv.org/abs/2507.14049) | Non-autoregressive end-effector prediction | Yes | 7x | - | [A] |
| Zora | [2611.11098](https://arxiv.org/abs/2611.11098), 2026-11-08 | Retrieval-based draft for speculative decoding + custom hardware core | No (software part) | up to 3.4x software-only, 7.6x with hardware support | GPUs + proposed accelerator | [A] |

5b. Fewer denoising / flow steps (consistency, one-step, MeanFlow, shortcut, adaptive solvers, caching)

| Method | arXiv / date / venue | Policy class | Mechanism | Retraining | Headline numbers | Hardware | Ver. |
|---|---|---|---|---|---|---|---|
| Consistency Policy | [2405.07503](https://arxiv.org/abs/2405.07503), 2024-05-13 | Diffusion Policy (not VLA) | Consistency distillation | Yes | "order of magnitude" faster; inference on a laptop GPU | laptop GPU | [A] |
| ManiCM | [2406.01586](https://arxiv.org/abs/2406.01586), 2024-06-03 | 3D diffusion policy | Consistency model | Yes | 10x average inference speed | - | [A] |
| OneDP | [2410.21257](https://arxiv.org/abs/2410.21257), 2024-10-28 | Diffusion Policy | One-step distillation (KL along diffusion chain; +2-10% pretraining cost) | Yes | 1.5 Hz -> 62 Hz | - | [A] |
| FlowPolicy | [2412.04987](https://arxiv.org/abs/2412.04987), 2024-12-06 | 3D flow policy | Consistency flow matching | Yes | 7x inference speed | - | [A] |
| Falcon | [2503.00339](https://arxiv.org/abs/2503.00339), 2025-03-01 | Diffusion policies | Reuse partially denoised actions from history | No | 2-7x | - | [A] |
| MP1 | [2507.10543](https://arxiv.org/abs/2507.10543), 2025-07-14, AAAI 2026 | 3D point-cloud policy | MeanFlow, 1-NFE | Yes | 6.8 ms average inference; 19x faster than DP3, ~2x faster than FlowPolicy | - | [A] |
| LightDP | [2508.00697](https://arxiv.org/abs/2508.00697), 2025-08-01, ICCV 2025 | Diffusion Transformer Policy | Prune denoiser + consistency distillation | Yes | "real-time action prediction on mobile devices" | mobile devices (unnamed in abstract) | [A] |
| SnapFlow | [2604.05656](https://arxiv.org/abs/2604.05656), 2026-04-07 | **pi0 / pi0.5 flow-matching VLAs** | Progressive self-distillation with shortcut samples (own two-step Euler endpoint as target); no separate teacher | Yes | 1-NFE pi0.5: 98.75% LIBERO vs 97.75% teacher (10 NFE); batch-1 model latency ~3.3x lower | - | [A] |
| AdaVLA | [2608.29208](https://arxiv.org/abs/2608.29208), 2026-08-29, IROS 2026 | pi0.5, X-VLA, SmolVLA (real robot) | Training-free adaptive step count from flow-trajectory curvature + adaptive MLP pruning | **No** | 1.87x (pi0.5), 2.24x (X-VLA) with negligible SR loss | **Jetson AGX Orin**; real-robot validation with **SmolVLA** | [A] |
| IMLE-VLA | [2609.10915](https://arxiv.org/abs/2609.10915), 2026-09-10, IROS 2026 | pi0.5 | Single-step conditional generator trained with cIMLE | Yes | 55 Hz vs 15 Hz (3.67x); LIBERO 98.0% | - | [A] |
| Stage-aware two-step denoising | [2609.39822](https://arxiv.org/abs/2609.39822), 2026-09-30 | pi0.5 | Non-uniform 2-step schedule (10 -> 2 steps) | No | model inference 61.557 ms -> 21.956 ms; "small reduction in task performance" | - | [A] |
| FastOPD | [2610.02832](https://arxiv.org/abs/2610.02832), 2026-10-02 | pi0.5, LingBot-VLA, MolmoAct2 | On-policy distillation into few-step compact student | Yes | 2 steps: retains 84% of pi0.5 LIBERO performance, latency -78.1% | - | [A] |
| MambaAct | [2610.22951](https://arxiv.org/abs/2610.22951), 2026-10-16 | pi0.5 | Replace transformer action expert with 10x smaller Mamba expert + context compressor | Yes | denoising latency -64.8% (LIBERO), -58.8% (real) | - | [A] |
| PICES | [2610.19656](https://arxiv.org/abs/2610.19656), 2026-10-14 | action-chunking VLAs | Perturbation-informed adaptive execution horizon (fewer policy calls) | No | up to 2.3x wall-clock policy inference time | **Jetson Orin and Thor** | [A] |
| Also: ActionCache, BAC, EfficientVLA diffusion-feature cache | see Family 2 | | | | | | |

- Shortcut models and MeanFlow as generic generative-model techniques were not separately verified here; within VLAs the "shortcut" idea appears in SnapFlow ([2604.05656](https://arxiv.org/abs/2604.05656) [A]) and MeanFlow in MP1 ([2507.10543](https://arxiv.org/abs/2507.10543) [A]).
- SnapFlow states the problem for SmolVLA-like models directly: flow-matching VLAs integrate "over about ten sequential action-expert evaluations, so action generation dominates model latency and caps the achievable control frequency", and existing remedies either shrink each evaluation or import one-step objectives designed for training from scratch - [2604.05656](https://arxiv.org/abs/2604.05656) [A].

#### Family 6 - Dual-system (fast / slow) designs

| Method | Source / date | System 2 (slow) | System 1 (fast) | Interface | Rates / headline numbers | Hardware | Ver. |
|---|---|---|---|---|---|---|---|
| HiRT | [2410.05273](https://arxiv.org/abs/2410.05273), 2024-09-12, CoRL 2024 | InstructBLIP VLM at low rate | EfficientNet-B3-based policy [S, [2510.17111](https://arxiv.org/abs/2510.17111)] | latent vector | doubles control frequency at comparable SR on static tasks; dynamic tasks 48% -> 75% | - | [A] |
| RoboDual | [2410.08001](https://arxiv.org/abs/2410.08001), 2024-10-10 | OpenVLA generalist | DiT specialist, 20M trainable params | latent + discretized actions | +26.7% real-world vs OpenVLA, +12% CALVIN; 3.8x higher control frequency | - | [A] |
| Helix | [figure.ai/news/helix](https://www.figure.ai/news/helix), 2025-02-20 (company post, not peer-reviewed) | 7B open-weight VLM, 7-9 Hz | 80M cross-attention encoder-decoder transformer, 200 Hz upper-body control | single continuous latent vector; trained end-to-end | ~500 h teleop data | "dual low-power embedded GPUs" on-robot, one per system; no latency figures | [F] |
| GR00T N1 | [2503.14734](https://arxiv.org/abs/2503.14734), 2025-03-18 | VLM (Eagle) | DiT flow-matching head | jointly trained | 2.2B params; 63.9 ms per 16-action chunk on one L40; VLM 10 Hz, actions 120 Hz - per [2611.18372](https://arxiv.org/abs/2611.18372) Table 2 [S] | L40 [S] | [A]/[S] |
| GR00T N1.7 | [github.com/NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) README | Cosmos-Reason2-2B (Qwen3-VL) | flow-matching DiT reduced from 32 to 16 layers, action horizon 40 | - | 3B; TensorRT: ~40 Hz on RTX PRO 6000 Blackwell, ~28 Hz on Jetson AGX Thor (batch-1 LIBERO, MAXN) | **Jetson AGX Thor** | [F] |
| OpenHelix | [2505.03912](https://arxiv.org/abs/2505.03912), 2025-05-06 | LLaVA-7B | 3D Diffuser Actor [S] | special token | survey + empirical analysis + open-source low-cost dual-system model; no numbers in abstract | - | [A] |
| Fast-in-Slow (FiS-VLA) | [2506.01953](https://arxiv.org/abs/2506.01953), 2025-06-02 | full VLM | final transformer blocks of the same VLM reused as System 1 (shared params) | shared layers | 117.7 Hz with chunk 8; +8% sim, +11% real | - | [A] |
| Hume | [2505.21432](https://arxiv.org/abs/2505.21432), 2025-05-27 | value-guided thinking (PaliGemma-based) | lightweight cascaded denoising policy, asynchronous | noisy action candidates | no rates in abstract | - | [A] |
| SP-VLA (model scheduling) | [2506.12723](https://arxiv.org/abs/2506.12723) | VLA for "deliberative" actions | lightweight generator for "intuitive" actions | action-aware router | 1.5x / 2.4x | - | [A] |
| EMS | [2608.06434](https://arxiv.org/abs/2608.06434), 2026-08-06 | large model | small model, environment-aware switching | modular | effective 93.4 Hz on LIBERO | - | [A] |
| VLA-ULAP | [2609.18663](https://arxiv.org/abs/2609.18663), 2026-09-16 | cloud VLA (pi0.5) | ultra-lightweight local action predictor | interleaving | removes 70.3-71.9% of VLA calls on SO-101 without observed SR loss | SO-101 + cloud | [A] |
| SkipVLA | [2609.20648](https://arxiv.org/abs/2609.20648), 2026-09-17 | VLA for contact-rich phases | classical motion planner for free-space motion | pose predictor on frozen backbone | up to 2.5x faster task completion | - | [A] |

- Survey table of dual systems (S1 / S2 / interface): LCB (3D Diffuser Actor / LLaVA / special token), HiRT (EfficientNet-B3 / InstructBLIP / latent), RoboDual (DiT / OpenVLA / latent), OpenHelix (3D Diffuser Actor / LLaVA-7B / special token), FiS (LLaMA2-7B deep layers / shallow layers / latent), Hume (Transformer / PaliGemma-4B / noisy action), HyperVLA (Transformer / T5 / network parameters), SP-VLA (ridge regression / LLaMA-7B / adaptive routing) - [2510.17111](https://arxiv.org/abs/2510.17111) [F].
- The EVLA survey lists SmolVLA itself under "hierarchical systems" (alongside HiRT, DP-VLA, RoboDual, HAMSTER, FiS, Fast ECoT, MinD) - [2510.24795](https://arxiv.org/abs/2510.24795) [F].

#### Family 7 - Distillation (as the main lever)

- Depth distillation of flow VLAs: Shallow-pi, 18 -> 6 layers, losses = flow-matching task loss + teacher-velocity KD + KL on action-to-vision-language cross-attention at a middle layer (matching all tokens gave 0.0 success) - [2601.20262](https://arxiv.org/abs/2601.20262) [F].
- Step distillation of flow VLAs: SnapFlow (1-NFE self-distillation) - [2604.05656](https://arxiv.org/abs/2604.05656) [A]; FastOPD (on-policy, 2 steps) - [2610.02832](https://arxiv.org/abs/2610.02832) [A]; DriftOPD (one-step, reverse-KL; 2610.00317, title only seen) [U].
- Distillation supporting dynamic depth: MoLe-VLA CogKD - [2503.20384](https://arxiv.org/abs/2503.20384) [A]; DySL-VLA skip-aware two-stage KD - [2602.22896](https://arxiv.org/abs/2602.22896) [S]; CEED-VLA consistency distillation - [2506.13725](https://arxiv.org/abs/2506.13725) [A].
- Distillation for recovery after structural pruning: offline hidden-state distillation lifts a 63%-reduced OpenVLA-OFT student (LIBERO-Long 93.2% -> 0.8% after pruning) to within 3.5 points of the teacher in ~8 GPU-hours; on a 6-DoF arm the 72%-reduced student runs 2.23x faster on-board with 62% less memory - [2609.19579](https://arxiv.org/abs/2609.19579) [A]. SPAR prunes pi0.5's language backbone 2.34B -> 0.09B and uses RL to raise LIBERO from 84.55% to 93.45%, 1.98x speedup - [2610.26487](https://arxiv.org/abs/2610.26487) [A].
- Vision-encoder distillation: BitVLA's Quantize-then-Distill (1.58-bit vision encoder with FP teacher) - [2506.07530](https://arxiv.org/abs/2506.07530) [A].
- Reasoning distillation into small models: CoTinyVLA (35B teacher -> 0.9B) - [2607.25487](https://arxiv.org/abs/2607.25487) [A]; RT-VLA for driving (44.8x vision-only / 7.9x vision+language inference-time reduction vs SimLingo) - [2606.14010](https://arxiv.org/abs/2606.14010) [A].

#### Family 8 - Inference runtimes, compilation and systems engineering (no retraining)

| System | Source / date | What it does | Headline numbers | Hardware | Ver. |
|---|---|---|---|---|---|
| vla.cpp | [2606.08094](https://arxiv.org/abs/2606.08094), 2026-06-06 (v2 2026-09-21) | C++ runtime on ggml / llama.cpp; single GGUF bundle per policy; 11 models (SmolVLA, Evo-1, pi0, pi0.5, GR00T-N1.5/1.6/1.7, VLA-JEPA, VLA-Adapter, OpenVLA-OFT, BitVLA); backends CUDA, Metal, CPU, SYCL, OpenVINO; quantization BF16 / Q8_0 / Q4_0 / ternary | RTX 3090 server-side SmolVLA: 55.8 ms vs PyTorch eager 174.1 ms vs torch.compile 65.9 ms (1.18x over the faster PyTorch mode); BitVLA 51.5 vs 319.7 / 94.6 ms (1.84x); pi0, Evo-1, GR00T-N1.6/1.7 ~1.00-1.02x. LIBERO-Object parity e.g. SmolVLA 96 (ref 96.0). GR00T-N1.7 Q4_0: 190/200 vs BF16 196/200, speedup only 1.081x; quantizing the vision tower too ("+V") gives 1.131x (Q8_0+V) / 1.144x (Q4_0+V) | RTX 3060/3090/5070, **Jetson AGX Orin, Jetson Orin Nano 8GB**, Apple M4, Intel Arc, CPUs | [F] |
| "Running VLAs at Real-time Speed" (realtime-vla) | [2510.26742](https://arxiv.org/abs/2510.26742), 2025-10-30 | pi0-level model: CUDA graph (~2x), graph simplification / operator fusion (7-8 ms), Triton GEMM tuning, pass only KV cache to action expert (skip last-layer LLM features, ~0.7 ms), fused scalar ops (~4 ms) | 2 views: naive torch 106.5 ms -> openpi/jax 53.7 ms -> 27.3 ms; 1 view 20.0 ms; 3 views 36.8 ms; roofline says "remaining room of improvement is at most 30%" | single RTX 4090, BF16 | [F] |
| VLA-HWBench | [2610.34717](https://arxiv.org/abs/2610.34717), 2026-10-26 | 18 open VLAs (19 configs), seven-stage breakdown, comparison of torch.compile / CUDA graphs / FlashRT / TensorRT / FP8 / NVFP4 | SmolVLA torch.compile (reduce-overhead): 191 -> 66.4 ms (2.88x) on Thor, 1.28x on AMD; pi0.5 eager -> FlashRT 405 -> 72.8 ms (5.56x), + NVFP4 44.4 ms; GR00T N1.7 TensorRT 1.24x, FlashRT 1.47x, NVFP4 45.6 ms; X-VLA FP32 -> BF16 646 -> 76.7 ms (8.42x) | **Jetson AGX Thor**, AMD Ryzen AI Max+ PRO 395 | [F] |
| "Where Is VLA Inference Time Spent?" (VLA-Triage) | [2611.06710](https://arxiv.org/abs/2611.06710), 2026-11-04 | 7 optimization families x 7 VLAs (incl. SmolVLA) x 4 platforms; interaction analysis | SmolVLA eager bf16 on H100: 208.3 ms; one denoising step eager / compiled+graphs: 18 / 2.8 ms (H100), 40 / 3.9 (A10G), 27 / 3.7 (Thor), 37 / 2.6 (RTX 4090); "rest of inference": 27 / 26, 65 / 65, 46 / 45, 55 / 55 ms | H100, A10G, **Jetson Thor**, RTX 4090 | [F] |
| EagleVLA-Edge engine | [2607.12659](https://arxiv.org/abs/2607.12659), 2026-07-14, CoRL 2026 (Spotlight) | llama.cpp-based engine: CUDA graph reuse, GPU-resident buffers, flow-matching unrolling (plus async algorithm, see Family 9) | pi0.5 on Orin: 1420.8 ms (0.70 Hz, naive PyTorch) -> 271.1 ms total latency / 97.1 ms reaction time / 10.30 Hz; 9.20x control-frequency gain over vla.cpp (893.0 ms); Thor: 457.9 ms -> 132.8 ms / 16.37 Hz | **Jetson Orin** (30 W / 50 W), **Jetson Thor** | [F via alphaxiv mirror] |
| GR00T TensorRT pipeline | [Isaac-GR00T README](https://github.com/NVIDIA/Isaac-GR00T) | ONNX + TensorRT full-pipeline export | ~28 Hz on Thor, ~40 Hz on RTX PRO 6000 Blackwell (N1.7) | Thor | [F] |
| eetmie/jetson-orin-nano-vla (community, single author) | [GitHub README](https://github.com/eetmie/jetson-orin-nano-vla), 2026 | Split ONNX bundle (vision / language / denoise graphs) -> per-piece TensorRT engines chained on GPU, replayed as one CUDA graph; AOT Triton kernels; mostly FP16; parity check cosine >= 0.99999 vs stock policy | see Q4 | **Jetson Orin Nano Super 8GB** | [F] |
| OxyGen | [2603.14371](https://arxiv.org/abs/2603.14371), 2026-03-15, NeurIPS 2026 | Unified KV-cache management across tasks and control cycles for MoT VLAs (cross-task KV sharing, cross-frame continuous batching) | up to 3.7x over isolated execution; >200 tokens/s language and 70 Hz action simultaneously | RTX 4090, **Jetson AGX Thor** | [A] |
| Reflex (2607.14695) | [2607.14695](https://arxiv.org/abs/2607.14695), 2026-07-16, ICML 2026 | Streaming inference for flow VLAs: fixes "global timestep injection invalidates KV-[cache]" problem; AdaRMSNorm against BF16 collapse; async pipeline decoupling **visual encoding** from action generation; operator fusion | 2.58x speedup, 50 Hz stable streaming, reaction latency -54% | - | [A] |
| RhinoVLA deployment stack | [2606.07383](https://arxiv.org/abs/2606.07383), 2026-06-05 (company technical report) | Hardware-aware compilation, W8A16 mixed precision, **parallel (batched) visual encoding** of 3 views | 5.84 Hz -> 11.69 Hz end-to-end; 3-view encoding 34.52 -> 24.31 ms | Huixi R1 (vendor SoC); profiles pi0.5 on Jetson AGX Orin | [F] |
| LiteVLA-Edge | [2603.03380](https://arxiv.org/abs/2603.03380), 2026-03-03 | SmolVLM-256M-based VLA, LoRA fine-tune, GGUF Q4_K_M via llama.cpp CUDA, ROS 2 bridge | 150.5 ms mean (6.6 Hz), std ~0.13 ms; **no task-success evaluation** | Jetson AGX Orin 64GB (paper also says Orin NX in one section - internal inconsistency) | [F] |
| Hardware co-design | EdgeDAE [2610.00311](https://arxiv.org/abs/2610.00311) (ASP-DAC 2027); Deltoris [2608.04428](https://arxiv.org/abs/2608.04428); Zora [2611.11098](https://arxiv.org/abs/2611.11098) | FPGA offload of the diffusion action expert; bit-sparsity accelerator; RAG-draft speculative core | EdgeDAE: end-to-end latency -52.5% (Octo-Small) / -38.7% (Octo-Base) vs edge GPU; Deltoris: up to 34.2x over mobile GPUs | FPGA-GPU; simulated accelerators | [A] |
| Serving / co-inference | Robion [2609.12075](https://arxiv.org/abs/2609.12075); EcoVLA [2608.15502](https://arxiv.org/abs/2608.15502) | Multi-robot GPU serving; device-edge split | Robion: up to 64 robots on 4 GPUs within 98% SLO; EcoVLA: energy efficiency up to +236% under a 20 Hz constraint | servers / device-edge | [A] |

#### Family 9 - Asynchronous and latency-aware execution (adjacent family; speeds up the *robot*, not the model)

- SmolVLA async inference stack: sync total 137.5 s (avg 13.75 s/episode) vs async 97.0 s (avg 9.70 s), ~30% faster; fixed-time test 19 vs 9 pick-place cycles; but success 78.3% (sync) vs 73.3% (async) averaged over three tasks - [2506.01844](https://arxiv.org/abs/2506.01844) [F].
- Real-Time Chunking (RTC): inference-time inpainting, "applicable to any diffusion- or flow-based VLA out of the box with no re-training", NeurIPS 2025 - [2506.07339](https://arxiv.org/abs/2506.07339) [A]; training-time RTC as cheaper drop-in - [2512.05964](https://arxiv.org/abs/2512.05964) [A].
- VLASH: future-state-aware async inference, reaction latency up to 11.8x lower than synchronous; 1.5-2.0x task-completion speedup with action quantization - [2512.01031](https://arxiv.org/abs/2512.01031) [A].
- A2C2 correction head (+23 pp on Kinetix, +7 pp on LIBERO-Spatial vs RTC) - [2509.23224](https://arxiv.org/abs/2509.23224) [A]; FreeBar barrier-constrained guidance (latency per chunk >45% lower than RTC), CoRL 2026 - [2610.14997](https://arxiv.org/abs/2610.14997) [A]; Action ControlNet delay-aware adapter evaluated on SO-ARM101 - [2606.25985](https://arxiv.org/abs/2606.25985) [A]; FlashVLA streaming decoding (>=30 Hz on a single GPU) - [2608.27384](https://arxiv.org/abs/2608.27384) [A].
- EagleVLA: ~40M-param future-correction module predicting the compressed VLM final-layer state at a future step + confidence-based scheduler deciding when to call the VLM vs only the action expert; LIBERO pi0.5 97.4/98.6/96.8/92.5 vs synchronous 97.3/99.6/96.7/93.5; +14.8% over VLASH, +3.9% over RTC - [2607.12659](https://arxiv.org/abs/2607.12659) [F via alphaxiv mirror].
- Path-preserving Time Scaling (PTS): training-free execution layer; throughput +33% (pi0.5) and **+104% (SmolVLA)** over the strongest deployable baseline under injected latency - [2611.01390](https://arxiv.org/abs/2611.01390) [A].
- Six real-time execution methods compared on physical garment folding with pi0.5: "Legato performs best overall among training-based methods, while Temporal Smoothing leads among training-free methods" - [2609.39822](https://arxiv.org/abs/2609.39822) [A].

### Inferences
- The two 2025 surveys were written before most quantization, runtime and measurement papers existed; their taxonomies therefore under-weight Families 4, 8, 9 and the evaluation-validity strand. A 2026 prior-art section should not rely on their category boundaries alone.
- Almost all token-pruning, caching, quantization and early-exit work up to late 2025 targets 7B OpenVLA / OpenVLA-OFT / CogACT (large LLM, autoregressive or parallel-decoded). Work whose *subject* is a flow-matching VLA with a separate action expert appears mainly from 2026 (Shallow-pi, SnapFlow, AdaVLA, HoloQ-VLA, CHASE-VLA, FoldQuantVLA, ActionCache, MambaAct, Reflex, vla.cpp, EagleVLA), and it mostly uses pi0 / pi0.5 / GR00T, not SmolVLA.
- SmolVLA appears as a *measured subject* in vla.cpp, VLA-HWBench, VLA-Triage (2611.06710), Shallow-pi (as a latency baseline), AdaVLA (real-robot validation), PTS, NanoVLA and Evo-1 (as a baseline), but in this search no paper was found whose main contribution is an optimization method designed for and evaluated on SmolVLA-class models on an Orin Nano.

### Gaps
- MiniVLA: no arXiv ID exists as far as could be determined; the Stanford blog was unreachable (HTTP 503), so its parameters, encoders and results are unverified here.
- DySL-VLA and QuantVLA numbers come from search summaries of the arXiv listings, not from full text.
- "Shortcut models" and "MeanFlow" as standalone generative-model papers were not fetched; no dedicated "MeanFlow VLA" or "shortcut VLA" paper was verified beyond SnapFlow (shortcut-style) and MP1 (MeanFlow, non-VLA).
- HiRT / RoboDual / FiS / OpenHelix hardware and per-system latency were not available from abstracts.
- Titles seen but not opened (not verified beyond title/date): DriftOPD (2610.00317), Scene-Adaptive VLA (2610.24590), AutoPruner (2610.23880), SACQ (2610.41408), AGQ (2610.33074), VLA-ACL (2610.08133), BFA++ (2602.20566), DTP (2601.16065), SAFE-Pruner (2605.29662), UniFS (2606.22794), CondenseVLA (2610.25927).

---

## Q2. For each family: typical speedup, hardware, and is training required?

### Takeaway
Reported gains fall into three bands: training-free token/caching/quantization tricks give roughly 1.2-2x end-to-end on the models they target; architectural or training-based changes (parallel decoding, one-step heads, depth distillation, small backbones) give 2-26x relative to a slow baseline; and pure runtime engineering gives 1-5.6x depending heavily on how bad the eager-PyTorch baseline was. Most numbers are from desktop or datacenter GPUs, and 2026 measurement studies show the bands do not compose multiplicatively.

### Cited Findings

| Family | Typical reported gain (range from papers above) | Typical hardware | Training needed? | Sources |
|---|---|---|---|---|
| 1. Small backbone / static layer removal | Model-level: TinyVLA 14 ms vs OpenVLA 292 ms [S]; VLA-Adapter 219.2 Hz vs OpenVLA-OFT 71.4 Hz; Evo-1 16.4 Hz vs pi0 11.5 Hz; NanoVLA "52x FPS" vs OpenVLA; Shallow-pi >2x (H100 25.5 -> 11.3 ms; Orin 364 -> 110 ms) | A6000, H100, RTX 4090(d); Orin Nano Super (NanoVLA); Orin / Thor (Shallow-pi) | Yes (new model or distillation) | [2611.18372](https://arxiv.org/abs/2611.18372), [2509.09372](https://arxiv.org/abs/2509.09372), [2511.04555](https://arxiv.org/abs/2511.04555), [2510.25122](https://arxiv.org/abs/2510.25122), [2601.20262](https://arxiv.org/abs/2601.20262) |
| 2. Token pruning / merging / caching | 1.35x-1.99x end-to-end on 7B-class VLAs (ADP 1.35x, SP-VLA 1.5x, SpecPrune 1.57x, ActPrune 1.61x, VLA-Cache 1.7x, SAPrune 1.718x, EfficientVLA 1.93x, VLA-Pruner 1.99x); latency -36% (FlashVLA) / -38.2% (LightVLA); CogVLA 2.8x (with parallel decoding); prefill-only: -48% on AGX Orin (TRIAGE), 1.78x (GeoScaffold); outlier 8.3x at 1024x1024 (ST-Merge) | Mostly unspecified desktop/server GPUs; AGX Orin (TRIAGE); Thor (rMuscle) | Mostly **no**; yes for LightVLA, CogVLA, learned token caching, KV-Efficient | rows in Family 2 |
| 3. Dynamic depth / early exit | LLM-compute -5.2-6.5x (DeeR, LLM only); compute up to 5.6x lower (MoLe); 3.75x latency vs RoboFlamingo (DySL); >4x (CEED, decoding iterations) | Not stated in abstracts (simulation benchmarks CALVIN / RLBench / LIBERO) | **Yes** in all cases | [2411.02359](https://arxiv.org/abs/2411.02359), [2503.20384](https://arxiv.org/abs/2503.20384), [2602.22896](https://arxiv.org/abs/2602.22896), [2506.13725](https://arxiv.org/abs/2506.13725) |
| 4. Quantization / binarization | Memory: -42.9% to -74% typical, 2-3x for real low-bit models (LoopQ), up to 11.0x (BitVLA), 8x (RLRC). Speed: 1.22x (QuantVLA), 1.20-1.33x on Orin (FoldQuantVLA, native W4A4 TensorRT plugins), 1.49x (QVLA), 1.52x (Mix-QVLA), 1.73x on Thor with NVFP4 (QCRL), 1.9x (DyQ-VLA, SQAP-VLA), 2.5x on an edge GPU (QAIL / SQIL), 4.4x latency (BitVLA vs OpenVLA-OFT); **or none / negative**: on H100 at batch 1 no quantized pi0.5 or OpenVLA-OFT configuration is faster than 16-bit (1.06-1.58x slower), Thor slower on every 4-bit configuration (LoopQ) | H100, L4, Thor, AGX Orin, "edge GPU" | PTQ: no (calibration only). QAT / native low-bit (QAIL, SQIL, BitVLA) and recovery (RLRC, QCRL, PIVOT-Q): yes | rows in Family 4; [2611.05093](https://arxiv.org/abs/2611.05093) |
| 5a. Decoding scheme (AR VLAs) | 26x throughput (OpenVLA-OFT), 39x vs OpenVLA (VOTE), 7x (EdgeVLA), 2.52x (PD-VLA, training-free), 1.42x (Spec-VLA) | A100 (OFT) [S]; AGX Orin (VOTE) | Yes, except PD-VLA / speculative decoding | rows in 5a |
| 5b. Fewer denoising steps | Step-count reductions 10 -> 1 or 2: ~3.3x model latency (SnapFlow), 3.67x Hz (IMLE-VLA), 61.6 -> 22.0 ms (two-step, training-free), -78.1% latency (FastOPD, retains 84%); training-free adaptive: 1.87-2.24x on AGX Orin (AdaVLA), 2-7x (Falcon); non-VLA one-step policies ~10x (Consistency Policy, ManiCM), 1.5 -> 62 Hz (OneDP) | Mostly unspecified; AGX Orin (AdaVLA); Orin / Thor (PICES) | Distillation / new head: yes. Adaptive solvers, caching, two-step schedules: no | rows in 5b |
| 6. Dual-system | Control-rate decoupling rather than model speedup: S2 7-9 Hz / S1 200 Hz (Helix); VLM 10 Hz / actions 120 Hz (GR00T N1) [S]; 117.7 Hz (FiS, chunk 8); 3.8x control frequency (RoboDual); 2x (HiRT) | Embedded GPUs on robot (Helix, unnamed); L40 (GR00T N1) [S] | Yes (joint or staged training) | rows in Family 6 |
| 7. Distillation | >2x (Shallow-pi depth); ~3.3x (SnapFlow steps); 2.23x on-board + 62% less memory (hidden-state distillation after 72% pruning) | H100, Orin, Thor | Yes; Shallow-pi notes higher training-time memory (teacher + student loaded) | [2601.20262](https://arxiv.org/abs/2601.20262), [2604.05656](https://arxiv.org/abs/2604.05656), [2609.19579](https://arxiv.org/abs/2609.19579) |
| 8. Runtimes / compilation | Baseline-dependent: SmolVLA eager 174.1 -> 55.8 ms (3.1x) but only 1.18x over torch.compile (vla.cpp, RTX 3090); SmolVLA 2.88x from torch.compile on Thor; pi0 106.5 -> 27.3 ms on RTX 4090 (realtime-vla); pi0.5 5.56x eager -> FlashRT on Thor; GR00T N1.7 TensorRT 1.24x; several models ~1.00x in vla.cpp | RTX 3060-5070, RTX 4090, Thor, AGX Orin, Orin Nano | **No** | [2606.08094](https://arxiv.org/abs/2606.08094), [2610.34717](https://arxiv.org/abs/2610.34717), [2510.26742](https://arxiv.org/abs/2510.26742) |
| 9. Async / latency-aware execution | Task-level: ~30% faster completion, 2x cycles in fixed time (SmolVLA async); reaction latency up to 11.8x lower (VLASH); 14.71x control frequency on Orin when combined with an engine (EagleVLA); throughput +104% for SmolVLA (PTS) | SO100 (SmolVLA); Orin / Thor (EagleVLA) | RTC, VLASH (state roll-forward), PTS: no; A2C2, ACNet, EagleVLA correction module: small trained module | rows in Family 9 |

- Stacking is not multiplicative: "Stacked gains cannot be predicted reliably from standalone measurements"; of 84 compilation-and-pruning configurations 8 exceeded the percentage-reuse prediction by more than 5.3% (largest gap 19%), and in 29 of 32 flow-model comparisons latency exceeded the prediction by more than 5.3%, up to 227%; compilation speeds individual pi0.5 and SmolVLA denoising steps by more than 2x, "which shrinks the savings from removing steps" - [2611.06710](https://arxiv.org/abs/2611.06710) [F].
- Token-pruning savings shrink once attention is already efficient: FastV at 50% saves 21.4% against unfused attention but only 6.9% against FlashAttention-2 on H100; for CogACT on A10G the pruning saving falls from 74.2 ms to 33.5 ms once compilation is applied - [2611.06710](https://arxiv.org/abs/2611.06710) [F].
- Latency follows the number of action-generation iterations, not model size: parameter count explains at most 15% of the variance in log latency across 19 configurations; action generation is the largest stage in 14 of 19 Jetson Thor configurations - [2610.34717](https://arxiv.org/abs/2610.34717) [F].
- For quantization, the engine alone changes latency by 2.1x, and pi0.5's memory-bound action expert (63% of GPU time, 13% of parameters) stays in 16 bits in the evaluated methods - [2611.05093](https://arxiv.org/abs/2611.05093) [A].

### Inferences
- For a SmolVLA-class model the 7B-era token-pruning speedups (1.3-2x) are unlikely to transfer at face value: SmolVLA already feeds only 64 visual tokens per frame to a 16-layer truncated LLM, and on Thor its prefill stage is only 15.0 ms of a 187 ms eager call (8%) per VLA-HWBench, so LLM-side token pruning attacks a small slice.
- The biggest training-free lever demonstrated on SmolVLA so far is execution mode (eager -> compiled / graph-captured / C++ runtime): 2.9-3.1x in two independent studies (VLA-HWBench on Thor, vla.cpp on RTX 3090), with a further "floor" set by the non-iterative part (~45 ms on Thor after compilation per 2611.06710).

### Gaps
- No single paper reports all families on one model and one edge device; the closest are VLA-HWBench (Thor only, no token pruning/caching) and VLA-Triage (Thor is the only embedded platform; latency only, no accuracy).
- Energy / power numbers are rare: QAIL / SQIL (2.5x energy saving), EdgeDAE, EcoVLA, VLA-ULAP report energy; VLA-HWBench explicitly reports no measured power.

---

## Q3. Which methods actually reduce the cost of the vision encoder itself (as opposed to tokens inside the LLM)?

### Takeaway
Very few. The dominant token-pruning / caching literature selects or reuses tokens *after* the vision encoder and saves LLM prefill only; methods that touch the encoder itself are limited to (a) swapping in a cheaper encoder or fewer/lower-resolution views at design time, (b) quantizing / binarizing the vision tower, (c) batching or asynchronously scheduling the encoder, and (d) a handful of 2026 preprints that merge or incrementally update tokens inside or before the encoder. Independent profiling does support that in small flow VLAs the vision encoder is a larger share than the VLM prefix, but the iterative action expert, not the encoder, is still the largest stage at 10 flow steps in eager mode.

### Cited Findings

**(a) Profiling evidence on where time goes (vision vs prefix vs action expert)**

| Model / device / mode | Vision encoder | VLM prefix / backbone | Action expert / generation | Total | Source |
|---|---|---|---|---|---|
| SmolVLA, Jetson AGX Thor, PyTorch eager (LeRobot 0.6.1) | 20.9 ms (11%) | 15.0 ms (8%) | 146.6 ms (78%) | 187 ms | [2610.34717](https://arxiv.org/abs/2610.34717) [F] |
| SmolVLA, AMD Ryzen AI Max+ 395, eager | 40.8 ms | 10.9 ms | 87.5 ms (~63%) | 140 ms | [2610.34717](https://arxiv.org/abs/2610.34717) [F] |
| SmolVLA, Thor: fitted per-step cost 14.5 ms, fixed cost 41 ms (eager); with compile + graphs one step = 3.7 ms and "rest of inference" = 45 ms | (inside the 45 ms "rest") | (inside "rest") | 10 x 3.7 = 37 ms compiled | ~82 ms implied | [2610.34717](https://arxiv.org/abs/2610.34717), [2611.06710](https://arxiv.org/abs/2611.06710) [F] |
| SmolVLA, RTX 5070, vla.cpp, T = 10 | 38% | 14% | remainder | - | [2606.08094](https://arxiv.org/abs/2606.08094) [F] |
| pi0 (2 views), RTX 4090, fully optimized | 4.059 ms (15%) | 12.503 ms (46%) | 11.001 ms (40%) | 27.3 ms | [2510.26742](https://arxiv.org/abs/2510.26742) [F] |
| pi0, RTX 5070, vla.cpp, T = 10 | 21% | 54% | remainder | - | [2606.08094](https://arxiv.org/abs/2606.08094) [F] |
| pi0.5, Jetson AGX Thor, eager | 163.1 ms | 69.4 ms | 168.1 ms | 403 ms | [2610.34717](https://arxiv.org/abs/2610.34717) [F] |
| pi0.5, Jetson AGX Orin, PyTorch-SDPA | 69.3 ms (8%) | 528.0 ms | 257.0 ms | ~858 ms | [2606.07383](https://arxiv.org/abs/2606.07383) [F] |
| GR00T N1.7, RTX 3060, vla.cpp, T = 4 | vision 18.5-18.9 ms (vision + backbone 58%) | - | 42% | - | [2606.08094](https://arxiv.org/abs/2606.08094) [F] |
| GR00T N1.7, Thor, eager | 19.6 ms | 19.4 ms | 67.7 ms | 123 ms | [2610.34717](https://arxiv.org/abs/2610.34717) [F] |
| RhinoVLA (2.5B), Huixi R1, optimized | 24.31 ms (28.4%, 3 views) | 20.78 ms (24.3%) | 36.71 ms (42.9%) | 85.54 ms | [2606.07383](https://arxiv.org/abs/2606.07383) [F] |
| LiteVLA-H, AGX Orin | 18.2 ms | 47.8 ms multimodal prefill | - | - | [2605.00884](https://arxiv.org/abs/2605.00884) [S, search snippet] |
| MolmoAct-7B, AGX Orin / Thor | not quantified | - | generation ~75% | - | [2603.02271](https://arxiv.org/abs/2603.02271) [F] |

- After optimization the bottleneck moves to stages that were not optimized: CPU preprocessing grows from 12% to 39% of the GR00T N1.7 call - [2610.34717](https://arxiv.org/abs/2610.34717) [F].
- Roofline view for pi0 on RTX 4090: "most operations in vision encoder and LLM are computation-limited, and most operations in the AE are bandwidth-limited" - [2510.26742](https://arxiv.org/abs/2510.26742) [F]. On Thor, memory bandwidth limits "only the stages that process few tokens ..., such as the denoising steps, whereas software limits all other stages" - [2610.34717](https://arxiv.org/abs/2610.34717) [A].

**(b) Methods whose saving is located in or before the vision encoder**

| Method | What it does to the encoder | Evidence / number | Caveat | Source |
|---|---|---|---|---|
| SmolVLA | Uses only the global image (no tiling), so the SigLIP encoder is run once per view; pixel shuffle then reduces to 64 tokens (the shuffle is post-encoder) | No encoder-latency number in paper | No ablation on token count / tiling reported in the fetched text | [2506.01844](https://arxiv.org/abs/2506.01844) [F] |
| NanoVLA | Replaces ViT with **ResNet18**; caches the *language* embedding once per task | Per-task time saving 62% (Qwen 0.5B) / 35% (BERT-base) from language caching | "The visual embedding is updated every frame, so the vision encoder still runs each step" | [2510.25122](https://arxiv.org/abs/2510.25122) [F] |
| HiRT / Helix / GR00T-style dual systems | System 1 has its own lightweight visual pathway (EfficientNet-B3 in HiRT; 80M S1 in Helix); large VLM vision tower runs at low rate | HiRT 2x control frequency; Helix 7-9 Hz vs 200 Hz | Requires dual-system training | [2410.05273](https://arxiv.org/abs/2410.05273), [2510.17111](https://arxiv.org/abs/2510.17111), [Helix](https://www.figure.ai/news/helix) |
| TurboVLA | Independent vision and language encoders with light bidirectional interaction, no LLM | 31.2 ms, 0.9 GiB on RTX 4090 | Encoder cost not separately reported in abstract | [2607.27205](https://arxiv.org/abs/2607.27205) [A] |
| BitVLA | Quantize-then-Distill compresses the **vision encoder to 1.58-bit** | Whole-model memory -11.0x, latency -4.4x vs OpenVLA-OFT; in vla.cpp BitVLA is "about 99.5% of time in the prefix/backbone" and the ternary tensor-core path gives 4.57x (RTX 3060) / 4.02x (AGX Orin) | Speed needs a ternary-aware runtime (CUDA-specific tensor-core path) | [2506.07530](https://arxiv.org/abs/2506.07530) [A], [2606.08094](https://arxiv.org/abs/2606.08094) [F] |
| vla.cpp "+V" variants | Also quantizes the vision tower (GGUF Q8_0+V / Q4_0+V) | GR00T-N1.7 on RTX 3060: speedup 1.077x -> 1.131x (Q8) and 1.081x -> 1.144x (Q4); success 195/200 for both +V variants vs 196/200 BF16 | Gains are a few percent | [2606.08094](https://arxiv.org/abs/2606.08094) [F] |
| RhinoVLA | Batches the three camera views through the encoder; chooses a VLM with 64 merged tokens/image | 3-view encoding 34.52 -> 24.31 ms | Vendor SoC; technical report with internal inconsistencies | [2606.07383](https://arxiv.org/abs/2606.07383) [F] |
| Reflex (2607.14695) | Async pipeline that "decouples visual encoding from action generation" | 2.58x overall, 50 Hz streaming | Scheduling, not less encoder compute | [2607.14695](https://arxiv.org/abs/2607.14695) [A] |
| CogVLA (EFA-Routing) | FiLM-injects the instruction **inside** SigLIP / DINOv2 and aggregates to 64 tokens per branch (25% of original) | Whole-model FLOPs 8.48T -> 2.72T | "The paper doesn't report vision encoder FLOPs separately"; ablation savings "mostly reflect LLM-side savings"; requires training | [2508.21046](https://arxiv.org/abs/2508.21046) [F] |
| ST-Merge | Merges redundant tokens "directly during the visual encoding phase" across space and time | 8.3x on pi0.5 at 1024x1024 | Small accuracy drop at lower resolutions; preprint | [2606.29350](https://arxiv.org/abs/2606.29350) [A] |
| "Reducing Temporal Redundancy" | "incrementally update only tokens corresponding to dynamic scene regions instead of re-encoding entire frames" | >2x combined with a 2-step schedule | Preprint; perception-side share not separated in abstract | [2607.12287](https://arxiv.org/abs/2607.12287) [A] |
| rMuscle | Context cache "reuses visual-token outputs" across executions | 1.29-1.42x on RTX 4090 / Thor | Factory-style repeated executions | [2609.19104](https://arxiv.org/abs/2609.19104) [A] |

**(c) Methods that, per their own descriptions, act on tokens inside the LLM (encoder untouched)**

- VLA-Cache reuses "cached key-value representations" of static visual tokens with a reuse ratio adjusted "across decoder layers" - [2502.02175](https://arxiv.org/abs/2502.02175) [A].
- EfficientVLA prunes "layers from the language module", selects visual tokens for the LLM, and caches "intermediate features" in the diffusion action head - [2506.10100](https://arxiv.org/abs/2506.10100) [A].
- LightVLA targets "the heavy attention-based computation over large sets of visual tokens" - [2509.12594](https://arxiv.org/abs/2509.12594) [A]; FlashVLA (2505.21200), SP-VLA, SpecPrune-VLA, ADP, VLA-Pruner, SAPrune, GeoScaffold, FastVLA, TRIAGE all describe LLM-side token selection, and several report "prefill" or "LLM-backbone" metrics explicitly (GeoScaffold: prefill speedup; FastVLA: LLM-backbone FLOPs; TRIAGE: prefill latency; T3R: LLM-forward profiling) - [2609.36967](https://arxiv.org/abs/2609.36967), [2610.22146](https://arxiv.org/abs/2610.22146), [2610.34519](https://arxiv.org/abs/2610.34519), [2611.08413](https://arxiv.org/abs/2611.08413) [A].
- Quantization papers mostly exclude the encoder: QuantVLA quantizes linear layers of the language backbone and DiT head ([2602.20309](https://arxiv.org/abs/2602.20309) [S]); ActQuant is backbone weight-only ([2605.24011](https://arxiv.org/abs/2605.24011) [A]); in the VLA-Triage study "Quantization covers only language-model linear layers. The vision encoder and action head are not quantized" ([2611.06710](https://arxiv.org/abs/2611.06710) [F]). On an Orin Nano, TensorRT INT8 calibration for ACT "quantizes the ResNet18 backbone but accepts zero of 145 transformer layers" ([2608.03938](https://arxiv.org/abs/2608.03938) [A]) - i.e. the opposite pattern (CNN backbone quantizes, transformers do not) with stock tooling.
- The survey framing matches: [2510.17111](https://arxiv.org/abs/2510.17111) [F] says visual tokens dominate the token sequence and compute footprint and names SigLIP / DINOv2, "but no encoder cost figures are given"; [2510.24795](https://arxiv.org/abs/2510.24795) [F] does not analyze vision-encoder cost at all.
- Shallow-pi reduces only transformer depth in the VLM backbone and action head and lists "visual token reduction and diffusion step reduction" as future work - [2601.20262](https://arxiv.org/abs/2601.20262) [F].

### Inferences
- The premise "in small VLAs the vision encoder dominates" needs qualification when cited. Supported: for SmolVLA the encoder costs more than the VLM prefix in every profile found (20.9 vs 15.0 ms on Thor; 38% vs 14% in vla.cpp on RTX 5070), the reverse of pi0 / pi0.5 on desktop GPUs (15% vs 46% on RTX 4090). Not supported as stated: with 10 flow steps in eager PyTorch the action expert is the largest stage (78% on Thor). The encoder (plus prefix and fixed overhead) becomes the dominant *floor* only once the denoising loop is compiled / graph-captured (10 x 3.7 ms = 37 ms vs 45 ms "rest" on Thor) or the step count is cut. This is derived from combining two papers' tables, not stated by either.
- Consequently the encoder becomes the binding constraint precisely in the regime a Jetson deployment paper would aim for (compiled runtime + few flow steps + KV-cached prefix), and that regime is where the existing token-pruning literature offers the least, because it leaves the encoder untouched.
- No verified peer-reviewed method was found that skips or partially re-runs a ViT / SigLIP encoder across control steps for a flow-matching VLA with reported closed-loop success on edge hardware; the closest items are 2026 preprints (ST-Merge, temporal-redundancy reduction, rMuscle), all read at abstract level only.

### Gaps
- No per-stage SmolVLA profile on a Jetson Orin Nano 8GB from a paper was found; the community repo says per-stage timings exist in its JSON outputs but the README does not print them.
- Whether TTF-VLA, VLA-Cache or CogVLA reduce any encoder-side compute in practice could not be confirmed from full text (only abstracts / partial HTML).
- Encoder-internal token pruning or merging from the general ViT literature (e.g. ToMe-style) applied to VLA encoders was not searched specifically.

---

## Q4. Which methods have been evaluated on real edge hardware (Jetson Orin Nano / NX / AGX) versus desktop / server GPUs?

### Takeaway
Until late 2025 almost every efficiency paper reported numbers on desktop or datacenter GPUs (RTX 4090, A6000, A100, H100) or gave no hardware at all. Jetson results appear mainly in 2026 and concentrate on AGX Orin (64 GB) and AGX Thor (128 GB); evidence on the 8 GB Orin Nano is limited to vla.cpp, NanoVLA, one community benchmark repository and one non-VLA (ACT) paper. No Orin NX VLA result was verified.

### Cited Findings

**Jetson Orin Nano (8 GB) - the target class**

| Work | Model(s) | Runtime | Numbers | Notes | Source |
|---|---|---|---|---|---|
| vla.cpp | SmolVLA, BitVLA, Evo-1, GR00T-N1.5, pi0 | vla.cpp (ggml / CUDA) | "Client inference time per executed action" (Table III): SmolVLA 141.81 ms (S=4), BitVLA 355.65 ms (S=8), Evo-1 458.84 ms (S=8), GR00T-N1.5 84.76 ms (S=16, remote simulator), pi0 39.10 ms (S=32/50, remote simulator); GR00T-N1.6 / N1.7 **OOM**. BitVLA 200/200 LIBERO-Object on the Nano; BitVLA peak process RSS 2199 MiB, peak system memory 6504 MiB | Metric is amortized per executed action (S appears to be actions executed per chunk - reading not confirmed); same table gives SmolVLA 65.41 ms on AGX Orin and 28.16 ms on RTX 3060 | [2606.08094](https://arxiv.org/abs/2606.08094) [F] |
| NanoVLA | NanoVLA-S/L/R vs OpenVLA, SmolVLA | not specified in fetched text | "about 52x higher FPS and +13.8% SR" vs OpenVLA (OpenVLA measured with only its 4-bit Llama2 backbone because full 4-bit OpenVLA ran out of memory); vs SmolVLA +43.8% FPS at 50 AC steps | Orin Nano **Super** dev kit; exact FPS only in figures | [2510.25122](https://arxiv.org/abs/2510.25122) [F] |
| eetmie/jetson-orin-nano-vla (community repo, single author, not peer-reviewed) | SmolVLA 450M, Evo-1 775M, X-VLA 0.9B, GR00T N1.6 / N1.7 3B, pi0.5 3.3B | split ONNX -> TensorRT engines -> one CUDA graph, AOT Triton kernels, mostly FP16 | p50 latency / rate / whole-board RAM: SmolVLA (2 cams) 107.55 ms / 9.30 Hz / 1.78 GB; Evo-1 332.77 ms / 3.00 Hz / 2.76 GB; X-VLA (3 cams) 342.59 ms / 2.92 Hz / 3.04 GB; GR00T N1.7 273.70 ms / 3.66 Hz / 5.78 GB; GR00T N1.6 273.19 ms / 3.66 Hz / 5.33 GB; pi0.5 LIBERO (2 cams) 438.47 ms / 2.28 Hz / 6.39 GB; pi0.5 base (3 cams) 643.39 ms / 1.55 Hz / 6.18 GB; single-view SmolVLA "almost 12 Hz" | Orin Nano Super, JetPack 7.2.1, MAXN_SUPER, 300 s runs on **synthetic observations**, default denoising steps, parity cosine >= 0.99999; no task-success evaluation; "most models are too large for a single TensorRT engine on 8 GB, hence the split" | [GitHub README](https://github.com/eetmie/jetson-orin-nano-vla) [F] |
| Same repo, earlier figures surfaced by web search | SmolVLA, 2 cams | PyTorch FP32 vs split ONNX FP16 | 1167.93 ms (0.86 Hz) vs 189.89 ms (5.25 Hz) | **Conflicts with the README fetched directly (107.55 ms)**; probably an older revision or a different pipeline stage. The companion HF page (eetmie/smolvla-base-onnx) describes nine ONNX graphs, two 512x512 cameras, a 50-action chunk and ten denoising steps | search snippet [S] |
| Quantized ACT (not a VLA) | ACT, bimanual | TensorRT FP32 / FP16 / INT8 | 114.02 ms -> 17.93 ms (FP16, 6.4x) -> 12.65 ms (INT8, 9.0x); success 19/20, 18/20, 19/20 | Useful as the only peer-style paper found with closed-loop success on an entry-level Jetson | [2608.03938](https://arxiv.org/abs/2608.03938) [A] |

**Jetson AGX Orin (32 / 64 GB)**

| Work | What was measured | Source |
|---|---|---|
| Cross-Platform Scaling of VLA Models from Edge to Cloud GPUs | OpenVLA, SpatialVLA, OpenVLA-OFT, QwenVLA, VOTE on AGX Orin (15 / 30 / 50 W / MAX) vs H100 / A100 / A6000 / V100; Orin MAX: VOTE-MLP4 55.57 Hz vs OpenVLA 1.20 Hz; peak memory on Orin: OpenVLA 14.35 GB, OpenVLA-OFT 19.20 GB, SpatialVLA 7.82 GB, QwenVLA 7.39 GB; throughput "falls non-linearly as power drops from 50 W to 30 W". (The per-chunk latency values as extracted, e.g. "0.03 ms" on H100, look like a unit problem and should be re-read from the PDF before citing.) | [2509.11480](https://arxiv.org/abs/2509.11480) [F] |
| Shallow-pi | pi0 364 ms -> 110 ms; SmolVLA 230 ms on the same Orin (ALOHA) | [2601.20262](https://arxiv.org/abs/2601.20262) [F] |
| EagleVLA | pi0.5: 1420.8 ms naive PyTorch -> 271.1 ms total / 97.1 ms reaction / 10.30 Hz; vla.cpp 893.0 ms; real folding task on X2-W: RTX 4090 baseline 10/10, 7/10, 10/10 vs naive async on Orin 6/10, 0/10, 5/10 vs EagleVLA 10/10, 8/10, 9/10 | [2607.12659](https://arxiv.org/abs/2607.12659) [F via alphaxiv] |
| vla.cpp | SmolVLA 65.41, BitVLA 101.11, Evo-1 131.01, GR00T-N1.5 28.78, pi0 27.90, GR00T-N1.6 26.70, GR00T-N1.7 26.84 ms per executed action | [2606.08094](https://arxiv.org/abs/2606.08094) [F] |
| AdaVLA | pi0.5 1.87x, X-VLA 2.24x on LIBERO; SmolVLA real-robot validation | [2608.29208](https://arxiv.org/abs/2608.29208) [A] |
| TRIAGE | prefill latency up to -48% | [2610.34519](https://arxiv.org/abs/2610.34519) [A] |
| FoldQuantVLA | W4A4 1.20-1.33x over floating-point TensorRT | [2609.24433](https://arxiv.org/abs/2609.24433) [A] |
| RhinoVLA | pi0.5 per-stage profile (69.3 / 528.0 / 257.0 ms) | [2606.07383](https://arxiv.org/abs/2606.07383) [F] |
| LiteVLA-Edge / LiteVLA-H | 150.5 ms (6.6 Hz) Q4_K_M GGUF; aerial dual-rate, sentence-level output at 149.90-164.57 ms | [2603.03380](https://arxiv.org/abs/2603.03380) [F], [2605.00884](https://arxiv.org/abs/2605.00884) [A] |
| Characterizing VLA Models | MolmoAct-7B on AGX Orin 64GB and Thor: generation ~75% of latency; Thor has ~5x BF16 compute but end-to-end improves only ~1.4x; latencies 200-300x above the 10 Hz requirement | [2603.02271](https://arxiv.org/abs/2603.02271) [F] |
| PICES | up to 2.3x wall-clock policy inference on Orin and Thor | [2610.19656](https://arxiv.org/abs/2610.19656) [A] |
| QAIL / SQIL | "edge GPU", 2.5x speedup and energy saving (device not named in abstracts) | [2412.01034](https://arxiv.org/abs/2412.01034), [2505.15304](https://arxiv.org/abs/2505.15304) [A] |

**Jetson AGX Thor**: VLA-HWBench ([2610.34717](https://arxiv.org/abs/2610.34717)), VLA-Triage ([2611.06710](https://arxiv.org/abs/2611.06710)), LoopQ ([2611.05093](https://arxiv.org/abs/2611.05093)), OxyGen ([2603.14371](https://arxiv.org/abs/2603.14371)), QCRL ([2610.15675](https://arxiv.org/abs/2610.15675)), rMuscle ([2609.19104](https://arxiv.org/abs/2609.19104)), Shallow-pi (pi0.5 130 -> 78 ms), EagleVLA (457.9 -> 132.8 ms), GR00T N1.7 TensorRT (~28 Hz, [README](https://github.com/NVIDIA/Isaac-GR00T)).

**Desktop / server only (or hardware not stated)**: SmolVLA paper itself (no latency numbers at all) - [2506.01844](https://arxiv.org/abs/2506.01844) [F]; Evo-1 (RTX 4090d) - [2511.04555](https://arxiv.org/abs/2511.04555) [F]; VLA-Adapter (H100) - [2509.09372](https://arxiv.org/abs/2509.09372) [F]; CogVLA (A800, inference hardware unspecified) - [2508.21046](https://arxiv.org/abs/2508.21046) [F]; realtime-vla (RTX 4090) - [2510.26742](https://arxiv.org/abs/2510.26742) [F]; TurboVLA (RTX 4090) - [2607.27205](https://arxiv.org/abs/2607.27205) [A]; OpenVLA-OFT (A100) and TinyVLA (A6000) per [2611.18372](https://arxiv.org/abs/2611.18372) [S]; and, with no device named in their abstracts, VLA-Cache, EfficientVLA, LightVLA, FlashVLA (2505.21200), SP-VLA, SpecPrune-VLA, VLA-Pruner, DeeR-VLA, MoLe-VLA, CEED-VLA, QVLA, HBVLA, SnapFlow.

- The edge-deployment survey notes OpenVLA-7B is "roughly 14 GB in FP16, near or above the practical budget of a Jetson Orin NX" - [2603.16952](https://arxiv.org/abs/2603.16952) [F].
- The life-cycle survey observes that across papers "speed is reported in incompatible forms (throughput, latency per chunk, frames per second, effective control rate)" - [2611.18372](https://arxiv.org/abs/2611.18372) [F].

### Inferences
- The three SmolVLA numbers on Orin-class boards are not mutually comparable: 230 ms (AGX Orin, Shallow-pi's baseline, presumably PyTorch), 65.41 ms per executed action (AGX Orin, vla.cpp, amortized), 141.81 ms per executed action (Orin Nano, vla.cpp), 107.55 ms p50 per call (Orin Nano Super, TensorRT + CUDA graph, community repo). They differ in runtime, metric definition (per call vs per executed action), camera count and power mode.
- A paper reporting per-stage latency, memory, power and closed-loop success for a SmolVLA-class model on a (non-Super or Super) Orin Nano 8 GB would fill a gap no verified peer-reviewed source currently covers; the nearest are vla.cpp (latency + BitVLA success on Nano) and the unreviewed community repo (latency + RAM, synthetic inputs).

### Gaps
- A Nov 2026 preprints.org manuscript (202611.0970) on SmolVLA speedups on a Jetson Xavier NX "conditioned on matching the reference implementation's outputs" surfaced in search but returned HTTP 403; its content is unverified.
- No verified VLA result on Jetson Orin NX; LiteVLA-Edge mentions Orin NX in one section but states AGX Orin elsewhere.
- QAIL / SQIL "edge GPU" identity was not confirmed from full text.
- NanoVLA's absolute FPS on the Orin Nano is only in figures.

---

## Q5. What limitations and open problems do recent (2025-2026) efficiency papers and surveys explicitly state?

### Takeaway
The stated open problems have shifted from "models are too big" (2025 surveys) to "our measurements and evaluations are unreliable" (2026): speedups depend on execution mode and hardware, do not compose, often leave the true bottleneck (iterative action expert, vision encoder, CPU preprocessing, memory bandwidth) untouched, and can silently break closed-loop behaviour that average benchmark success hides.

### Cited Findings

**Measurement validity and non-transferable speedups**
- Only one of 25 audited papers states the execution mode of both baseline and method; eager-mode measurements can overstate or understate savings after compilation; gains "vary in magnitude, sign, and rank across GPUs"; of 21 kernel-replacing comparisons 6 flipped from speed-up to slowdown across platforms - [2611.06710](https://arxiv.org/abs/2611.06710) [F].
- "Kernel fusion with 4-bit weights helps most ... but it exists for only these two models [pi0.5, GR00T N1.7] and only on the Jetson"; compilation can change actions (two compiled pi0.5 processes differ by up to 2.0 in one action dimension, a flipped gripper command); only 10 of 19 Jetson configurations keep up with LIBERO's control rate; no per-model power or memory measurements - [2610.34717](https://arxiv.org/abs/2610.34717) [F].
- Recommendation: "low-latency edge VLAs should use few action-generation iterations, inference software should extend kernel fusion and low-bit weights to more models and platforms, and edge hardware needs high memory bandwidth and a fast host CPU" - [2610.34717](https://arxiv.org/abs/2610.34717) [A].
- Action generation is memory-bandwidth-bound on edge SoCs: "up to 75% of end-to-end latency is consumed by the memory-bound action-generation phase", and more compute alone does not help (Thor ~5x compute, ~1.4x latency) - [2603.02271](https://arxiv.org/abs/2603.02271) [A]/[F]; EdgeDAE makes the same argument for diffusion action experts ("repeated parameter loading from DRAM across multiple steps") - [2610.00311](https://arxiv.org/abs/2610.00311) [A].

**Quantization**
- "VLA-specific PTQ methods underperform compared to simpler generic baselines. At 4 bits, RTN, AWQ and GPTQ match or exceed ActQuant and QuantVLA"; "Standard LIBERO hides the losses that do exist: configurations with no significant LIBERO loss lose 1.5-4.3 points under paired LIBERO-plus perturbations"; "Speed is a weaker story" (no speedup at batch 1 on H100; Thor slower on every 4-bit configuration) - [2611.05093](https://arxiv.org/abs/2611.05093) [A].
- Layer scope, numeric format and calibration interact: under uncalibrated W4A4 RTN, expanding a pi0.5 action-head subset from 126 to 167 layers raises success from 7.0% to 70.5%; results "establish recipe-dependent interactions ... rather than universal layer-sensitivity rules" - [2609.25376](https://arxiv.org/abs/2609.25376) [A].
- Simulation-selected mixed-precision allocations "show no detected success-rate loss on X-VLA but fail on pi0.5" on a real Franka, "highlighting the need for real-world evaluation before deployment" - [2610.38661](https://arxiv.org/abs/2610.38661) [A].
- The diffusion / flow action head "is highly sensitive" to low-bit PTQ; "prior work leaves it at full precision or falls back to mixed-precision schemes" - [2605.28803](https://arxiv.org/abs/2605.28803) [A]. Quantization errors accumulate in closed loop ("kinematic drift") - [2604.11572](https://arxiv.org/abs/2604.11572) [A]; rotation errors show "cumulative drift" - [2603.07904](https://arxiv.org/abs/2603.07904) [A].
- Whether PTQ preserves robustness / generalization (not just in-distribution success) "remains underexplored" - [2610.05745](https://arxiv.org/abs/2610.05745) [A].
- In vla.cpp, "fixed-input error alone does not predict these success rates" across GGUF quantization levels; the ternary fast path is CUDA-specific; the largest models exceed the Orin Nano's memory; robot studies use small samples in quasi-static scenes - [2606.08094](https://arxiv.org/abs/2606.08094) [F].
- Full W8A8 "noticeably degrades" success on an edge SoC, so W8A16 is deployed - [2606.07383](https://arxiv.org/abs/2606.07383) [F]. Uniform W4A4 drops GR00T N1.7 real-robot success to 80.0% vs 92.5% with selected W8A8 projections - [2609.24433](https://arxiv.org/abs/2609.24433) [A].

**Token pruning / caching / skipping**
- "directly applying VLM-oriented pruning methods to VLA inference can cause severe degradation", because attention differs between the prefill stage and the action-decode stage - [2511.16449](https://arxiv.org/abs/2511.16449) [A].
- Existing pruning "ignore[s] the global context, leading to >20% success rate drop and limited speedup in some scenarios" - [2509.05614](https://arxiv.org/abs/2509.05614) [A].
- "Token pruning methods often rely on manually predefined thresholds or fixed pruning ratios" - [2510.17111](https://arxiv.org/abs/2510.17111) [F]; CogVLA's own limitation: sparsity ratios and pruning schedules are predefined and may not adapt to instruction or scene difficulty - [2508.21046](https://arxiv.org/abs/2508.21046) [F].
- Self-referential gates collapse: at skip ratio 0.9 on LIBERO-Object, success falls to 0.68 (reuse) and 0.31 (deletion) vs dense 1.00 when the gate is harvested from the model's own accelerated forward, "and the collapse is invisible to the action-level detectors"; fix = one dense refresh pass run during chunk execution, off the critical path; resulting serve latency is only 18-22% below dense - [2608.00391](https://arxiv.org/abs/2608.00391) [A].
- Reuse vs prune is token-dependent: beyond a drift threshold, KV reuse yields up to 4.4x the action error of pruning - [2610.34519](https://arxiv.org/abs/2610.34519) [A].
- Pruning must preserve spatial structure, not only task-relevant tokens - [2609.36967](https://arxiv.org/abs/2609.36967) [A].
- Blind KV caching "hurts" on some suites; a confidence gate is needed - [2608.10824](https://arxiv.org/abs/2608.10824) [A].

**Structural pruning / compression**
- "pruning VLA models yields drastic performance degradation and increased safety violations" - [2510.08464](https://arxiv.org/abs/2510.08464) [A]; removing 63% of OpenVLA-OFT's backbone drops LIBERO-Long from 93.2% to 0.8% before recovery - [2609.19579](https://arxiv.org/abs/2609.19579) [A].
- "Compressed manipulation policies can pass offline evaluation while failing in closed-loop execution": an 8-layer Octo distillation passes every offline check (0.996 / 1.000 teacher ratios) and scores 0/72 vs the teacher's 40/72; "offline gates ... are insufficient acceptance tests for compressed policies" - [2609.23048](https://arxiv.org/abs/2609.23048) [A].
- Survey-level statement: "aggressive structural optimization often engenders a tension between architectural parsimony and representational fidelity"; reliance on static importance metrics limits adaptability; hierarchical / asynchronous frameworks yield unstable control signals - [2510.24795](https://arxiv.org/abs/2510.24795) [F].

**Evaluation of accelerated policies**
- Some training-free accelerators score *higher* than their baseline because of benchmark flaws: an audit of seven benchmarks finds 22 bugs and 4 design limitations; "bug fixes can reverse method rankings"; "Gains attributed to acceleration can therefore be artifacts of the benchmark" - [2609.37771](https://arxiv.org/abs/2609.37771) [A].
- Average success hides acceleration-induced failures; paired rollouts are needed; selectors without guarantees exceed the failure budget in up to 75% of trials - [2610.08917](https://arxiv.org/abs/2610.08917) [A].
- Some VLAs with near-perfect benchmark scores fail when objects or the camera move slightly; LIBERO-style benchmarks are saturating - [2611.18372](https://arxiv.org/abs/2611.18372) [A]/[F].
- Deployment-oriented papers frequently omit task success: LiteVLA-Edge reports timing only and states "latency alone does not characterize embodied performance" - [2603.03380](https://arxiv.org/abs/2603.03380) [F]; VLA-Triage measures latency only with random token selection - [2611.06710](https://arxiv.org/abs/2611.06710) [F].

**Action generation and asynchrony**
- Iterative flow sampling is named the cap on control frequency for pi0-style VLAs - [2604.05656](https://arxiv.org/abs/2604.05656) [A]; most acceleration "largely target[s] VLM inference costs, failing to address the iterative ODE solving process" - [2608.29208](https://arxiv.org/abs/2608.29208) [A].
- Flow-matching VLAs are structurally awkward for streaming: "global timestep injection invalidates KV-[cache reuse]" and BF16 can collapse numerically under continuous high-frequency inference - [2607.14695](https://arxiv.org/abs/2607.14695) [A].
- Asynchronous inference introduces "temporal misalignment ... between the prediction and execution intervals", and "existing asynchronous methods either degrade accuracy or introduce runtime overhead" - [2512.01031](https://arxiv.org/abs/2512.01031) [A]; SmolVLA's own async stack lowers average success (73.3% vs 78.3%) while speeding completion - [2506.01844](https://arxiv.org/abs/2506.01844) [F].
- Naive async on a Jetson Orin collapses on a real folding task (0/10 on the folding subtask) - [2607.12659](https://arxiv.org/abs/2607.12659) [F via alphaxiv]; the authors still expect a residual gap to high-end GPUs as models grow.
- Robot-side delays (perception acquisition, communication scheduling, physical response) are a separate part of the timing gap, motivating "joint optimization of model-inference efficiency and robot-system timing" - [2609.39822](https://arxiv.org/abs/2609.39822) [A].

**Small-model-specific limitations stated by their authors**
- SmolVLA: pretraining only on SO100 data (~23k trajectories vs ~1M for OpenVLA); VLM backbone pretrained mainly on document reading / OCR, suitability for robotics unclear; long-horizon tasks open - [2506.01844](https://arxiv.org/abs/2506.01844) [F].
- VLA-Adapter: no large-scale embodied pre-training and small scale limit real-world generalization - [2509.09372](https://arxiv.org/abs/2509.09372) [F].
- NanoVLA: encoder-only language model (BERT) variant is weakest on LIBERO-90; routing is threshold-sensitive - [2510.25122](https://arxiv.org/abs/2510.25122) [F].
- Shallow-pi: distillation needs teacher and student in memory; visual-token and diffusion-step reduction left to future work - [2601.20262](https://arxiv.org/abs/2601.20262) [F].
- Controlled study: returns from scaling the (aligned) action head "diminish sharply near the size today's pi-series VLAs already use", and "extra inference passes give no measurable benefit" - [2609.13984](https://arxiv.org/abs/2609.13984) [A].

**System-level (beyond the model)**
- "Deploying foundation models in embodied edge systems is fundamentally a systems problem, not just a problem of model compression"; quantization and pruning "do not resolve the dominant failures of embodied deployment"; reported figures include kernel-launch overhead of 30-60% of end-to-end latency on Jetson Orin during autoregressive decoding, ROS 2 serialization penalties up to 50% vs shared-memory designs, VLA throughput reduced 15-30% under background I/O on Orin-class platforms, scheduling jitter 5-20 ms on Jetson-class platforms under CFS - [2603.16952](https://arxiv.org/abs/2603.16952) [F] (survey-level claims; underlying sources not checked).
- "holistic system optimizations" and algorithm-system co-design are called for; memory scaling alone is insufficient - [2603.02271](https://arxiv.org/abs/2603.02271) [F].

### Inferences
- Problems that recur across independent 2026 sources and remain open: (1) no standard latency-reporting protocol (execution mode, per-call vs per-action, power mode, camera count); (2) closed-loop, perturbation-aware (LIBERO-Plus-style) and real-robot validation of every acceleration; (3) the action expert and the vision encoder are the least-compressed components even though they dominate once the LLM is optimized; (4) low-bit kernels that actually run faster on Jetson-class devices exist for very few models; (5) composition of optimizations is unpredictable; (6) memory bandwidth, CPU preprocessing and middleware become bottlenecks after model-level optimization.
- For an 8 GB device specifically, memory headroom (unified memory shared with sensors and OS) is an explicit binding constraint: 3B-class models OOM in vla.cpp on the Nano, and the community repo must split models into several TensorRT engines to build them at all.

### Gaps
- The dedicated challenge sections of [2603.16952](https://arxiv.org/abs/2603.16952) (Section 5) and [2611.18372](https://arxiv.org/abs/2611.18372) (Section 10) were truncated in the fetch; only earlier-section statements are captured.
- Most limitation statements for Oct-Nov 2026 preprints come from abstracts; fuller limitation sections were not read.

---

## Q6. Are there existing survey papers on efficient VLAs (2025-2026)? What are their taxonomies and stated open problems?

### Takeaway
Yes: two dedicated efficient-VLA surveys (both Oct 2025; one accepted to TPAMI), one edge-deployment survey for embodied foundation models (Mar 2026), and one general VLA life-cycle survey with a real-time-execution axis (Nov 2026). None of them organizes methods by which pipeline stage (vision encoder / prefix / action expert) is actually accelerated, none reports Jetson measurements of its own, and neither dedicated survey mentions Jetson at all.

### Cited Findings

**S1. "A Survey on Efficient Vision-Language-Action Models"** - Zhaoshu Yu, Bo Wang, Pengpeng Zeng, Haonan Zhang et al.; [arXiv 2510.24795](https://arxiv.org/abs/2510.24795); submitted 2025-10-27, v3 2026-09-28; "Accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)"; project page evla-survey.github.io [A]/[F].
- Three pillars: Efficient Model Design, Efficient Training, Efficient Data Collection [A].
- Efficient Model Design -> Efficient Architectures: efficient attention (SARA-RT, Long-VLA, dVLA, RetoVLA, KV-Efficient VLA); transformer alternatives (RoboMamba, FlowRAM); efficient action decoding = parallel decoding (OpenVLA-OFT, EdgeVLA, PD-VLA, CEED-VLA, Spec-VLA) + generative decoding (TinyVLA, HybridVLA, FreqPolicy, FlowRAM, MinD, VQ-VLA, AMS, NinA, Discrete Diffusion VLA); lightweight components (RoboMamba, TinyVLA, EdgeVLA, MiniVLA, CLIP-RT, DiVLA-2B, SVLR, NORA, SmolVLA, SP-VLA); mixture-of-experts (GeRM, FedVLA, TAVP); hierarchical systems (HiRT, DP-VLA, SmolVLA, RoboDual, HAMSTER, FiS, Fast ECoT, MinD) [F].
- Efficient Model Design -> Model Compression: layer pruning (training-free: DeeR-VLA, SmolVLA, RLRC, FLOWER; training-based: MoLe-VLA, LightDP); quantization (OpenVLA, QAIL, FAST, SQIL, BitVLA, RLRC, SQAP-VLA); token optimization = compression / pruning / caching (lists in Q1) [F].
- Efficient Training: data-efficient pre-training, efficient action representation, multi-stage / RL / LoRA; post-training by supervised fine-tuning or RL. Efficient Data Collection: human-in-the-loop, simulation, internet-scale and cross-domain data, self-exploration, augmentation [F].
- Its Table I (efficiency metrics of representative VLAs): RT-2-PaLI-X 55B, 330-1000 ms, 1-3 Hz; RT-2 5B, 200 ms, 5 Hz; OpenVLA 7B, 166 ms, 6 Hz; pi0 3.3B, 73 ms, 20/50 Hz; HiRobot 3B, 73 ms; GR00T N1 2.2B, 63.9 ms. No hardware column [F].
- Stated challenges: "Efficient VLA design confronts a fundamental tension between architectural parsimony and representational richness"; speed gains "inevitably trade[s] model capability for speed"; training: RL post-training "grapples with high-variance gradients and reward sparsity", action compression "distorts continuous kinematics"; data: "Data remains the bedrock and bottleneck", sim-to-real gaps, no standardized cross-domain repositories [F].
- Stated future work: "Future VLA designs must evolve toward intrinsic adaptability" (context-aware token pruning, modality-agnostic backbones); "Hardware-software co-design" that "may shatter current latency barriers"; decentralized / continual training; generative, self-sustaining data ecosystems [F].
- Model-design limitations (Sec. III-C): semantic drift in long-horizon tasks under aggressive structural optimization; asynchronous execution in hierarchical frameworks "yields unstable control signals"; static importance metrics limit adaptability [F].
- Jetson: not mentioned. Vision-encoder cost: not analyzed [F].

**S2. "Efficient Vision-Language-Action Models for Embodied Manipulation: A Systematic Survey"** - Weifan Guan, Qinghao Hu, Aosheng Li, Jian Cheng; [arXiv 2510.17111](https://arxiv.org/abs/2510.17111); 2025-10-20 (v3 2025-10-23) [A].
- Four dimensions: (1) model architecture = static backbone selection (RoboMamba, TinyVLA, SmolVLA, NORA), dynamic computation pathways (SmolVLA, FLOWER, DeeR-VLA, MoLe-VLA, EfficientVLA), dual-system design (LCB, HiRT, RoboDual, OpenHelix, FiS, Hume, HyperVLA, SP-VLA; RT-Cache as retrieval fast path); (2) perception feature = selective feature processing and temporal sharing / reuse; (3) action generation = raw action generation (action chunking, RTC, FAST, OmniSAT, VOTE, SpatialVLA) and reasoning-aware generation (ECoT, Fast ECoT, ECoT-Lite; UniPi, SuSIE, VPP, CoT-VLA, DreamVLA); (4) training and inference = PEFT, distillation (CEED-VLA, MoLe-VLA, VITA-VLA), pruning recovery (GLUESTICK), quantization (SQIL, BitVLA); inference techniques (OpenVLA-OFT, Spec-VLA, PD-VLA, CEED-VLA) [F].
- No table with params / latency / hardware / success; scattered numbers only (OpenVLA 7B ~5 Hz; pi0 3B ~10 Hz; RT-2 55B ~3 Hz; BitVLA memory ~15.1 GB -> ~1.4 GB) [F].
- Stated open problems / directions: VLA scaling laws, adaptive layer-skipping, cloud-edge partitioning (Sec. 3.4); "Token pruning methods often rely on manually predefined thresholds or fixed pruning ratios" (Sec. 4.3); "Reasoning-based VLAs are likely to become the mainstream paradigm", needing selective reasoning (Sec. 5.3); most efficiency methods are inherited from VLM research - "Future work should design efficiency techniques that directly target embodied decision-making" (Sec. 6.3); shift "from merely compressing models to optimizing the allocation of resources" (Sec. 7.1); hybrid reactive + selectively invoked reasoning module with arbitration (Sec. 7.3); a three-part evaluation framework (efficiency, performance / robustness, interpretability) and a large open benchmark (Sec. 7.5) [F].
- Jetson: not mentioned [F].

**S3. "Embodied Foundation Models at the Edge: A Survey of Deployment Constraints and Mitigation Strategies"** - Utkarsh Grover, Ravi Ranjan, Mingyang Mao et al.; [arXiv 2603.16952](https://arxiv.org/abs/2603.16952); 2026-03-16 [A].
- Organizing device: the "Deployment Gauntlet" of eight coupled barriers - sensor fusion tax; heterogeneous compute mismatch; unified memory bottleneck; energy and thermal ceiling; long-horizon drift; safety and verification gap; real-time scheduling interference; communication constraints [F].
- Mitigation families: transport-aware sensing and synchronization relaxation (incl. asynchronous action generation); hardware-aligned model and compiler co-design; memory-traffic reduction and schedulable memory use; energy- and thermal-aware execution; online robustness / runtime assurance; real-time systems support and timing-aware communication [F].
- Core claims: "autoregressive Vision-Language-Action policies are constrained primarily by memory bandwidth, whereas diffusion-based controllers are limited more by compute latency and sustained execution cost"; "Reliable deployment therefore depends on system-level co-design across memory, scheduling, communication, and model architecture, including decompositions that separate fast control from slower semantic reasoning" [A].

**S4. "Vision-Language-Action Models: A Life-Cycle Survey of Design, Training and Deployment"** - Suvarna Kadam, Soran Parsa; [arXiv 2611.18372](https://arxiv.org/abs/2611.18372); 2026-11-13; 88 works reviewed, Dec 2022 - Oct 2026 [A].
- Seven axes; axis (g) "real-time execution" covers asynchronous inference (SmolVLA), RTC, remote VLA + edge adapter (AsyncVLA), multi-rate hierarchies, dual-system rate split (GR00T N1), KV-cache reuse (StreamVLN), parallel decoding (OpenVLA-OFT), quantized serving (OpenVLA), action chunking [F].
- Its Table 2 lists SmolVLA as 0.45B, speed "n.r.", hardware "consumer GPUs and CPUs"; pi0 3.3B, 73 ms per chunk on board (RTX 4090), up to 50 Hz control; GR00T N1 2.2B, 63.9 ms per 16-action chunk (L40); OpenVLA-OFT 7B, 71.4 Hz, 0.11 s (A100); VLA-Adapter 0.5B, 219.2 Hz, 0.04 s (H100); TinyVLA 1B, 14 ms vs OpenVLA 292 ms (A6000) [F].
- Trend noted: "model sizes have split: most recent models have 3-7 billion parameters, while a new group under one billion runs on consumer GPUs" [A].
- Stated issue: speeds are reported in incompatible units and cannot be ranked; the fetched part does not treat token pruning or distillation as deployment techniques [F].

**Related (not a VLA survey)**: an efficient-VLM-for-edge survey, arXiv 2504.09724, surfaced in search; not opened [U].

### Inferences
- A prior-art section can legitimately claim that existing surveys (a) classify by technique type rather than by accelerated pipeline stage, (b) contain no first-hand edge measurements, and (c) pre-date or omit the 2026 measurement literature (vla.cpp, VLA-HWBench, VLA-Triage, LoopQ). The claim should be scoped to the four surveys above.
- S1 and S2 disagree on where SmolVLA belongs (S1: lightweight component + hierarchical system + training-free layer pruning + token compression; S2: static backbone selection + dynamic computation pathway), which illustrates that SmolVLA-class models combine several families rather than exemplifying one.

### Gaps
- Sections V-VII of S1 and the last ~20k characters of S2 were only partially captured; S3 Section 5 and S4 Section 10 (open challenges) were not captured.
- No efficient-VLA survey dated later than Oct 2025 with a dedicated efficiency taxonomy was found via arXiv title search ("survey" + VLA + efficiency terms); surveys whose titles lack those words could have been missed.
- Awesome-list repositories accompanying S1 / S2 were not inspected.
