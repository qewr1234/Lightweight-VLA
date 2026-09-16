#!/usr/bin/env python
"""Build a weights-free SmolVLM2 cache so the tests run without network access.

`tests/test_lightweight.py` and `tests/test_batched_vision.py` construct the policy with
`load_vlm_weights=False`, so no checkpoint is ever read - but `SmolVLMWithExpertModel`
still calls `AutoConfig.from_pretrained` and `AutoProcessor.from_pretrained` on
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, which reaches huggingface.co. Behind a
firewall (or on a Jetson with no route) that is the only thing standing between you and
the equivalence checks.

This script writes a hub-layout cache holding exactly what those two calls need:

    config.json                 verbatim from the model card (the real architecture)
    preprocessor_config.json    verbatim
    processor_config.json       verbatim
    tokenizer_config.json       the real special-token wiring, trimmed
    tokenizer.json              SYNTHESIZED - see below

The tokenizer is a stand-in. The real `tokenizer.json` is 3.5 MB of learned BPE merges;
this builds a byte-level tokenizer with the same vocabulary size (49280) and the same
128 added tokens at their real ids (`<image>` 49190, `<fake_token_around_image>` 49189,
`<global-img>` 49152, ...), which is all the policy reads from it:
`modeling_smolvla.py` only ever asks for `fake_image_token_id` / `global_image_token_id`.
Text tokenized with it round-trips through bytes instead of real merges, so token ids for
a given string will NOT match the real model. That is irrelevant for the equivalence
tests (they pass random token ids straight in) and WRONG for anything that reads a task
string - never point a real policy at this cache.

Usage:
    python tests/offline_vlm_cache.py                      # -> ./.hf_offline
    HF_HOME=$PWD/.hf_offline HF_HUB_OFFLINE=1 python tests/test_lightweight.py
"""

import argparse
import json
from pathlib import Path

MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
REVISION = "0" * 40  # no real commit is being pinned; offline resolution only needs a ref

VOCAB_SIZE = 49280
BASE_VOCAB = 49152  # ids below this are ordinary text tokens, above are the added tokens

# Verbatim from the model card.
CONFIG = {
    "architectures": ["SmolVLMForConditionalGeneration"],
    "image_token_id": 49190,
    "model_type": "smolvlm",
    "pad_token_id": 128002,
    "scale_factor": 4,
    "text_config": {
        "architectures": ["VLlama3ForCausalLM"],
        "head_dim": 64,
        "hidden_size": 960,
        "intermediate_size": 2560,
        "is_llama_config": True,
        "max_position_embeddings": 8192,
        "model_type": "llama",
        "neftune_noise_alpha": 0.0,
        "num_attention_heads": 15,
        "num_hidden_layers": 32,
        "num_key_value_heads": 5,
        "pad_token_id": 2,
        "pixel_shuffle_factor": 4,
        "qk_layer_norms": False,
        "rms_norm_eps": 1e-05,
        "rope_interleaved": False,
        "rope_theta": 100000,
        "torch_dtype": "bfloat16",
        "use_resampler": False,
        "vocab_size": VOCAB_SIZE,
    },
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
    "transformers_version": "4.47.1",
    "use_cache": False,
    "use_reentrant_checkpointing": False,
    "vision_config": {
        "hidden_size": 768,
        "image_size": 512,
        "max_image_size": {"longest_edge": 512},
        "model_type": "smolvlm_vision",
        "num_attention_heads": 12,
        "patch_size": 16,
        "size": {"longest_edge": 2048},
        "tie_word_embeddings": False,
        "use_base_siglip": False,
    },
    "vocab_size": VOCAB_SIZE,
}

PREPROCESSOR_CONFIG = {
    "do_convert_rgb": True,
    "do_image_splitting": True,
    "do_normalize": True,
    "do_pad": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.5, 0.5, 0.5],
    "image_processor_type": "SmolVLMImageProcessor",
    "image_std": [0.5, 0.5, 0.5],
    "max_image_size": {"longest_edge": 512},
    "processor_class": "SmolVLMProcessor",
    "resample": 1,
    "rescale_factor": 0.00392156862745098,
    "size": {"longest_edge": 2048},
    "video_sampling": {"fps": 1, "max_frames": 64, "video_size": {"longest_edge": 512}},
}

PROCESSOR_CONFIG = {"image_seq_len": 64, "processor_class": "SmolVLMProcessor"}

# The three ids the tokenizer_config names and the base vocab must therefore contain.
RESERVED_BASE = {"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2}

# Added tokens, at the ids the real tokenizer assigns them (added_tokens.json).
ADDED_TOKENS = {
    "<global-img>": BASE_VOCAB,
    **{
        f"<row_{r}_col_{c}>": BASE_VOCAB + 1 + (r - 1) * 6 + (c - 1)
        for r in range(1, 7)
        for c in range(1, 7)
    },
    "<fake_token_around_image>": 49189,
    "<image>": 49190,
    **{f"<|reserved_special_token_{i}|>": 49191 + i for i in range(88)},
    "<end_of_utterance>": 49279,
}


def build_tokenizer(dest: Path) -> None:
    """Byte-level stand-in with the real vocab size and the real added-token ids."""
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = dict(RESERVED_BASE)
    for ch in alphabet:
        vocab[ch] = len(vocab)
    while len(vocab) < BASE_VOCAB:  # filler so the added tokens land on their real ids
        vocab[f"<unused_{len(vocab)}>"] = len(vocab)

    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    for token, token_id in sorted(ADDED_TOKENS.items(), key=lambda kv: kv[1]):
        assigned = tokenizer.add_special_tokens(
            [AddedToken(token, special=True, normalized=False)]
        )
        if assigned:  # newly added -> must have landed where the real tokenizer put it
            got = tokenizer.token_to_id(token)
            if got != token_id:
                raise AssertionError(f"{token}: id {got}, expected {token_id}")

    if tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise AssertionError(f"vocab {tokenizer.get_vocab_size()}, expected {VOCAB_SIZE}")
    tokenizer.save(str(dest / "tokenizer.json"))

    (dest / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "additional_special_tokens": [
                    "<fake_token_around_image>",
                    "<image>",
                    "<end_of_utterance>",
                ],
                "bos_token": "<|im_start|>",
                "clean_up_tokenization_spaces": False,
                "end_of_utterance_token": "<end_of_utterance>",
                "eos_token": "<end_of_utterance>",
                "extra_special_tokens": {
                    "end_of_utterance_token": "<end_of_utterance>",
                    "fake_image_token": "<fake_token_around_image>",
                    "global_image_token": "<global-img>",
                    "image_token": "<image>",
                },
                "fake_image_token": "<fake_token_around_image>",
                "global_image_token": "<global-img>",
                "image_token": "<image>",
                "model_max_length": 8192,
                "pad_token": "<|im_end|>",
                "processor_class": "SmolVLMProcessor",
                "tokenizer_class": "PreTrainedTokenizerFast",
                "truncation_side": "left",
                "unk_token": "<|endoftext|>",
            },
            indent=2,
        )
    )


def build_cache(hf_home: Path) -> Path:
    """Write the hub-layout cache under `hf_home` and return the snapshot directory."""
    repo = hf_home / "hub" / ("models--" + MODEL_ID.replace("/", "--"))
    snapshot = repo / "snapshots" / REVISION
    snapshot.mkdir(parents=True, exist_ok=True)
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    (repo / "refs" / "main").write_text(REVISION)

    (snapshot / "config.json").write_text(json.dumps(CONFIG, indent=2))
    (snapshot / "preprocessor_config.json").write_text(json.dumps(PREPROCESSOR_CONFIG, indent=2))
    (snapshot / "processor_config.json").write_text(json.dumps(PROCESSOR_CONFIG, indent=2))
    build_tokenizer(snapshot)
    return snapshot


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--hf-home",
        type=Path,
        default=Path(__file__).resolve().parent.parent / ".hf_offline",
        help="directory to use as HF_HOME (default: ./.hf_offline)",
    )
    args = p.parse_args()

    snapshot = build_cache(args.hf_home)
    print(f"wrote {snapshot}")
    for f in sorted(snapshot.iterdir()):
        print(f"  {f.name:28s} {f.stat().st_size:>9,d} B")
    print(
        f"\nrun the tests with:\n"
        f"  HF_HOME={args.hf_home} HF_HUB_OFFLINE=1 python tests/test_lightweight.py"
    )


if __name__ == "__main__":
    main()
