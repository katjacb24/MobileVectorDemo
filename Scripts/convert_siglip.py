#!/usr/bin/env python3
"""
convert_siglip.py

Converts google/siglip-base-patch16-224 (Apache 2.0) from PyTorch to Core ML.
Downloads model weights from HuggingFace on first run (~350 MB), then exports:

  SigLIPVision.mlpackage  — takes a 224x224 RGB image, outputs a 768-dim embedding
  SigLIPText.mlpackage    — takes tokenized text (64 tokens), outputs a 768-dim embedding

Both embeddings are L2-normalised inside the model, so cosine similarity in Swift
reduces to a plain dot product.

Usage:
    python3 convert_siglip.py
    python3 convert_siglip.py --output ~/Models/SigLIP

Requirements:
    pip install torch transformers coremltools pillow

Tested with:
    torch 2.x, transformers 4.40+, coremltools 7.x
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct
from transformers import SiglipVisionModel, SiglipTextModel, AutoTokenizer

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_ID       = "google/siglip-base-patch16-224"  # Apache 2.0
IMAGE_SIZE     = 224   # pixels (height and width)
TEXT_MAX_LEN   = 64    # maximum token sequence length SigLIP was trained with
EMBED_DIM      = 768   # output embedding dimensionality


# ── Trace-friendly attention-pooling head ─────────────────────────────────────
# SigLIP's vision head uses torch.nn.MultiheadAttention. When traced, that module
# emits an `aten::Int` on a multi-element tensor that coremltools cannot convert
# (TypeError: only 0-dimensional arrays can be converted to Python scalars).
# We replace the head's forward with a mathematically identical manual attention
# built from the SAME trained weights (in_proj/out_proj), using only linear,
# matmul and softmax — all of which coremltools handles cleanly. Output is
# numerically equivalent to the original head.

def _patch_attention_pooling_head(vision_model: SiglipVisionModel) -> None:
    head = vision_model.vision_model.head
    mha = head.attention  # torch.nn.MultiheadAttention

    def manual_forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)  # [B, 1, D]

        embed_dim = mha.embed_dim
        num_heads = mha.num_heads
        head_dim = embed_dim // num_heads
        scale = head_dim ** -0.5

        q_w, k_w, v_w = mha.in_proj_weight.chunk(3, dim=0)
        q_b, k_b, v_b = mha.in_proj_bias.chunk(3, dim=0)

        q = F.linear(probe, q_w, q_b)          # [B, 1, D]
        k = F.linear(hidden_state, k_w, k_b)   # [B, L, D]
        v = F.linear(hidden_state, v_w, v_b)   # [B, L, D]

        lq = q.shape[1]
        lk = k.shape[1]
        q = q.view(batch_size, lq, num_heads, head_dim).transpose(1, 2)  # [B,H,1,hd]
        k = k.view(batch_size, lk, num_heads, head_dim).transpose(1, 2)  # [B,H,L,hd]
        v = v.view(batch_size, lk, num_heads, head_dim).transpose(1, 2)  # [B,H,L,hd]

        attn = torch.matmul(q, k.transpose(-2, -1)) * scale             # [B,H,1,L]
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)                                     # [B,H,1,hd]
        out = out.transpose(1, 2).reshape(batch_size, lq, embed_dim)    # [B,1,D]
        out = mha.out_proj(out)                                         # [B,1,D]

        residual = out
        out = self.layernorm(out)
        out = residual + self.mlp(out)
        return out[:, 0]

    import types
    head.forward = types.MethodType(manual_forward, head)


# ── Model wrappers ────────────────────────────────────────────────────────────
# torch.jit.trace requires the model to return a plain tensor, not a dataclass.
# We also bake in L2 normalisation so Swift only needs a dot product for cosine
# similarity — no post-processing required on the Swift side.

class VisionEncoderWrapper(torch.nn.Module):
    """Wraps SiglipVisionModel: pixel_values → L2-normalised 768-dim embedding."""

    def __init__(self, model: SiglipVisionModel):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [1, 3, 224, 224] float32, values in [-1, 1]
        outputs = self.model(pixel_values=pixel_values)
        embedding = outputs.pooler_output          # [1, 768]
        return F.normalize(embedding, p=2, dim=-1) # L2-normalise


class TextEncoderWrapper(torch.nn.Module):
    """Wraps SiglipTextModel: (input_ids, attention_mask) → L2-normalised 768-dim embedding."""

    def __init__(self, model: SiglipTextModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # input_ids / attention_mask: [1, 64] int32 from Core ML
        # transformers expects int64 (long), so we cast here
        outputs = self.model(
            input_ids=input_ids.long(),
            attention_mask=attention_mask.long(),
        )
        embedding = outputs.pooler_output          # [1, 768]
        return F.normalize(embedding, p=2, dim=-1) # L2-normalise


# ── Conversion helpers ────────────────────────────────────────────────────────

def convert_vision_encoder(output_dir: str) -> str:
    print("\n── Vision Encoder ──────────────────────────────────────────────")
    print(f"Loading {MODEL_ID} (vision)…")

    vision_model = SiglipVisionModel.from_pretrained(MODEL_ID)
    vision_model.eval()
    _patch_attention_pooling_head(vision_model)  # avoid nn.MultiheadAttention trace issue

    wrapper = VisionEncoderWrapper(vision_model)
    wrapper.eval()

    # Example input used by torch.jit.trace to record the compute graph.
    # Shape must match what the model will receive at runtime.
    example_pixels = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)

    print("Tracing…")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_pixels)

    print("Converting to Core ML (fp16, iOS 17+)…")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="pixel_values",
                shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
                dtype=np.float32,
            )
        ],
        outputs=[
            ct.TensorType(name="embedding", dtype=np.float32)
        ],
        # fp16 weights halve file size and run efficiently on Neural Engine
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )

    mlmodel.short_description = (
        "SigLIP ViT-B/16 vision encoder. "
        "Input: pixel_values [1,3,224,224] normalised to [-1,1]. "
        "Output: L2-normalised embedding [1,768]."
    )

    out_path = os.path.join(output_dir, "SigLIPVision.mlpackage")
    mlmodel.save(out_path)
    print(f"Saved → {out_path}")

    # Print tensor shapes so you can confirm correctness at a glance
    spec = mlmodel.get_spec()
    inp  = spec.description.input[0]
    outp = spec.description.output[0]
    print(f"  input  '{inp.name}':  {list(inp.type.multiArrayType.shape)}")
    print(f"  output '{outp.name}': {list(outp.type.multiArrayType.shape)}")

    return out_path


def convert_text_encoder(output_dir: str) -> str:
    print("\n── Text Encoder ────────────────────────────────────────────────")
    print(f"Loading {MODEL_ID} (text)…")

    text_model = SiglipTextModel.from_pretrained(MODEL_ID)
    text_model.eval()

    wrapper = TextEncoderWrapper(text_model)
    wrapper.eval()

    # Use int32 for tracing — Core ML does not support int64 inputs.
    # The wrapper casts to int64 internally before passing to transformers.
    example_ids  = torch.zeros(1, TEXT_MAX_LEN, dtype=torch.int32)
    example_mask = torch.ones(1,  TEXT_MAX_LEN, dtype=torch.int32)

    print("Tracing…")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (example_ids, example_mask))

    print("Converting to Core ML (fp16, iOS 17+)…")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids",      shape=(1, TEXT_MAX_LEN), dtype=np.int32),
            ct.TensorType(name="attention_mask",  shape=(1, TEXT_MAX_LEN), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="embedding", dtype=np.float32)
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )

    mlmodel.short_description = (
        "SigLIP ViT-B/16 text encoder. "
        f"Inputs: input_ids [1,{TEXT_MAX_LEN}] int32, attention_mask [1,{TEXT_MAX_LEN}] int32. "
        "Output: L2-normalised embedding [1,768]."
    )

    out_path = os.path.join(output_dir, "SigLIPText.mlpackage")
    mlmodel.save(out_path)
    print(f"Saved → {out_path}")

    spec = mlmodel.get_spec()
    for inp in spec.description.input:
        print(f"  input  '{inp.name}': {list(inp.type.multiArrayType.shape)}")
    outp = spec.description.output[0]
    print(f"  output '{outp.name}': {list(outp.type.multiArrayType.shape)}")

    return out_path


def save_tokenizer(output_dir: str) -> str:
    """
    Saves the SigLIP tokenizer files to <output_dir>/tokenizer/, then generates
    vocab.json — the format the pure-Swift SigLIPTokenizer reads.

    save_pretrained() only writes the SentencePiece model (spiece.model) plus
    config/special-tokens JSON; it does NOT emit the per-piece scores the Swift
    Unigram/Viterbi tokenizer needs. generate_vocab_json() reads spiece.model
    directly and writes {pieces, scores, unk_id, eos_id, pad_id}.
    """
    print("\n── Tokenizer ───────────────────────────────────────────────────")
    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)

    print(f"Loading tokenizer from {MODEL_ID}…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(tokenizer_dir)

    generate_vocab_json(tokenizer_dir)

    files = os.listdir(tokenizer_dir)
    print(f"Saved {len(files)} tokenizer file(s) → {tokenizer_dir}")
    for f in sorted(files):
        print(f"  {f}")

    return tokenizer_dir


def generate_vocab_json(tokenizer_dir: str) -> str:
    """
    Reads <tokenizer_dir>/spiece.model and writes vocab.json with the vocabulary
    pieces and their SentencePiece log-prob scores (indexed by token id), plus the
    unk/eos/pad ids. This is exactly what the Swift SigLIPTokenizer's VocabFile
    decodes: {"pieces": [...], "scores": [...], "unk_id": int, "eos_id": int, "pad_id": int}.
    """
    import json
    import sentencepiece as spm

    spiece_path = os.path.join(tokenizer_dir, "spiece.model")
    sp = spm.SentencePieceProcessor()
    sp.Load(spiece_path)

    n = sp.GetPieceSize()
    pieces = [sp.IdToPiece(i) for i in range(n)]
    scores = [sp.GetScore(i) for i in range(n)]

    vocab = {
        "pieces": pieces,
        "scores": scores,
        "unk_id": sp.unk_id(),
        "eos_id": sp.eos_id(),
        "pad_id": sp.pad_id(),
    }

    vocab_path = os.path.join(tokenizer_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    print(
        f"Generated vocab.json: {n} pieces  "
        f"(unk={vocab['unk_id']}, eos={vocab['eos_id']}, pad={vocab['pad_id']})"
    )
    return vocab_path


# ── Verification ──────────────────────────────────────────────────────────────

def verify(vision_path: str, text_path: str) -> None:
    """
    Loads both converted models and runs a quick forward pass to confirm
    output shapes and that embeddings are already L2-normalised (norm ≈ 1.0).
    """
    print("\n── Verification ────────────────────────────────────────────────")

    vm = ct.models.MLModel(vision_path)
    tm = ct.models.MLModel(text_path)

    # Vision: uniform grey image (values in [-1, 1])
    pixels = np.full((1, 3, IMAGE_SIZE, IMAGE_SIZE), 0.5, dtype=np.float32)
    v_out  = vm.predict({"pixel_values": pixels})
    v_emb  = v_out["embedding"]
    v_norm = float(np.linalg.norm(v_emb))
    print(f"Vision  output shape: {v_emb.shape}  |  L2 norm: {v_norm:.4f} (should be ≈1.0)")
    assert v_emb.shape == (1, EMBED_DIM), f"Unexpected shape: {v_emb.shape}"

    # Text: dummy token ids (all zeros, attention mask all ones)
    ids   = np.zeros((1, TEXT_MAX_LEN), dtype=np.int32)
    mask  = np.ones((1,  TEXT_MAX_LEN), dtype=np.int32)
    t_out = tm.predict({"input_ids": ids, "attention_mask": mask})
    t_emb = t_out["embedding"]
    t_norm = float(np.linalg.norm(t_emb))
    print(f"Text    output shape: {t_emb.shape}  |  L2 norm: {t_norm:.4f} (should be ≈1.0)")
    assert t_emb.shape == (1, EMBED_DIM), f"Unexpected shape: {t_emb.shape}"

    print("Verification passed ✓")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SigLIP ViT-B/16 (Apache 2.0) to Core ML (.mlpackage)"
    )
    parser.add_argument(
        "--output",
        default=os.path.expanduser("~/Models/SigLIP"),
        help="Directory to save outputs (default: ~/Models/SigLIP)",
    )
    args = parser.parse_args()
    output_dir = os.path.expanduser(args.output)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Model source:     {MODEL_ID}  (Apache 2.0)")

    vision_path    = convert_vision_encoder(output_dir)
    text_path      = convert_text_encoder(output_dir)
    tokenizer_dir  = save_tokenizer(output_dir)
    verify(vision_path, text_path)

    print("\n══ Conversion complete ════════════════════════════════════════")
    print(f"  SigLIPVision.mlpackage → {vision_path}")
    print(f"  SigLIPText.mlpackage   → {text_path}")
    print(f"  tokenizer/             → {tokenizer_dir}")
    print()
    print("Next step:")
    print("  Open MobileDemo.xcodeproj and update the model paths in")
    print("  SigLIPEmbedder.swift to point to the files above.")


if __name__ == "__main__":
    main()
