# Ginobili port — design notes

Status: groundwork only. The architecture is being proven in PyTorch
(Sukuna-Ginobili, Gates 0-2 on a B200) before any C++ lands. This document
fixes the port's shape so the work, when it starts, starts aligned.

## What Ginobili is

A Llama-3.1-8B-Instruct backbone (frozen) with a parallel expert branch at
every FFN stage: 12 micro experts (SwiGLU, intermediate 512) and a learned
router selecting top-k per token (default k=6, configurable). The stage
output is:

    y = x + F_frozen(z) + sum_{i in TopK(R(z))} p_i * E_i(z)

## GGUF layout (proposal)

Per layer `L`, alongside the stock FFN tensors:

- `blk.{L}.ffn_router.weight` — router projection (hidden x 12), F32/F16.
- `blk.{L}.ffn_exp.{i}.gate.weight` / `.up.weight` / `.down.weight`
  for i in 0..11 — quantize like ordinary FFN tensors.

Metadata keys:

- `ginobili.expert_count` (uint32, 12)
- `ginobili.expert_intermediate` (uint32, 512)
- `ginobili.top_k_default` (uint32, 6)

## Graph wiring

Ginobili registers as its own architecture whose graph reuses the Llama
attention and norm blocks unchanged. llama.cpp already carries MoE
plumbing for other architectures. The novel part is the hybrid stage:
dense FFN output and the routed expert sum are computed from the same
normalized input and added. No changes to attention, norms, embeddings,
or the LM head - and no changes to the Llama architecture's own graph.

## Runtime knobs

- `--ginobili-top-k N` — fixed k for the run.
- `--ginobili-full-court` — k = expert_count for the whole run.
- Per-token adaptive widening (router-confidence based) is NOT natural in
  ggml's graph model. It stays a research item and is not promised here.

## Converter

A model class in the HF-to-GGUF converter that reads a Ginobili checkpoint
(stock Llama weights referenced, expert banks in a companion file) and
emits the layout above. Flows through Hearth, which vendors this tree.

## Sequencing

1. PyTorch proof: Gates 0-2 against real 8B weights (B200).
2. Converter + graph on this branch; verify logits parity vs PyTorch.
3. Quantization pass (experts Q4_K etc., router high precision).
4. V100 serving validation (quantized weights + compressed KV cache;
   full-precision 256K context does not fit the card and is out of scope).

Upstream ambition: if the model earns downloads, this layout is the
candidate to propose to ggml-org. Demand first, merge second.

## Merge safety (the upstream discipline)

Everything here is built so it could merge into ggml-org/llama.cpp one
day without a fight:

- Ginobili is its OWN architecture registration, reusing Llama components,
  not a patch to the Llama graph. Stock models take byte-identical code
  paths; nothing about existing behavior changes.
- Purely additive: new arch, new tensors, new converter class, new flags.
  No edits to shared paths, no fork-only hacks, no behavior flags that
  alter other models.
- Upstream conventions throughout: tensor naming, gguf-py metadata,
  code style, and a slice-model test for conversion and inference.
- The branch stays rebased on upstream master so the eventual diff is
  small, current, and reviewable.
- Demand is the upstream argument: the model's own downloads make the
  case. The code just has to be ready to be judged.
