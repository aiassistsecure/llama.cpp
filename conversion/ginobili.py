from __future__ import annotations

import re
from typing import Iterable

import torch
from torch import Tensor

from .base import ModelBase, gguf, logger
from .llama import LlamaModel

# Ginobili: a frozen Llama base with a grafted micro-expert bank in
# parallel with each layer's dense FFN. The checkpoint is a standard
# Llama checkpoint plus, per layer:
#   model.layers.{bid}.mlp.bank.router.weight            [n_expert, hidden]
#   model.layers.{bid}.mlp.bank.experts.{e}.gate_proj.weight
#   model.layers.{bid}.mlp.bank.experts.{e}.up_proj.weight
#   model.layers.{bid}.mlp.bank.experts.{e}.down_proj.weight
# Experts are stacked into single 3d tensors (ffn_gate_exps / ffn_up_exps
# / ffn_down_exps), following the merged-expert convention of the other
# MoE converters. The router maps to ffn_gate_inp.


@ModelBase.register("GinobiliForCausalLM")
class GinobiliModel(LlamaModel):
    model_arch = gguf.MODEL_ARCH.GINOBILI

    _bank: list[dict[str, Tensor]] | None = None

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_expert_count(self.hparams["ginobili_expert_count"])
        self.gguf_writer.add_expert_used_count(self.hparams["ginobili_top_k"])
        self.gguf_writer.add_expert_feed_forward_length(self.hparams["ginobili_expert_intermediate"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        m = re.match(r"model\.layers\.(\d+)\.mlp\.bank\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", name)
        if m is not None:
            assert bid is not None
            n_experts = self.hparams["ginobili_expert_count"]
            if self._bank is None:
                self._bank = [{} for _ in range(self.block_count)]
            self._bank[bid][name] = data_torch
            if len(self._bank[bid]) < n_experts * 3:
                return
            # the layer's bank is complete: stack each projection
            for proj in ("gate_proj", "up_proj", "down_proj"):
                datas = []
                for xid in range(n_experts):
                    ename = f"model.layers.{bid}.mlp.bank.experts.{xid}.{proj}.weight"
                    datas.append(self._bank[bid][ename])
                    del self._bank[bid][ename]
                merged = torch.stack(datas, dim=0)
                merged_name = f"model.layers.{bid}.mlp.bank.experts.{proj}.weight"
                yield from super().modify_tensors(merged, merged_name, bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)
