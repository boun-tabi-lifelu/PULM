
from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from plm_benchmark.config import ModelConfig
from plm_benchmark.models.peta_heads import (
    Attention1dPPIHead,
    Attention1dPoolingHead,
    PetaHeadConfig,
)
from plm_benchmark.tasks import TaskSpec


def _problem_type(spec: TaskSpec) -> str:
    """Map task type to the loss family used by the head."""
    if spec.task_type == "regression":
        return "regression"
    if spec.task_type == "multilabel":
        return "multilabel"
    return "classification"  # classification | ppi


class EsmPetaClassifier(nn.Module):
    """Single downstream model: encoder -> attention1d pooling -> linear head.

    ``attention1d`` is only the pooling mechanism; the final logits come from the
    projection head (Linear -> ReLU -> Linear). Works for any encoder exposing
    ``.last_hidden_state`` and ``.config.hidden_size`` (pretrained PLM or scratch).
    """

    def __init__(
        self,
        encoder,
        head_config: PetaHeadConfig,
        *,
        is_ppi: bool = False,
        problem_type: str = "classification",
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.is_ppi = is_ppi
        self.problem_type = problem_type
        self.freeze_encoder = freeze_encoder
        self.num_labels = head_config.num_labels
        self.head = Attention1dPPIHead(head_config) if is_ppi else Attention1dPoolingHead(head_config)

    def train(self, mode: bool = True):  # keep a frozen encoder deterministic (no dropout)
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _loss(self, logits, labels):
        if labels is None:
            return None
        if self.problem_type == "regression":
            return nn.functional.mse_loss(logits.squeeze(-1), labels.squeeze().float())
        if self.problem_type == "multilabel":
            return nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
        return nn.functional.cross_entropy(logits, labels.long())

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs) -> SequenceClassifierOutput:
        ctx = torch.no_grad() if self.freeze_encoder else nullcontext()
        with ctx:
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.head(hidden, attention_mask)
        loss = self._loss(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


def build_model(
    model_cfg: ModelConfig,
    spec: TaskSpec,
    *,
    tokenizer=None,
    freeze: bool = False,
    use_lora: bool = False,
    scratch_dim: int = 320,
    scratch_layers: int = 0,
    scratch_heads: int = 8,
    lora_r: int = 4,
) -> tuple[EsmPetaClassifier, AutoTokenizer]:
    """Assemble encoder + attention1d head for any task/backend.

    ``tokenizer`` (an explicit object) is required for the scratch baseline and
    optional otherwise (defaults to the checkpoint's own tokenizer).
    """
    is_ppi = spec.task_type == "ppi"

    if model_cfg.backend == "scratch":
        if tokenizer is None:
            raise ValueError("The scratch baseline requires an explicit tokenizer (--tokenizer).")
        from plm_benchmark.models.scratch import ScratchEncoder

        encoder = ScratchEncoder(
            vocab_size=len(tokenizer),
            hidden_size=scratch_dim,
            num_layers=scratch_layers,
            num_heads=scratch_heads,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_cfg.checkpoint)
        model_type = AutoConfig.from_pretrained(model_cfg.checkpoint).model_type
        enc_kwargs: dict = {"add_pooling_layer": False} if model_type == "esm" else {}
        encoder = AutoModel.from_pretrained(model_cfg.checkpoint, **enc_kwargs)
        if use_lora:
            from peft import LoraConfig, inject_adapter_in_model

            lora_cfg = LoraConfig(
                r=lora_r, lora_alpha=1, bias="all", target_modules=["query", "key", "value", "dense"]
            )
            encoder = inject_adapter_in_model(lora_cfg, encoder)

    if freeze:
        for param in encoder.parameters():
            param.requires_grad = False

    head_config = PetaHeadConfig(hidden_size=encoder.config.hidden_size, num_labels=spec.num_labels)
    model = EsmPetaClassifier(
        encoder,
        head_config,
        is_ppi=is_ppi,
        problem_type=_problem_type(spec),
        freeze_encoder=freeze,
    )
    return model, tokenizer
