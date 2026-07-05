
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from plm_benchmark.models.peta_heads import (
    Attention1dPPIHead,
    Attention1dPoolingHead,
    PetaHeadConfig,
)


class EsmPetaClassifier(nn.Module):
    def __init__(
        self,
        encoder,
        head_config: PetaHeadConfig,
        *,
        is_ppi: bool = False,
        problem_type: str = "classification",
    ):
        super().__init__()
        self.encoder = encoder
        self.is_ppi = is_ppi
        self.problem_type = problem_type
        self.num_labels = head_config.num_labels
        self.head = Attention1dPPIHead(head_config) if is_ppi else Attention1dPoolingHead(head_config)

    def _loss(self, logits, labels):
        if labels is None:
            return None
        if self.problem_type == "regression":
            return nn.functional.mse_loss(logits.squeeze(-1), labels.squeeze().float())
        if self.problem_type == "multilabel":
            return nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
        return nn.functional.cross_entropy(logits, labels.long())

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs) -> SequenceClassifierOutput:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.head(hidden, attention_mask)
        loss = self._loss(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


def load_peta_model(
    checkpoint: str,
    num_labels: int,
    *,
    is_ppi: bool = False,
    problem_type: str = "classification",
    train_encoder: bool = True,
) -> tuple[EsmPetaClassifier, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model_type = AutoConfig.from_pretrained(checkpoint).model_type
    enc_kwargs: dict = {}
    if model_type == "esm":
        enc_kwargs["add_pooling_layer"] = False
    encoder = AutoModel.from_pretrained(checkpoint, **enc_kwargs)

    if not train_encoder:
        for param in encoder.parameters():
            param.requires_grad = False

    head_config = PetaHeadConfig(hidden_size=encoder.config.hidden_size, num_labels=num_labels)
    model = EsmPetaClassifier(
        encoder, head_config, is_ppi=is_ppi, problem_type=problem_type
    )
    return model, tokenizer
