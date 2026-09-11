from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


EDIT_CATEGORIES = ("Ins", "Del", "Sub", "Swp", "Dup")
EDIT_WEIGHTS = (0.2, 0.2, 0.4, 0.1, 0.1)


def class_margin(logits: Tensor, labels: Tensor) -> Tensor:
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("logits must have shape (batch, classes), with at least 2 classes")
    if labels.shape != (logits.shape[0],) or labels.dtype != torch.long:
        raise ValueError("labels must be a one-dimensional int64 tensor")
    truth = logits.gather(1, labels[:, None]).squeeze(1)
    others = logits.masked_fill(
        F.one_hot(labels, num_classes=logits.shape[1]).bool(), -torch.inf
    )
    return truth - others.amax(1)


def reverse_kl(student_logits: Tensor, teacher_logits: Tensor, temperature: float = 8.0) -> Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("teacher and student logits must have identical class ordering and shape")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    log_student = F.log_softmax(student_logits / temperature, dim=-1)
    log_teacher = F.log_softmax(teacher_logits.detach() / temperature, dim=-1)
    return (log_student.exp() * (log_student - log_teacher)).sum(-1).mean()


def edit_consistency(
    clean_logits: Tensor,
    neighbor_logits: Tensor,
    owners: Tensor,
    categories: Tensor,
    weights: Tensor | None = None,
) -> Tensor:
    if clean_logits.ndim != 2 or neighbor_logits.ndim != 2:
        raise ValueError("logits must be two-dimensional")
    if clean_logits.shape[1] != neighbor_logits.shape[1]:
        raise ValueError("class dimensions must match")
    if owners.ndim != 1 or categories.shape != owners.shape:
        raise ValueError("owners and categories must be aligned one-dimensional tensors")
    if owners.numel() != neighbor_logits.shape[0]:
        raise ValueError("neighbor metadata must match the neighbor batch")
    if owners.dtype != torch.long or categories.dtype != torch.long:
        raise TypeError("owners and categories must use int64")
    if owners.numel() == 0:
        return clean_logits.sum() * 0
    batch = clean_logits.shape[0]
    if ((owners < 0) | (owners >= batch)).any():
        raise ValueError("neighbor owner index out of range")
    if ((categories < 0) | (categories >= len(EDIT_CATEGORIES))).any():
        raise ValueError("edit category index out of range")
    if weights is None:
        weights = clean_logits.new_tensor(EDIT_WEIGHTS)
    weights = weights.to(clean_logits)
    if weights.shape != (5,) or (weights <= 0).any() or not torch.isfinite(weights).all():
        raise ValueError("five positive finite category weights are required")
    indices = owners * 5 + categories
    distances = (clean_logits.index_select(0, owners) - neighbor_logits).square().sum(-1)
    sums = clean_logits.new_zeros(batch * 5).index_add(0, indices, distances)
    counts = torch.bincount(indices, minlength=batch * 5).reshape(batch, 5)
    category_means = sums.reshape(batch, 5) / counts.clamp_min(1)
    active_weights = (counts > 0).to(clean_logits.dtype) * weights
    denominator = active_weights.sum(1, keepdim=True)
    normalized = active_weights / denominator.clamp_min(torch.finfo(clean_logits.dtype).tiny)
    return (category_means * normalized).sum(1).mean()


@dataclass(frozen=True)
class LossTerms:
    total: Tensor
    transfer: Tensor
    consistency: Tensor
    margin: Tensor


class QADObjective(nn.Module):
    def __init__(
        self,
        temperature: float = 8.0,
        beta: float = 0.7,
        target_margin: float = 1.0,
        consistency_weight: float = 0.1,
        margin_weight: float = 0.1,
        warmup_epochs: int = 5,
        ema_decay: float = 0.9,
        normalizer_floor: float = 1e-8,
    ):
        super().__init__()
        values = (temperature, beta, target_margin, consistency_weight, margin_weight, ema_decay, normalizer_floor)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("loss parameters must be finite")
        if temperature <= 0 or not 0 <= beta <= 1 or target_margin < 0:
            raise ValueError("invalid temperature, beta, or target margin")
        if consistency_weight < 0 or margin_weight < 0 or warmup_epochs < 1:
            raise ValueError("invalid regularization configuration")
        if not 0 <= ema_decay < 1 or normalizer_floor <= 0:
            raise ValueError("invalid normalizer configuration")
        self.temperature = temperature
        self.beta = beta
        self.target_margin = target_margin
        self.consistency_weight = consistency_weight
        self.margin_weight = margin_weight
        self.warmup_epochs = warmup_epochs
        self.ema_decay = ema_decay
        self.normalizer_floor = normalizer_floor
        self.register_buffer("eta_consistency", torch.tensor(1.0))
        self.register_buffer("eta_margin", torch.tensor(1.0))
        self.register_buffer("category_weights", torch.tensor(EDIT_WEIGHTS))

    def forward(
        self,
        student_logits: Tensor,
        labels: Tensor,
        epoch: int,
        teacher_logits: Tensor | None = None,
        neighbor_logits: Tensor | None = None,
        owners: Tensor | None = None,
        categories: Tensor | None = None,
    ) -> LossTerms:
        if epoch < 1:
            raise ValueError("epoch numbers start at 1")
        ce = F.cross_entropy(student_logits, labels)
        transfer = (1 - self.beta) * ce
        if self.beta > 0:
            if teacher_logits is None:
                raise ValueError("teacher logits are required when beta is positive")
            transfer = transfer + self.beta * self.temperature ** 2 * reverse_kl(
                student_logits, teacher_logits, self.temperature
            )
        if neighbor_logits is None:
            if self.consistency_weight > 0:
                raise ValueError("neighbor logits and metadata are required")
            consistency = student_logits.sum() * 0
        else:
            if owners is None or categories is None:
                raise ValueError("neighbor metadata is required")
            consistency = edit_consistency(
                student_logits, neighbor_logits, owners, categories, self.category_weights
            )
        margin = F.relu(self.target_margin - class_margin(student_logits, labels)).mean()
        ramp = min(1.0, epoch / self.warmup_epochs)
        total = transfer
        total = total + ramp * self.consistency_weight * consistency / self.eta_consistency.detach()
        total = total + ramp * self.margin_weight * margin / self.eta_margin.detach()
        return LossTerms(total, transfer, consistency, margin)

    @torch.no_grad()
    def update_normalizers(self, consistency_mean: float, margin_mean: float) -> None:
        for normalizer, value in (
            (self.eta_consistency, consistency_mean),
            (self.eta_margin, margin_mean),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError("epoch loss means must be finite and nonnegative")
            normalizer.mul_(self.ema_decay).add_((1 - self.ema_decay) * value)
            normalizer.clamp_(min=self.normalizer_floor)

    @torch.no_grad()
    def reset_normalizers(self) -> None:
        self.eta_consistency.fill_(1)
        self.eta_margin.fill_(1)


@torch.no_grad()
def sampled_sensitivity(left_logits: Tensor, right_logits: Tensor) -> Tensor:
    if left_logits.ndim != 2 or left_logits.shape != right_logits.shape or left_logits.shape[0] == 0:
        raise ValueError("a nonempty aligned batch of unit-edit logit pairs is required")
    return torch.linalg.vector_norm(left_logits - right_logits, dim=-1).amax()


@torch.no_grad()
def sampled_radius(logits: Tensor, labels: Tensor, sensitivity: float | Tensor) -> Tensor:
    margins = class_margin(logits, labels).double()
    bound = torch.as_tensor(sensitivity, dtype=torch.float64, device=logits.device)
    if bound.numel() != 1 or not torch.isfinite(bound) or bound < 0:
        raise ValueError("sampled sensitivity must be a finite nonnegative scalar")
    result = torch.full_like(margins, torch.nan)
    if bound > 0:
        valid = margins > 0
        result[valid] = (torch.ceil(margins[valid] / (math.sqrt(2) * bound)) - 1).clamp_min(0)
    return result


@torch.no_grad()
def conditional_radius(logits: Tensor, labels: Tensor, global_upper_bound: float | Tensor) -> Tensor:
    margins = class_margin(logits, labels).double()
    bound = torch.as_tensor(global_upper_bound, dtype=torch.float64, device=logits.device)
    if bound.numel() != 1 or not torch.isfinite(bound) or bound < 0:
        raise ValueError("global upper bound must be a finite nonnegative scalar")
    result = torch.full_like(margins, torch.nan)
    valid = margins > 0
    if bound == 0:
        result[valid] = torch.inf
    else:
        result[valid] = (torch.ceil(margins[valid] / (math.sqrt(2) * bound)) - 1).clamp_min(0)
    return result
