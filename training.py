from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
import random

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .edits import NeighborhoodSampler
from .encoders import normalize_text
from .objectives import QADObjective
from .quantum import VariationalCircuit


RUN_SEEDS = (17, 29, 43, 71, 101)


def seed_everything(seed: int) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 17
    batch_size: int = 32
    max_epochs: int = 20
    warmup_epochs: int = 5
    patience: int = 3
    min_epochs: int = 8
    pretrained_lr: float = 2e-5
    new_lr: float = 1e-3
    weight_decay: float = 0.01

    def __post_init__(self):
        if min(self.batch_size, self.max_epochs, self.warmup_epochs, self.patience, self.min_epochs) < 1:
            raise ValueError("training counts must be positive")
        if self.min_epochs > self.max_epochs:
            raise ValueError("min_epochs must not exceed max_epochs")
        if not all(math.isfinite(value) for value in (self.pretrained_lr, self.new_lr, self.weight_decay)):
            raise ValueError("optimizer settings must be finite")
        if self.pretrained_lr <= 0 or self.new_lr <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")


@dataclass(frozen=True)
class FitResult:
    model: nn.Module
    epoch: int
    validation_accuracy: float


class _TextDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int] | Tensor):
        if isinstance(texts, str) or not len(texts) or len(texts) != len(labels):
            raise ValueError("nonempty aligned texts and labels are required")
        self.texts = tuple(normalize_text(text) for text in texts)
        if any(not text for text in self.texts):
            raise ValueError("empty texts are not permitted")
        labels = torch.as_tensor(labels).detach().cpu()
        if labels.ndim != 1:
            raise ValueError("labels must be one-dimensional")
        if labels.is_complex() or (labels.is_floating_point() and not torch.equal(labels, labels.round())):
            raise ValueError("labels must be integers")
        self.labels = labels.long()
        if (self.labels < 0).any():
            raise ValueError("labels must be nonnegative")
        observed = {}
        for text, label in zip(self.texts, self.labels.tolist()):
            if text in observed and observed[text] != label:
                raise ValueError("identical normalized texts have conflicting labels")
            observed[text] = label

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int):
        return self.texts[index], self.labels[index]


def _collate(batch):
    texts, labels = zip(*batch)
    return list(texts), torch.stack(labels)


def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.AdamW:
    method = getattr(model, "pretrained_parameters", None)
    pretrained_ids = {id(parameter) for parameter in method()} if method is not None else set()
    rotation_ids = {
        id(module.rotations)
        for module in model.modules()
        if isinstance(module, VariationalCircuit)
    }
    groups = {}
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        learning_rate = config.pretrained_lr if id(parameter) in pretrained_ids else config.new_lr
        decay = config.weight_decay if parameter.ndim == 2 and id(parameter) not in rotation_ids else 0.0
        groups.setdefault((learning_rate, decay), []).append(parameter)
    if not groups:
        raise ValueError("the student has no trainable parameters")
    return torch.optim.AdamW([
        {"params": parameters, "lr": lr, "initial_lr": lr, "weight_decay": decay}
        for (lr, decay), parameters in groups.items()
    ])


class Trainer:
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module | None = None,
        objective: QADObjective | None = None,
        sampler: NeighborhoodSampler | None = None,
        config: TrainingConfig | None = None,
        device: str | torch.device | None = None,
    ):
        self.config = config or TrainingConfig()
        self.device = torch.device(device) if device is not None else next(student.parameters()).device
        self.student = student.to(self.device)
        self.objective = (objective or QADObjective()).to(self.device)
        self.sampler = sampler
        if self.objective.beta > 0 and teacher is None:
            raise ValueError("teacher is required for distillation")
        if self.objective.consistency_weight > 0 and sampler is None:
            raise ValueError("a valid-neighbor sampler is required for edit consistency")
        if teacher is not None:
            student_ids = {id(parameter) for parameter in student.parameters()}
            if any(id(parameter) in student_ids for parameter in teacher.parameters()):
                raise ValueError("teacher and student must not share parameter objects")
            student_order = getattr(student, "class_order", None)
            teacher_order = getattr(teacher, "class_order", None)
            if student_order is None or teacher_order is None or tuple(student_order) != tuple(teacher_order):
                raise ValueError("teacher and student must declare matching class_order")
            teacher.to(self.device).requires_grad_(False).eval()
        self.teacher = teacher

    @torch.inference_mode()
    def _validation_accuracy(self, loader: DataLoader) -> float:
        self.student.eval()
        correct = count = 0
        for texts, labels in loader:
            logits = self.student(texts)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("nonfinite validation logits")
            predictions = logits.argmax(-1).cpu()
            correct += predictions.eq(labels).sum().item()
            count += labels.numel()
        return correct / count

    def fit(
        self,
        training_texts: Sequence[str],
        training_labels: Sequence[int] | Tensor,
        validation_texts: Sequence[str],
        validation_labels: Sequence[int] | Tensor,
    ) -> FitResult:
        seed_everything(self.config.seed)
        training = _TextDataset(training_texts, training_labels)
        validation = _TextDataset(validation_texts, validation_labels)
        if set(training.texts).intersection(validation.texts):
            raise ValueError("training and validation texts must be disjoint")
        class_order = getattr(self.student, "class_order", None)
        if class_order is not None:
            for partition in (training, validation):
                if (partition.labels >= len(class_order)).any():
                    raise ValueError("labels exceed the declared class ordering")
        generator = torch.Generator().manual_seed(self.config.seed)
        train_loader = DataLoader(
            training, batch_size=self.config.batch_size, shuffle=True,
            collate_fn=_collate, generator=generator, num_workers=0,
        )
        validation_loader = DataLoader(
            validation, batch_size=self.config.batch_size, shuffle=False,
            collate_fn=_collate, num_workers=0,
        )
        self.objective.reset_normalizers()
        if self.sampler is not None:
            self.sampler.reset(self.config.seed)
        optimizer = build_optimizer(self.student, self.config)
        warmup_steps = self.config.warmup_epochs * len(train_loader)
        total_steps = self.config.max_epochs * len(train_loader)
        best_accuracy = -math.inf
        best_epoch = stale_epochs = global_step = 0
        best_state = None
        for epoch in range(1, self.config.max_epochs + 1):
            self.student.train()
            if self.teacher is not None:
                self.teacher.eval()
            consistency_sum = margin_sum = 0.0
            samples = 0
            for texts, labels in train_loader:
                labels = labels.to(self.device)
                neighbors = None
                if self.objective.consistency_weight > 0:
                    neighbors = self.sampler.sample(texts)
                teacher_logits = None
                if self.objective.beta > 0:
                    with torch.no_grad():
                        teacher_logits = self.teacher(texts)
                optimizer.zero_grad(set_to_none=True)
                clean_logits = self.student(texts)
                neighbor_logits = owners = categories = None
                if neighbors is not None:
                    neighbor_logits = (
                        self.student(neighbors.texts)
                        if neighbors.texts else clean_logits.new_empty((0, clean_logits.shape[1]))
                    )
                    owners = torch.tensor(neighbors.owners, dtype=torch.long, device=self.device)
                    categories = torch.tensor(neighbors.categories, dtype=torch.long, device=self.device)
                losses = self.objective(
                    clean_logits, labels, epoch, teacher_logits,
                    neighbor_logits, owners, categories,
                )
                if not torch.isfinite(losses.total):
                    raise FloatingPointError("nonfinite training loss")
                losses.total.backward()
                global_step += 1
                if global_step <= warmup_steps:
                    multiplier = global_step / warmup_steps
                else:
                    progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                    multiplier = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * multiplier
                optimizer.step()
                size = labels.numel()
                consistency_sum += losses.consistency.detach().item() * size
                margin_sum += losses.margin.detach().item() * size
                samples += size
            self.objective.update_normalizers(consistency_sum / samples, margin_sum / samples)
            accuracy = self._validation_accuracy(validation_loader)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
                stale_epochs = 0
                best_state = {
                    key: value.detach().cpu().clone() if isinstance(value, Tensor) else deepcopy(value)
                    for key, value in self.student.state_dict().items()
                }
            elif epoch > self.config.warmup_epochs:
                stale_epochs += 1
            if epoch >= self.config.min_epochs and stale_epochs >= self.config.patience:
                break
        if best_state is None:
            raise RuntimeError("no validation-selected state is available")
        self.student.load_state_dict(best_state)
        self.student.eval()
        return FitResult(self.student, best_epoch, best_accuracy)
