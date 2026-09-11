from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .edits import Edit, EditValidator, Validity, ordered_edits


@dataclass(frozen=True)
class AttackConfig:
    radius: int = 2
    beam_width: int = 10
    call_budget: int = 500
    candidate_cap: int = 500
    seed: int = 17

    def __post_init__(self):
        if self.radius < 1 or self.beam_width < 1:
            raise ValueError("radius and beam_width must be positive")
        if self.call_budget < 0 or self.candidate_cap < 0:
            raise ValueError("budgets must be nonnegative")


@dataclass(frozen=True)
class ScoredEdit:
    edit: Edit
    depth: int
    score: float
    prediction: int


@dataclass(frozen=True)
class CheckedEdit:
    edit: Edit
    depth: int
    validity: Validity


@dataclass(frozen=True)
class AttackResult:
    original: str
    adversarial: str | None
    clean_prediction: int
    eligible: bool
    success: bool | None
    completed: bool
    clean_calls: int
    attack_calls: int
    scored_candidates: int
    generated_candidates: int
    rejected_candidates: int
    invalid_candidates: int
    unresolved_candidates: int
    validator_revision: str
    stopping_reason: str
    path: tuple[Edit, ...]
    trace: tuple[ScoredEdit, ...]
    checks: tuple[CheckedEdit, ...] = ()


@dataclass(frozen=True)
class AttackMetrics:
    total: int
    clean_correct: int
    completed_attacks: int
    successful_attacks: int
    clean_accuracy: float
    attack_success_rate: float | None
    full_test_survival: float
    clean_calls: int
    attack_calls: int


def summarize_attacks(results: Sequence[AttackResult]) -> AttackMetrics:
    if not results:
        raise ValueError("at least one evaluated input is required")
    eligible = [result for result in results if result.eligible]
    for result in eligible:
        if not result.completed or result.unresolved_candidates or result.success is None:
            raise ValueError("ASR requires completed attacks with resolved validity on all clean-correct inputs")
        if result.success and result.adversarial is None:
            raise ValueError("a successful attack must include its adversarial text")
    total = len(results)
    correct = len(eligible)
    successful = sum(result.success is True for result in eligible)
    return AttackMetrics(
        total=total,
        clean_correct=correct,
        completed_attacks=correct,
        successful_attacks=successful,
        clean_accuracy=100.0 * correct / total,
        attack_success_rate=100.0 * successful / correct if correct else None,
        full_test_survival=100.0 * (correct - successful) / total,
        clean_calls=sum(result.clean_calls for result in results),
        attack_calls=sum(result.attack_calls for result in results),
    )


class BeamSearchAttack:
    def __init__(
        self,
        classifier: nn.Module,
        validator: EditValidator,
        config: AttackConfig | None = None,
    ):
        self.classifier = classifier
        self.validator = validator
        self.config = config or AttackConfig()

    def _score(self, text: str, label: int) -> tuple[int, float]:
        logits = self.classifier([text])
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError("classifier must return a (1, classes) logit tensor")
        if logits.shape[1] < 2 or not 0 <= label < logits.shape[1]:
            raise ValueError("label is outside the classifier's class range")
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite attack logits")
        row = logits[0]
        competitor = row.clone()
        competitor[label] = -torch.inf
        return int(row.argmax().item()), float((competitor.max() - row[label]).item())

    @torch.inference_mode()
    def __call__(self, original: str, label: int) -> AttackResult:
        if not isinstance(original, str) or not original.strip():
            raise ValueError("a nonempty original text is required")
        modes = [(module, module.training) for module in self.classifier.modules()]
        self.classifier.eval()
        try:
            return self._search(original, label)
        finally:
            for module, training in modes:
                module.training = training

    def _search(self, original: str, label: int) -> AttackResult:
        config = self.config
        clean_prediction, _ = self._score(original, label)
        eligible = clean_prediction == label
        calls = scored = generated = rejected = invalid = unresolved = 0
        trace = []
        checks = []
        paths = {original: ()}

        def result(
            reason: str,
            success: bool | None = False,
            adversarial: str | None = None,
            completed: bool = True,
        ) -> AttackResult:
            return AttackResult(
                original=original,
                adversarial=adversarial,
                clean_prediction=clean_prediction,
                eligible=eligible,
                success=None if unresolved and success is not True else success,
                completed=completed and eligible and unresolved == 0,
                clean_calls=1,
                attack_calls=calls,
                scored_candidates=scored,
                generated_candidates=generated,
                rejected_candidates=rejected,
                invalid_candidates=invalid,
                unresolved_candidates=unresolved,
                validator_revision=self.validator.revision,
                stopping_reason=reason,
                path=paths[adversarial] if adversarial is not None else (),
                trace=tuple(trace),
                checks=tuple(checks),
            )

        if not eligible:
            return result("initially_incorrect", success=None, completed=False)
        if config.call_budget < config.radius or config.candidate_cap < config.radius:
            return result("insufficient_depth_allocation", success=None, completed=False)
        beam = [original]
        visited = {original}
        for depth in range(1, config.radius + 1):
            remaining_depths = config.radius - depth + 1
            call_quota = (config.call_budget - calls) // remaining_depths
            candidate_quota = (config.candidate_cap - scored) // remaining_depths
            starting_calls, starting_scored = calls, scored
            current = []
            for edit in ordered_edits(beam, config.seed):
                if calls - starting_calls >= call_quota or scored - starting_scored >= candidate_quota:
                    break
                generated += 1
                if edit.text in visited:
                    rejected += 1
                    continue
                validity = self.validator(original, edit.text)
                if not isinstance(validity, Validity):
                    raise TypeError("validator must return a Validity value")
                checks.append(CheckedEdit(edit, depth, validity))
                if validity is Validity.INVALID:
                    invalid += 1
                    continue
                if validity is Validity.UNRESOLVED:
                    unresolved += 1
                    continue
                prediction, score = self._score(edit.text, label)
                calls += 1
                scored += 1
                visited.add(edit.text)
                paths[edit.text] = paths[edit.parent] + (edit,)
                record = ScoredEdit(edit, depth, score, prediction)
                current.append(record)
                trace.append(record)
                if prediction != label:
                    return result("valid_misclassification", True, edit.text)
            if not current:
                reason = "unresolved_validity" if unresolved else "no_valid_extension"
                return result(reason)
            current.sort(key=lambda record: -record.score)
            beam = [record.edit.text for record in current[:config.beam_width]]
        reason = "unresolved_validity" if unresolved else "search_allocation_exhausted"
        return result(reason)
