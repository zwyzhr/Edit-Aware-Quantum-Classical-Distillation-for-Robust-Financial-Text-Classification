from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from hashlib import sha256
from itertools import zip_longest
import math
import random
import string

import torch
from torch.nn import functional as F

from objectives import EDIT_CATEGORIES


class Validity(Enum):
    VALID = "valid"
    INVALID = "invalid"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class Edit:
    parent: str
    text: str
    category: str


@dataclass(frozen=True)
class NeighborBatch:
    texts: tuple[str, ...]
    owners: tuple[int, ...]
    categories: tuple[int, ...]


def _order_key(seed: int, category: str, text: str) -> bytes:
    data = f"{seed}\0{category}\0{text}".encode("utf-8")
    return sha256(data).digest()


def unit_edit_categories(text: str) -> dict[str, set[str]]:
    alphabet = string.ascii_uppercase + string.ascii_lowercase
    edits = {category: set() for category in EDIT_CATEGORIES}
    for position in range(len(text) + 1):
        for character in alphabet:
            edits["Ins"].add(text[:position] + character + text[position:])
    for position, original in enumerate(text):
        edits["Del"].add(text[:position] + text[position + 1:])
        edits["Dup"].add(text[:position] + original + text[position:])
        for character in alphabet:
            if character != original:
                edits["Sub"].add(text[:position] + character + text[position + 1:])
        if position + 1 < len(text) and original != text[position + 1]:
            edits["Swp"].add(
                text[:position] + text[position + 1] + original + text[position + 2:]
            )
    return edits


def ordered_edits(parents: Sequence[str], seed: int = 17) -> Iterator[Edit]:
    grouped = {category: {} for category in EDIT_CATEGORIES}
    for parent in parents:
        for category, candidates in unit_edit_categories(parent).items():
            for text in candidates:
                grouped[category].setdefault(text, Edit(parent, text, category))
    streams = []
    for category in EDIT_CATEGORIES:
        texts = sorted(
            grouped[category], key=lambda text: (_order_key(seed, category, text), text)
        )
        streams.append([grouped[category][text] for text in texts])
    seen = set()
    for row in zip_longest(*streams):
        for edit in row:
            if edit is not None and edit.text not in seen:
                seen.add(edit.text)
                yield edit


class SemanticScreen:
    def __init__(
        self,
        device: str = "cpu",
        cosine_threshold: float = 0.90,
        perplexity_threshold: float = 45.0,
        sentence_model: str = "sentence-transformers/all-mpnet-base-v2",
        language_model: str = "gpt2",
    ):
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not math.isfinite(cosine_threshold) or not -1 <= cosine_threshold <= 1:
            raise ValueError("cosine_threshold must lie in [-1, 1]")
        if not math.isfinite(perplexity_threshold) or perplexity_threshold <= 0:
            raise ValueError("perplexity_threshold must be positive")
        self.cosine_threshold = cosine_threshold
        self.perplexity_threshold = perplexity_threshold
        self.device = torch.device(device)
        self.encoder = SentenceTransformer(sentence_model, device=str(self.device))
        self.encoder.eval().requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(language_model)
        self.language_model = AutoModelForCausalLM.from_pretrained(language_model).to(self.device)
        self.language_model.eval().requires_grad_(False)
        self.context_size = int(self.language_model.config.max_position_embeddings)
        if self.context_size < 2:
            raise ValueError("language model context must contain at least two tokens")

    @lru_cache(maxsize=4096)
    @torch.inference_mode()
    def embedding(self, text: str) -> torch.Tensor:
        return self.encoder.encode(
            text, convert_to_tensor=True, normalize_embeddings=True,
            show_progress_bar=False,
        ).detach().cpu()

    @lru_cache(maxsize=4096)
    @torch.inference_mode()
    def perplexity(self, text: str) -> float:
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        length = ids.shape[1]
        if length < 2:
            return math.inf
        stride = max(1, self.context_size // 2)
        previous_end = 1
        total_loss = 0.0
        token_count = 0
        for start in range(0, length, stride):
            end = min(start + self.context_size, length)
            first_target = max(previous_end, start + 1)
            if first_target < end:
                outputs = self.language_model(
                    input_ids=ids[:, start:end], use_cache=False, return_dict=True
                )
                logits = outputs.logits[:, first_target - start - 1:-1]
                targets = ids[:, first_target:end]
                total_loss += F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum"
                ).item()
                token_count += targets.numel()
            previous_end = end
            if end == length:
                break
        mean_loss = total_loss / token_count
        return math.exp(mean_loss) if mean_loss < 709 else math.inf

    def __call__(self, original: str, candidate: str) -> bool:
        if not candidate.strip():
            return False
        cosine = torch.dot(self.embedding(original), self.embedding(candidate)).item()
        if not math.isfinite(cosine) or cosine < self.cosine_threshold:
            return False
        perplexity = self.perplexity(candidate)
        return math.isfinite(perplexity) and perplexity <= self.perplexity_threshold


class EditValidator:
    def __init__(
        self,
        financial_validator: Callable[[str, str], bool | None],
        sentiment_validator: Callable[[str, str], bool | None],
        screen: Callable[[str, str], bool],
        revision: str,
    ):
        if not revision.strip():
            raise ValueError("a validator revision identifier is required")
        self.financial_validator = financial_validator
        self.sentiment_validator = sentiment_validator
        self.screen = screen
        self.revision = revision

    @staticmethod
    def _decision(value: bool | None) -> Validity:
        if value is None:
            return Validity.UNRESOLVED
        if not isinstance(value, bool):
            raise TypeError("validity predicates must return bool or None")
        return Validity.VALID if value else Validity.INVALID

    def __call__(self, original: str, candidate: str) -> Validity:
        if not original.strip() or not candidate.strip():
            return Validity.INVALID
        forward = self._decision(self.financial_validator(original, candidate))
        backward = self._decision(self.financial_validator(candidate, original))
        if Validity.INVALID in (forward, backward):
            return Validity.INVALID
        screened = self._decision(self.screen(original, candidate))
        if screened is Validity.INVALID:
            return Validity.INVALID
        sentiment_forward = self._decision(self.sentiment_validator(original, candidate))
        sentiment_backward = self._decision(self.sentiment_validator(candidate, original))
        if Validity.INVALID in (sentiment_forward, sentiment_backward):
            return Validity.INVALID
        decisions = (forward, backward, screened, sentiment_forward, sentiment_backward)
        return Validity.UNRESOLVED if Validity.UNRESOLVED in decisions else Validity.VALID


class NeighborhoodSampler:
    def __init__(
        self,
        validator: EditValidator,
        neighbors_per_category: int | Mapping[str, int],
        seed: int = 17,
    ):
        self.validator = validator
        if isinstance(neighbors_per_category, int):
            counts = {category: neighbors_per_category for category in EDIT_CATEGORIES}
        else:
            if set(neighbors_per_category) - set(EDIT_CATEGORIES):
                raise ValueError("unknown edit category")
            counts = {category: neighbors_per_category.get(category, 0) for category in EDIT_CATEGORIES}
        if any(not isinstance(count, int) or count < 0 for count in counts.values()):
            raise ValueError("neighbor counts must be nonnegative integers")
        if not any(counts.values()):
            raise ValueError("at least one neighbor category must be enabled")
        self.counts = counts
        self.seed = seed
        self.rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        self.rng.seed(self.seed if seed is None else seed)

    def sample(self, texts: Sequence[str]) -> NeighborBatch:
        neighbors, owners, categories = [], [], []
        for owner, original in enumerate(texts):
            candidates = unit_edit_categories(original)
            seed = self.rng.getrandbits(64)
            for category_index, category in enumerate(EDIT_CATEGORIES):
                limit = self.counts[category]
                if limit == 0:
                    continue
                ordered = sorted(
                    candidates[category],
                    key=lambda text: (_order_key(seed, category, text), text),
                )
                accepted = 0
                for candidate in ordered:
                    if self.validator(original, candidate) is Validity.VALID:
                        neighbors.append(candidate)
                        owners.append(owner)
                        categories.append(category_index)
                        accepted += 1
                        if accepted == limit:
                            break
        return NeighborBatch(tuple(neighbors), tuple(owners), tuple(categories))
