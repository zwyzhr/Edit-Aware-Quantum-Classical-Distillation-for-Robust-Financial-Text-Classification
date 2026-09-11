from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
import unicodedata

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


CANONICAL_CLASSES = ("negative", "neutral", "positive")


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return " ".join(unicodedata.normalize("NFC", text).split())


def _transformer_inputs(tokenizer, texts: Sequence[str], device, max_tokens: int):
    if isinstance(texts, str) or len(texts) == 0:
        raise ValueError("a nonempty sequence of texts is required")
    return tokenizer(
        list(texts), padding=True, truncation=True,
        max_length=max_tokens, return_tensors="pt"
    ).to(device)


class FinBERTEncoder(nn.Module):
    def __init__(self, model_name_or_path: str = "ProsusAI/finbert", max_tokens: int = 128):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        if max_tokens < 2:
            raise ValueError("max_tokens must be at least 2")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.max_tokens = max_tokens
        self.output_dim = self.model.config.hidden_size

    def pretrained_parameters(self) -> Iterable[nn.Parameter]:
        return self.model.parameters()

    def forward(self, texts: Sequence[str]) -> Tensor:
        device = next(self.model.parameters()).device
        inputs = _transformer_inputs(self.tokenizer, texts, device, self.max_tokens)
        return self.model(**inputs, return_dict=True).last_hidden_state[:, 0]


class FinBERTTeacher(nn.Module):
    def __init__(
        self,
        model_name_or_path: str = "ProsusAI/finbert",
        max_tokens: int = 128,
        source_class_order: Sequence[str] | None = None,
    ):
        super().__init__()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if max_tokens < 2:
            raise ValueError("max_tokens must be at least 2")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.max_tokens = max_tokens
        self.class_order = CANONICAL_CLASSES
        if source_class_order is None:
            label_map = self.model.config.id2label
            source_class_order = [
                label_map.get(index, label_map.get(str(index), ""))
                for index in range(self.model.config.num_labels)
            ]
        source_class_order = tuple(str(label).lower() for label in source_class_order)
        if len(source_class_order) != 3 or set(source_class_order) != set(CANONICAL_CLASSES):
            raise ValueError("source_class_order must identify negative, neutral, and positive logits")
        self.register_buffer(
            "class_indices",
            torch.tensor([source_class_order.index(label) for label in CANONICAL_CLASSES]),
        )

    def pretrained_parameters(self) -> Iterable[nn.Parameter]:
        return self.model.base_model.parameters()

    def forward(self, texts: Sequence[str]) -> Tensor:
        device = next(self.model.parameters()).device
        inputs = _transformer_inputs(self.tokenizer, texts, device, self.max_tokens)
        logits = self.model(**inputs, return_dict=True).logits
        return logits.index_select(1, self.class_indices)


class TfidfSVDEncoder(nn.Module):
    def __init__(self, output_dim: int = 256, seed: int = 17):
        super().__init__()
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        if output_dim < 1:
            raise ValueError("output_dim must be positive")
        self.output_dim = output_dim
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_features=20000, lowercase=False
        )
        self.svd = TruncatedSVD(n_components=output_dim, random_state=seed)
        self.fitted = False
        self.register_buffer("anchor", torch.empty(0), persistent=False)

    def fit(self, training_texts: Sequence[str]) -> TfidfSVDEncoder:
        if isinstance(training_texts, str) or not len(training_texts):
            raise ValueError("training_texts must be a nonempty sequence")
        matrix = self.vectorizer.fit_transform([normalize_text(text) for text in training_texts])
        if self.output_dim > min(matrix.shape):
            raise ValueError("output_dim exceeds the available training matrix dimensions")
        self.svd.fit(matrix)
        self.fitted = True
        return self

    def forward(self, texts: Sequence[str]) -> Tensor:
        if not self.fitted:
            raise RuntimeError("fit the TF-IDF and SVD transforms on training texts first")
        if isinstance(texts, str) or not len(texts):
            raise ValueError("a nonempty sequence of texts is required")
        matrix = self.vectorizer.transform([normalize_text(text) for text in texts])
        features = self.svd.transform(matrix)
        tensor = torch.as_tensor(features, dtype=self.anchor.dtype, device=self.anchor.device)
        return F.normalize(tensor, p=2, dim=-1)

    def get_extra_state(self):
        return self.vectorizer, self.svd, self.fitted

    def set_extra_state(self, state) -> None:
        self.vectorizer, self.svd, self.fitted = state


class Vocabulary:
    def __init__(self, tokens: Sequence[str], tokenize: Callable[[str], Sequence[str]]):
        if len(set(tokens)) != len(tokens):
            raise ValueError("vocabulary tokens must be unique")
        self.tokenize = tokenize
        self.indices = {token: index + 2 for index, token in enumerate(tokens)}
        self.pad_id = 0
        self.unknown_id = 1

    @classmethod
    def from_training_texts(
        cls,
        training_texts: Sequence[str],
        tokenize: Callable[[str], Sequence[str]],
        max_size: int | None = None,
    ) -> Vocabulary:
        if max_size is not None and max_size < 1:
            raise ValueError("max_size must be positive")
        counts = Counter(
            token for text in training_texts for token in tokenize(normalize_text(text))
        )
        tokens = sorted(counts, key=lambda token: (-counts[token], token))
        return cls(tokens[:max_size], tokenize)

    def __len__(self) -> int:
        return len(self.indices) + 2

    def encode(self, text: str, max_tokens: int) -> list[int]:
        tokens = self.tokenize(text)
        return [self.indices.get(token, self.unknown_id) for token in tokens[:max_tokens]]


def _recurrent_states(recurrent: nn.Module, embeddings: Tensor, mask: Tensor) -> Tensor:
    lengths = mask.sum(-1)
    if (lengths < 1).any():
        raise ValueError("recurrent sequences must contain at least one token")
    packed = pack_padded_sequence(
        embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
    )
    encoded, _ = recurrent(packed)
    states, _ = pad_packed_sequence(
        encoded, batch_first=True, total_length=embeddings.shape[1]
    )
    return states


class RecurrentEncoder(nn.Module):
    def __init__(
        self,
        vocabulary: Vocabulary,
        cell: str = "lstm",
        embedding_dim: int = 200,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        max_tokens: int = 128,
    ):
        super().__init__()
        if cell.lower() not in ("lstm", "gru") or max_tokens < 1:
            raise ValueError("cell must be lstm or gru and max_tokens must be positive")
        self.vocabulary = vocabulary
        self.max_tokens = max_tokens
        self.output_dim = 2 * hidden_dim
        self.embedding = nn.Embedding(len(vocabulary), embedding_dim, padding_idx=0)
        recurrent = nn.LSTM if cell.lower() == "lstm" else nn.GRU
        self.recurrent = recurrent(
            embedding_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, texts: Sequence[str]) -> Tensor:
        if isinstance(texts, str) or not len(texts):
            raise ValueError("a nonempty sequence of texts is required")
        sequences = [
            self.vocabulary.encode(text, self.max_tokens) or [self.vocabulary.unknown_id]
            for text in texts
        ]
        width = max(map(len, sequences))
        ids = torch.zeros(len(sequences), width, dtype=torch.long, device=self.embedding.weight.device)
        for index, sequence in enumerate(sequences):
            ids[index, :len(sequence)] = torch.tensor(sequence, device=ids.device)
        mask = ids.ne(0)
        states = _recurrent_states(self.recurrent, self.dropout(self.embedding(ids)), mask)
        features = (states * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return self.dropout(features)


class _AttentionPool(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, attention_dim)
        self.context = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, states: Tensor, mask: Tensor) -> Tensor:
        scores = self.context(torch.tanh(self.projection(states))).squeeze(-1)
        weights = F.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)
        return (states * weights.unsqueeze(-1)).sum(1)


class HANEncoder(nn.Module):
    def __init__(
        self,
        vocabulary: Vocabulary,
        split_sentences: Callable[[str], Sequence[str]],
        embedding_dim: int,
        hidden_dim: int = 64,
        attention_dim: int = 64,
        dropout: float = 0.2,
        max_tokens: int = 128,
    ):
        super().__init__()
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.vocabulary = vocabulary
        self.split_sentences = split_sentences
        self.max_tokens = max_tokens
        self.output_dim = hidden_dim * 2
        self.embedding = nn.Embedding(len(vocabulary), embedding_dim, padding_idx=0)
        self.word_gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.sentence_gru = nn.GRU(self.output_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.word_attention = _AttentionPool(self.output_dim, attention_dim)
        self.sentence_attention = _AttentionPool(self.output_dim, attention_dim)
        self.dropout = nn.Dropout(dropout)

    def _document(self, text: str) -> list[list[int]]:
        sentences = []
        remaining = self.max_tokens
        for sentence in self.split_sentences(text):
            tokens = self.vocabulary.encode(sentence, remaining)
            if tokens:
                sentences.append(tokens)
                remaining -= len(tokens)
            if remaining == 0:
                break
        return sentences or [[self.vocabulary.unknown_id]]

    def forward(self, texts: Sequence[str]) -> Tensor:
        if isinstance(texts, str) or not len(texts):
            raise ValueError("a nonempty sequence of texts is required")
        documents = [self._document(text) for text in texts]
        sentences = [sentence for document in documents for sentence in document]
        width = max(map(len, sentences))
        ids = torch.zeros(
            len(sentences), width, dtype=torch.long, device=self.embedding.weight.device
        )
        for index, sentence in enumerate(sentences):
            ids[index, :len(sentence)] = torch.tensor(sentence, device=ids.device)
        word_mask = ids.ne(0)
        word_states = _recurrent_states(self.word_gru, self.dropout(self.embedding(ids)), word_mask)
        sentence_features = self.word_attention(word_states, word_mask)
        max_sentences = max(map(len, documents))
        sentence_inputs = sentence_features.new_zeros(len(documents), max_sentences, self.output_dim)
        sentence_mask = torch.zeros(len(documents), max_sentences, dtype=torch.bool, device=ids.device)
        offset = 0
        for index, document in enumerate(documents):
            count = len(document)
            sentence_inputs[index, :count] = sentence_features[offset:offset + count]
            sentence_mask[index, :count] = True
            offset += count
        sentence_states = _recurrent_states(
            self.sentence_gru, self.dropout(sentence_inputs), sentence_mask
        )
        return self.dropout(self.sentence_attention(sentence_states, sentence_mask))


class TextClassifier(nn.Module):
    def __init__(
        self, encoder: nn.Module, head: nn.Module,
        class_order: Sequence[str] = CANONICAL_CLASSES,
    ):
        super().__init__()
        if len(class_order) < 2 or len(set(class_order)) != len(class_order):
            raise ValueError("class_order must contain distinct class labels")
        self.encoder = encoder
        self.head = head
        self.class_order = tuple(class_order)

    def pretrained_parameters(self) -> Iterable[nn.Parameter]:
        method = getattr(self.encoder, "pretrained_parameters", None)
        return method() if method is not None else ()

    def forward(self, texts: Sequence[str]) -> Tensor:
        logits = self.head(self.encoder(texts))
        if logits.ndim != 2 or logits.shape[1] != len(self.class_order):
            raise ValueError("head outputs must match class_order")
        return logits
