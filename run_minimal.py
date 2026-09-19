from __future__ import annotations

import argparse
import math

import torch

from objectives import QADObjective
from quantum import MixedHead, QuantumHead, TanhHead


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a finite-value QAD interface smoke test.")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    features = torch.tensor(
        [[0.20, -0.10, 0.40, 0.30], [-0.30, 0.50, 0.10, -0.20]],
        dtype=torch.float32,
    )
    neighbor_features = features + torch.tensor(
        [[0.01, 0.00, -0.01, 0.00], [0.00, -0.01, 0.00, 0.01]]
    )
    labels = torch.tensor([2, 0], dtype=torch.long)

    quantum = QuantumHead(input_dim=4, classes=3, qubits=2, layers=1)
    student = MixedHead(quantum, TanhHead(input_dim=4, classes=3, hidden_dim=3))
    teacher = TanhHead(input_dim=4, classes=3, hidden_dim=4).eval()
    objective = QADObjective(warmup_epochs=1)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

    with torch.no_grad():
        teacher_logits = teacher(features)
    clean_logits = student(features)
    neighbor_logits = student(neighbor_features)
    losses = objective(
        clean_logits,
        labels,
        epoch=1,
        teacher_logits=teacher_logits,
        neighbor_logits=neighbor_logits,
        owners=torch.tensor([0, 1], dtype=torch.long),
        categories=torch.tensor([0, 2], dtype=torch.long),
    )
    optimizer.zero_grad(set_to_none=True)
    losses.total.backward()
    optimizer.step()

    values = {
        "total": float(losses.total.detach()),
        "transfer": float(losses.transfer.detach()),
        "consistency": float(losses.consistency.detach()),
        "margin": float(losses.margin.detach()),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError("non-finite loss detected")
    print("QAD smoke test passed; all loss terms are finite.")


if __name__ == "__main__":
    main()
