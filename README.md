# Edit-Aware Quantum-Classical Distillation for Robust Financial Text Classification

Reference implementation of the core components used by **Edit-aware quantum-classical distillation (QAD)** for three-way financial text classification.

This repository currently contains the method-level implementation. It does **not** include the paper's datasets, fixed train/validation/test splits, trained checkpoints, or the records required to reproduce the reported tables. The command below is therefore an interface smoke test, not a reproduction of a reported experiment.

## What is included

| Paper component | Implementation |
|---|---|
| Teacher supervision / distillation | `objectives.py`: `reverse_kl`, `QADObjective` |
| Local logit consistency under valid edits | `objectives.py`: `edit_consistency`; `edits.py`: `NeighborhoodSampler` |
| Class-margin regularization | `objectives.py`: `class_margin`, `QADObjective` |
| Hybrid quantum-classical student head | `quantum.py`: `VariationalCircuit`, `QuantumHead`, `MixedHead` |
| Financial-text encoders and classifier wrapper | `encoders.py`: FinBERT, TF-IDF/SVD, recurrent and HAN encoders, `TextClassifier` |
| Deterministic training and validation selection | `training.py`: `TrainingConfig`, `Trainer`, `RUN_SEEDS` |
| Five unit-edit categories | `edits.py`: insertion, deletion, substitution, swap and duplication |
| Validity screening and neighborhood construction | `edits.py`: `SemanticScreen`, `EditValidator`, `NeighborhoodSampler` |
| Budgeted adversarial evaluation | `attacks.py`: `BeamSearchAttack`, `summarize_attacks` |
| Margin-sensitivity utilities | `objectives.py`: `sampled_sensitivity`, `sampled_radius`, `conditional_radius` |

The canonical class order is `negative`, `neutral`, `positive` (`encoders.CANONICAL_CLASSES`). Teacher and student logits must use this same order.

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch installation can be platform-specific. If the command above does not select the desired CPU/CUDA build, install PyTorch using the official selector first, then install the remaining requirements.

`transformers` and `sentence-transformers` are needed by the FinBERT and semantic-screening paths. The dependency-free core quantum simulator is implemented directly with PyTorch; PennyLane or Qiskit is not required.

## Minimal run

Run the self-contained smoke test from the repository root:

```bash
python run_minimal.py --seed 17
```

It creates a small in-memory student head, evaluates all QAD loss terms, performs one optimizer step, and checks that the output is finite. It does not download a dataset, train a paper model, or print a claimed benchmark result.

## Using the training API

`training.Trainer` expects:

- a student accepting `Sequence[str]` and returning `(batch, classes)` logits;
- a frozen teacher with the same declared `class_order` when `beta > 0`;
- an `EditValidator` and `NeighborhoodSampler` when edit consistency is enabled;
- disjoint training and validation text sequences with integer labels in canonical class order.

The validity layer deliberately receives two project-specific predicates:

```python
EditValidator(
    financial_validator=...,
    sentiment_validator=...,
    screen=...,
    revision="your-validator-revision",
)
```

Those predicates are not defined in this repository. Returning `None` means unresolved validity; unresolved candidates cannot be silently counted as completed attacks by `summarize_attacks`.

## Reproduction status

The following items are still required for a complete public reproduction package:

1. Dataset acquisition instructions and exact dataset versions/identifiers.
2. Fixed sample-level train/validation/test split manifests and label mappings for every task.
3. Data preparation scripts, including deduplication and normalization decisions.
4. Concrete financial-equivalence and sentiment-preservation validator implementations, model/checkpoint identifiers, thresholds, and revision records.
5. Teacher checkpoint provenance and student configuration files for each encoder/classifier family.
6. Per-run hyperparameters, the stated tuning-budget protocol, seeds, hardware/software metadata, and checkpoint selection commands.
7. End-to-end train, clean-evaluation, attack-evaluation, and aggregation commands.
8. Machine-readable per-sample predictions/attack traces and scripts that regenerate the manuscript tables and figures.
9. Trained checkpoints or exact checkpoint-reconstruction instructions.
10. A repository license and citation metadata chosen by the authors.

Until these artifacts are added, the repository supports inspection and extension of the core method but should not be described as reproducing the numerical results in the paper.

## Metric semantics

`summarize_attacks` reports:

- clean accuracy over all evaluated inputs;
- attack success rate only over clean-correct inputs with completed, fully resolved attacks;
- full-test survival as clean-correct, non-successful attacks divided by all evaluated inputs.

The implementation rejects incomplete or unresolved attack records when computing ASR, preventing unresolved validity decisions from being treated as failures.

## Repository layout

```text
attacks.py       budgeted beam-search attacks and aggregate metrics
edits.py         edit generation, validity screening and neighborhood sampling
encoders.py      text encoders, teacher and classifier wrapper
objectives.py    QAD losses and margin-sensitivity analysis
quantum.py       differentiable variational circuit and hybrid heads
training.py      deterministic training loop and validation selection
run_minimal.py   dependency/interface smoke test
requirements.txt runtime dependencies
```

## Responsible reporting

Please report only results produced from disclosed data splits, configurations, validators, and checkpoints. The smoke-test output is a software sanity check and must not be cited as an experimental result.
