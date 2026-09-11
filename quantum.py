from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.autograd.function import once_differentiable


class _ParameterShift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, angles: Tensor, rotations: Tensor, circuit) -> Tensor:
        ctx.save_for_backward(angles, rotations)
        ctx.circuit = circuit
        return circuit.expectations(angles, rotations)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        angles, rotations = ctx.saved_tensors
        circuit = ctx.circuit
        grad_angles = grad_rotations = None
        shift = math.pi / 2
        if ctx.needs_input_grad[0]:
            grad_angles = torch.zeros_like(angles)
            for wire in range(circuit.qubits):
                offset = torch.zeros_like(angles)
                offset[:, wire] = shift
                difference = circuit.expectations(angles + offset, rotations)
                difference -= circuit.expectations(angles - offset, rotations)
                grad_angles[:, wire] = (difference * grad_output).sum(-1) / 2
        if ctx.needs_input_grad[1]:
            grad_rotations = torch.zeros_like(rotations)
            for layer in range(circuit.layers):
                for wire in range(circuit.qubits):
                    for axis in range(2):
                        offset = torch.zeros_like(rotations)
                        offset[layer, wire, axis] = shift
                        difference = circuit.expectations(angles, rotations + offset)
                        difference -= circuit.expectations(angles, rotations - offset)
                        grad_rotations[layer, wire, axis] = (
                            difference * grad_output
                        ).sum() / 2
        return grad_angles, grad_rotations, None


class VariationalCircuit(nn.Module):
    def __init__(
        self,
        qubits: int = 5,
        layers: int = 2,
        entanglement: bool = True,
        trainable_rotations: bool = True,
    ):
        super().__init__()
        if qubits < 2 or layers < 1:
            raise ValueError("qubits must be at least 2 and layers must be positive")
        self.qubits = qubits
        self.layers = layers
        self.entanglement = entanglement
        self.rotations = nn.Parameter(
            torch.empty(layers, qubits, 2), requires_grad=trainable_rotations
        )
        nn.init.uniform_(self.rotations, -0.05, 0.05)
        indices = torch.arange(1 << qubits)
        masks = torch.arange(qubits - 1, -1, -1)
        bits = (indices[:, None] >> masks[None, :]) & 1
        self.register_buffer("z_signs", (1 - 2 * bits).float(), persistent=False)
        permutations = []
        for control in range(qubits):
            target = (control + 1) % qubits
            control_bit = (indices >> (qubits - control - 1)) & 1
            permutations.append(indices ^ (control_bit << (qubits - target - 1)))
        self.register_buffer(
            "cnot_indices", torch.stack(permutations), persistent=False
        )

    @staticmethod
    def _ry(state: Tensor, angle: Tensor, wire: int) -> Tensor:
        paired = state.reshape(state.shape[0], 1 << wire, 2, -1)
        zero, one = paired.unbind(2)
        cosine = torch.cos(angle / 2).reshape(-1, 1, 1)
        sine = torch.sin(angle / 2).reshape(-1, 1, 1)
        return torch.stack(
            (cosine * zero - sine * one, sine * zero + cosine * one), dim=2
        ).reshape_as(state)

    def expectations(self, angles: Tensor, rotations: Tensor) -> Tensor:
        dtype = torch.complex128 if angles.dtype == torch.float64 else torch.complex64
        state = torch.zeros(
            angles.shape[0], 1 << self.qubits, device=angles.device, dtype=dtype
        )
        state[:, 0] = 1
        signs = self.z_signs.to(dtype=angles.dtype)
        for wire in range(self.qubits):
            state = self._ry(state, angles[:, wire], wire)
        for layer in range(self.layers):
            for wire in range(self.qubits):
                phase = torch.exp(-0.5j * rotations[layer, wire, 0] * signs[:, wire])
                state = state * phase
                state = self._ry(state, rotations[layer, wire, 1], wire)
            if self.entanglement:
                for wire in range(self.qubits):
                    state = state.index_select(1, self.cnot_indices[wire])
        probabilities = state.real.square() + state.imag.square()
        return probabilities @ signs

    def forward(self, angles: Tensor) -> Tensor:
        if angles.ndim != 2 or angles.shape[1] != self.qubits:
            raise ValueError("angles must have shape (batch, qubits)")
        if angles.dtype not in (torch.float32, torch.float64):
            raise TypeError("angles must use float32 or float64")
        if angles.device != self.rotations.device or angles.dtype != self.rotations.dtype:
            raise ValueError("angles and circuit parameters must share device and dtype")
        return _ParameterShift.apply(angles, self.rotations, self)


class QuantumHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        classes: int = 3,
        qubits: int = 5,
        layers: int = 2,
        zeta: float = 5.0,
        entanglement: bool = True,
        trainable_rotations: bool = True,
    ):
        super().__init__()
        if input_dim < 1 or classes < 2 or not math.isfinite(zeta) or zeta <= 0:
            raise ValueError("invalid input dimension, class count, or zeta")
        self.zeta = zeta
        self.projection = nn.Linear(input_dim, qubits, bias=False)
        self.circuit = VariationalCircuit(
            qubits, layers, entanglement, trainable_rotations
        )
        self.readout = nn.Linear(qubits, classes)

    def forward(self, features: Tensor) -> Tensor:
        angles = 2 * math.pi * torch.tanh(self.projection(features) / self.zeta)
        return self.readout(self.circuit(angles))

    @torch.no_grad()
    def feature_lipschitz_bound(self) -> Tensor:
        projection_norm = torch.linalg.matrix_norm(self.projection.weight, ord=2)
        readout_norm = torch.linalg.matrix_norm(self.readout.weight, ord=2)
        return (
            2 * math.pi / self.zeta
            * self.circuit.qubits * projection_norm * readout_norm
        )


class MixedHead(nn.Module):
    def __init__(self, quantum: QuantumHead, classical: nn.Module, alpha: float = 0.5):
        super().__init__()
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must lie in [0, 1]")
        self.quantum = quantum
        self.classical = classical
        self.register_buffer("alpha", torch.tensor(float(alpha)))

    def forward(self, features: Tensor) -> Tensor:
        if self.alpha.item() == 1:
            return self.quantum(features)
        if self.alpha.item() == 0:
            return self.classical(features)
        return (1 - self.alpha) * self.classical(features) + self.alpha * self.quantum(features)


class TanhHead(nn.Sequential):
    def __init__(self, input_dim: int, classes: int = 3, hidden_dim: int = 5):
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, classes),
        )
