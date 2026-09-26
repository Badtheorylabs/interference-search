"""Finite register programs with exact local counterexample certificates.

A program chooses one of two instructions at every slot. Each instruction
updates one or two registers. A failed output test identifies a register; all
programs that assign the same choices to its dependency scope have the same
output there. The verifier can therefore certify a whole wrong subspace
without inspecting every full program.

This is a controlled mechanism domain. The operations, dependencies and
verifier are known to the simulator; it is not an open-domain code result.
"""

from dataclasses import dataclass
from itertools import product
import random


@dataclass(frozen=True)
class Counterexample:
    register: int
    expected: int
    actual: int
    scope: tuple[int, ...]


class PublicRegisterGrammar:
    """Program operations visible to a search policy; target outputs are absent."""

    def __init__(self, registers: int, slots_per_register: int, modulus: int,
                 initial: tuple[int, ...], effects: tuple[tuple[int, ...], ...]):
        self.registers = registers
        self.slots_per_register = slots_per_register
        self.decisions = registers * slots_per_register
        self.modulus = modulus
        self.initial = initial
        self.effects = effects
        self._scopes = tuple(
            tuple(slot for slot, affected in enumerate(effects) if register in affected)
            for register in range(registers)
        )

    def scope(self, register: int) -> tuple[int, ...]:
        return self._scopes[register]

    def local_output(self, register: int, scope_values: tuple[int, ...]) -> int:
        scope = self.scope(register)
        if len(scope_values) != len(scope) or any(bit not in (0, 1) for bit in scope_values):
            raise ValueError("wrong local program shape")
        value = self.initial[register]
        for bit in scope_values:
            value = (value + 1) % self.modulus if bit == 0 else (2 * value + 1) % self.modulus
        return value

    def allowed_patterns(self, counterexample: Counterexample) -> tuple[tuple[int, ...], ...]:
        if counterexample.scope != self.scope(counterexample.register):
            raise ValueError("counterexample scope does not match grammar")
        return tuple(pattern for pattern in product((0, 1), repeat=len(counterexample.scope))
                     if self.local_output(counterexample.register, pattern) == counterexample.expected)

    def certify_forbidden(self, counterexample: Counterexample, pattern: tuple[int, ...]) -> bool:
        if counterexample.scope != self.scope(counterexample.register):
            raise ValueError("counterexample scope does not match grammar")
        return self.local_output(counterexample.register, pattern) != counterexample.expected


class RegisterProgram(PublicRegisterGrammar):
    def __init__(self, registers: int = 6, slots_per_register: int = 4,
                 topology: str = "independent", seed: int = 0, modulus: int = 31):
        if registers < 2 or slots_per_register < 1 or modulus < 3:
            raise ValueError("invalid register program size")
        if topology not in {"independent", "ring", "sparse"}:
            raise ValueError("unknown topology")
        self.topology = topology
        rng = random.Random(seed)
        initial = tuple(rng.randrange(1, modulus) for _ in range(registers))
        effects = []
        for slot in range(registers * slots_per_register):
            primary = slot // slots_per_register
            affected = [primary]
            if topology == "ring":
                affected.append((primary + 1) % registers)
            elif topology == "sparse" and rng.random() < 0.5:
                other = rng.randrange(registers - 1)
                if other >= primary:
                    other += 1
                affected.append(other)
            effects.append(tuple(affected))
        super().__init__(registers, slots_per_register, modulus, initial, tuple(effects))
        self.hidden = tuple(rng.randrange(2) for _ in range(self.decisions))
        self.expected = self.execute(self.hidden)

    def public_view(self) -> PublicRegisterGrammar:
        return PublicRegisterGrammar(self.registers, self.slots_per_register,
                                     self.modulus, self.initial, self.effects)

    def execute(self, choices: tuple[int, ...]) -> tuple[int, ...]:
        if len(choices) != self.decisions or any(bit not in (0, 1) for bit in choices):
            raise ValueError("program choices must be binary and complete")
        values = list(self.initial)
        for bit, affected in zip(choices, self.effects):
            for register in affected:
                values[register] = (values[register] + 1) % self.modulus if bit == 0 else (
                    2 * values[register] + 1
                ) % self.modulus
        return tuple(values)

    def verify(self, choices: tuple[int, ...]) -> Counterexample | None:
        actual = self.execute(choices)
        for register, (observed, expected) in enumerate(zip(actual, self.expected)):
            if observed != expected:
                return Counterexample(register, expected, observed, self.scope(register))
        return None
