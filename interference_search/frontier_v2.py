"""Frontier selection when an observed behaviour is not a proof of equivalence.

An exact key may safely suppress a repeat. A behaviour signature measured on a
finite test set is only evidence for diversity: different programs with that
signature can behave differently on unseen inputs or under later revisions.
The selector keeps representatives of those soft clusters without treating
them as one state.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Hashable, Iterable


@dataclass(frozen=True)
class Candidate:
    state: Any
    exact_key: Hashable
    signature: Hashable
    score: float
    support: int = 1


@dataclass(frozen=True)
class Selection:
    live: list[Candidate]
    exact_duplicates: int
    signature_collisions: int
    distinct_signatures: int


def select_frontier(
    candidates: Iterable[Candidate], width: int, first_pass_per_signature: int = 1
) -> Selection:
    """Select a level frontier with an initial diversity pass.

    Rank by judge score, then by support from distinct proposals. The first
    pass takes at most ``first_pass_per_signature`` candidates from each
    observed-behaviour group. Remaining capacity is filled by global rank, so
    candidates sharing the only available signature are not discarded.
    """
    if width < 1 or first_pass_per_signature < 1:
        raise ValueError("width and first_pass_per_signature must be positive")

    by_exact: dict[Hashable, Candidate] = {}
    exact_duplicates = 0
    for candidate in candidates:
        old = by_exact.get(candidate.exact_key)
        if old is None:
            by_exact[candidate.exact_key] = candidate
        else:
            exact_duplicates += 1
            if candidate.score > old.score:
                by_exact[candidate.exact_key] = Candidate(
                    candidate.state, candidate.exact_key, candidate.signature,
                    candidate.score, old.support + candidate.support,
                )
            else:
                by_exact[candidate.exact_key] = Candidate(
                    old.state, old.exact_key, old.signature,
                    old.score, old.support + candidate.support,
                )

    ranked = sorted(
        by_exact.values(),
        key=lambda candidate: (-candidate.score, -candidate.support, repr(candidate.exact_key)),
    )
    by_signature: dict[Hashable, list[Candidate]] = defaultdict(list)
    for candidate in ranked:
        by_signature[candidate.signature].append(candidate)

    chosen: list[Candidate] = []
    selected_keys: set[Hashable] = set()
    for depth in range(first_pass_per_signature):
        heads = [group[depth] for group in by_signature.values() if len(group) > depth]
        for candidate in sorted(
            heads,
            key=lambda item: (-item.score, -item.support, repr(item.exact_key)),
        ):
            if len(chosen) == width:
                break
            chosen.append(candidate)
            selected_keys.add(candidate.exact_key)
        if len(chosen) == width:
            break

    if len(chosen) < width:
        chosen.extend(candidate for candidate in ranked
                      if candidate.exact_key not in selected_keys)
        chosen = chosen[:width]

    return Selection(
        live=chosen,
        exact_duplicates=exact_duplicates,
        signature_collisions=len(ranked) - len(by_signature),
        distinct_signatures=len(by_signature),
    )
