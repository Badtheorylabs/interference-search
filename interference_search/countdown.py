"""Countdown: problem generation, the move rules, an exact solver, and a checker for attempts.

A state is a sorted tuple of the numbers still available (inputs not yet used plus values built so
far). A move picks two numbers, applies + - * /, and replaces them with the result; intermediate
values must be positive integers. A problem is solved when exactly one number is left and it equals
the target, so every input is used once.

Every attempt a model writes while reasoning is an arithmetic expression, so attempts can be
extracted, evaluated exactly and canonicalized without an LLM judge.
"""
import ast
import random
import re
from collections import Counter
from fractions import Fraction
from functools import lru_cache

OPS = ["+", "-", "*", "/"]


def _combine(a, b, op):
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if b == 0:
        return None
    return a / b


def gen_problem(rng: random.Random, n_numbers=4):
    """Random solvable instance: numbers, target, one reference solution."""
    while True:
        nums = [rng.randint(1, 25) for _ in range(n_numbers - 1)] + [rng.randint(25, 100)]
        rng.shuffle(nums)
        items = [(Fraction(n), str(n)) for n in nums]
        ok = True
        while len(items) > 1:
            i, j = rng.sample(range(len(items)), 2)
            (va, ea), (vb, eb) = items[i], items[j]
            op = rng.choice(OPS)
            v = _combine(va, vb, op)
            if v is None or v.denominator != 1 or v <= 0:
                ok = False
                break
            items = [x for k, x in enumerate(items) if k not in (i, j)] + [(v, f"({ea} {op} {eb})")]
        if not ok:
            continue
        target, expr = items[0]
        target = int(target)
        if 10 <= target <= 999 and target not in nums:
            return {"numbers": nums, "target": target, "solution": expr[1:-1]}


def count_solutions(numbers, target):
    """Distinct canonical expressions using every number once that hit target."""
    sols = set()

    def rec(items):
        if len(items) == 1:
            v, e = items[0]
            if v == target:
                parsed = parse_expr(e)
                if parsed:
                    sols.add(parsed[1])
            return
        for i in range(len(items)):
            for j in range(len(items)):
                if i == j:
                    continue
                (va, ea), (vb, eb) = items[i], items[j]
                rest = [x for k, x in enumerate(items) if k not in (i, j)]
                for op in OPS:
                    if op in "+*" and i > j:
                        continue
                    v = _combine(va, vb, op)
                    if v is None:
                        continue
                    rec(rest + [(v, f"({ea} {op} {eb})")])

    rec([(Fraction(n), str(n)) for n in numbers])
    return len(sols)


def gen_hard_problem(rng: random.Random, max_solutions=2):
    while True:
        p = gen_problem(rng)
        nums, t = p["numbers"], p["target"]
        # reject obvious anchors: one pairwise op lands within 25 of the target
        pair_vals = [_combine(Fraction(a), Fraction(b), op)
                     for i, a in enumerate(nums) for j, b in enumerate(nums) if i != j for op in OPS]
        if any(v is not None and abs(v - t) <= 25 for v in pair_vals):
            continue
        n = count_solutions(nums, t)
        if 1 <= n <= max_solutions:
            p["n_solutions"] = n
            return p


def prompt_text(p):
    nums = ", ".join(map(str, p["numbers"]))
    return (
        f"Using the numbers [{nums}], create an equation that equals {p['target']}. "
        "You can use +, -, *, / and each number exactly once. "
        "Show your work, then give the final expression inside <answer></answer> tags, "
        "for example <answer>(1 + 2) / 3</answer>."
    )


# ---------- expression parsing ----------

# a period only counts as part of an expression when it is a decimal point
_EXPR_RE = re.compile(r"(?:[\d\s+\-*/×÷x()]|(?<=\d)\.(?=\d))+")


def _normalize(s):
    # every replacement keeps the string length, so offsets still index the raw text
    s = s.replace("×", "*").replace("÷", "/").replace("\\times", "*     ").replace("\\div", "/   ")
    s = re.sub(r"(?<=\d)\s*x\s*(?=[\d(])", lambda m: "*".ljust(len(m.group(0))), s)
    # word operators between numbers: "20 times 11", "220 plus 210", "390 divided by 30"
    for pat, op in ((r"multiplied by|times", "*"), (r"divided by|over", "/"),
                    (r"plus|added to", "+"), (r"minus|less", "-")):
        s = re.sub(rf"(?<=[\d)])\s+(?:{pat})\s+(?=[\d(])",
                   lambda m, op=op: (" " + op).ljust(len(m.group(0))), s)
    return s


class _Canon(ast.NodeVisitor):
    """Evaluate exactly and build a canonical string (commutative ops sorted, flattened)."""

    def __init__(self):
        self.leaves = []

    def run(self, node):
        return self._go(node)

    def _flat(self, node, optype):
        # returns list of (sign, subnode) for + / -   or (power, subnode) for * and /
        if isinstance(node, ast.BinOp):
            if optype == "add" and isinstance(node.op, (ast.Add, ast.Sub)):
                left = self._flat(node.left, "add")
                right = self._flat(node.right, "add")
                if isinstance(node.op, ast.Sub):
                    right = [(-s, n) for s, n in right]
                return left + right
            if optype == "mul" and isinstance(node.op, (ast.Mult, ast.Div)):
                left = self._flat(node.left, "mul")
                right = self._flat(node.right, "mul")
                if isinstance(node.op, ast.Div):
                    right = [(-s, n) for s, n in right]
                return left + right
        return [(1, node)]

    def _go(self, node):
        if isinstance(node, ast.Expression):
            return self._go(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            self.leaves.append(node.value)
            return Fraction(node.value), str(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v, s = self._go(node.operand)
            return -v, f"neg({s})"
        if isinstance(node, ast.BinOp):
            kind = "add" if isinstance(node.op, (ast.Add, ast.Sub)) else "mul"
            parts = self._flat(node, kind)
            vals, strs = [], []
            for sign, sub in parts:
                v, s = self._go(sub)
                if kind == "add":
                    vals.append(v if sign > 0 else -v)
                    strs.append(("+" if sign > 0 else "-") + s)
                else:
                    if sign < 0 and v == 0:
                        raise ZeroDivisionError
                    vals.append(v if sign > 0 else 1 / v)
                    strs.append(("*" if sign > 0 else "/") + s)
            if kind == "add":
                total = sum(vals, Fraction(0))
            else:
                total = Fraction(1)
                for v in vals:
                    total *= v
            return total, kind + "(" + ",".join(sorted(strs)) + ")"
        raise ValueError("unsupported")


def parse_expr(s):
    """Return (value, canonical, leaves) or None."""
    s = _normalize(s).strip()
    if not s or not re.search(r"\d", s) or not re.search(r"[+\-*/]", s):
        return None
    try:
        tree = ast.parse(s, mode="eval")
        c = _Canon()
        v, canon = c.run(tree)
        return v, canon, c.leaves
    except Exception:
        return None


def _is_sub_multiset(a, b):
    ca, cb = Counter(a), Counter(b)
    return all(cb[k] >= n for k, n in ca.items())


def extract_attempts(text, numbers, target):
    """Find full attempts (use every given number once) in a trace.

    Handles two styles:
      * whole expressions, e.g. "(25 - 7) * 3 + 50"
      * step chains, e.g. "25 * 3 = 75" then "75 + 50 = 125", where an
        intermediate result is substituted back by its expression.
    Returns list of dicts with char offset, canonical form, value, correct flag.
    """
    out = []
    step_exprs = {}  # intermediate value -> (expr string, leaves)
    for m in _EXPR_RE.finditer(_normalize(text)):
        raw = m.group(0)
        seg = raw.strip()
        if len(seg) < 3:
            continue
        # trim unbalanced parens at the edges
        while seg.count("(") > seg.count(")") and seg.startswith("("):
            seg = seg[1:].strip()
        while seg.count(")") > seg.count("(") and seg.endswith(")"):
            seg = seg[:-1].strip()
        parsed = parse_expr(seg)
        if parsed is None:
            continue
        val, canon, leaves = parsed
        # substitute intermediate results that are not original numbers
        if not _is_sub_multiset(leaves, numbers):
            expanded = seg
            for leaf in set(leaves):
                if leaf in step_exprs and Counter(leaves)[leaf] > Counter(numbers)[leaf]:
                    e, _ = step_exprs[leaf]
                    expanded = re.sub(rf"(?<![\d.]){leaf}(?![\d.])", f"({e})", expanded)
            p2 = parse_expr(expanded)
            if p2 is None or not _is_sub_multiset(p2[2], numbers):
                continue
            val, canon, leaves = p2
            seg = expanded
        if val.denominator == 1:
            step_exprs[int(val)] = (seg, leaves)
        if sorted(leaves) == sorted(numbers):
            out.append({
                "offset": m.start(),
                "expr": seg,
                "canon": canon,
                "value": float(val),
                "correct": val == target,
            })
    return out


def check_answer(text, numbers, target):
    m = re.findall(r"<answer>(.*?)</answer>", text, re.S)
    if not m:
        return False
    parsed = parse_expr(m[-1])
    if parsed is None:
        return False
    val, _, leaves = parsed
    return sorted(leaves) == sorted(numbers) and val == target


# ---------- move rules and exact solving ----------

def moves(state):
    """Every state reachable in one move (duplicates included, one per way of getting there)."""
    s = list(state)
    out = []
    for i in range(len(s)):
        for j in range(len(s)):
            if i == j:
                continue
            a, b = s[i], s[j]
            rest = [x for k, x in enumerate(s) if k not in (i, j)]
            results = []
            if i < j:                      # commutative operations once per unordered pair
                results += [a + b, a * b]
            if a > b:
                results.append(a - b)
            if b > 1 and a % b == 0:
                results.append(a // b)
            for r in results:
                out.append(tuple(sorted(rest + [r])))
    return out


def children(state):
    """Distinct states reachable in one move: what an environment that merges states returns."""
    return sorted(set(moves(state)))


def apply_move(state, a, op, b):
    """Apply `a op b` to a state. Returns the new sorted state, or None if the move is illegal."""
    s = list(state)
    if a not in s:
        return None
    s.remove(a)
    if b not in s:
        return None
    s.remove(b)
    op = {"x": "*", "×": "*", "÷": "/"}.get(op, op)
    if op == "+":
        r = a + b
    elif op == "*":
        r = a * b
    elif op == "-":
        r = a - b
    else:
        if b == 0 or a % b:
            return None
        r = a // b
    if r <= 0:
        return None
    return tuple(sorted(s + [r]))


def solver(target):
    """Exact solver for one target: solvable(state) -> True if the target is still reachable."""
    @lru_cache(maxsize=None)
    def solvable(state):
        if len(state) == 1:
            return state[0] == target
        return any(solvable(n) for n in moves(state))
    return solvable


def reachable_states(numbers):
    """Every non-terminal state reachable from the start."""
    seen, stack = set(), [tuple(sorted(numbers))]
    while stack:
        s = stack.pop()
        if s in seen or len(s) == 1:
            continue
        seen.add(s)
        stack.extend(moves(s))
    return seen
