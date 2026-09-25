"""Code (MBPP) as an Interference Search domain, with Qwen3-1.7B proposing and the interpreter executing.

State    = (program, behaviour) where behaviour is what the program returns or raises on each test.
Execute  = run the program against every test in a subprocess with a timeout.
Merge    = two programs with identical behaviour on the tests are the same state.
Judge    = tests passed (execution is the judge; no learned model yet).
Propose  = from the empty state, write a fresh program; from a program, write a revision given only
           the current program and its test feedback (no chat history).
Cost     = generated tokens.

experiments/code/benchmark.py drives this class directly instead of through core.search, so that one
round of generation can be batched across all problems at once.
"""
import json
import re
import subprocess
import sys

RUNNER = r'''
import json, sys, signal, ast
src, tests = json.loads(sys.stdin.read())
out = []
def handler(*a): raise TimeoutError("timeout")
signal.signal(signal.SIGALRM, handler)
g = {}
try:
    signal.alarm(2); exec(src, g); signal.alarm(0)
except BaseException as e:
    print(json.dumps([["load", type(e).__name__ + ": " + str(e)[:120]]] * len(tests))); sys.exit()
for t in tests:
    try:
        node = ast.parse(t).body[0]
        signal.alarm(2)
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1 \
                and isinstance(node.test.ops[0], ast.Eq):
            left = eval(compile(ast.Expression(node.test.left), "<t>", "eval"), g)
            right = eval(compile(ast.Expression(node.test.comparators[0]), "<t>", "eval"), g)
            ok = left == right
            out.append(["pass" if ok else "fail", repr(left)[:120]])
        else:
            exec(t, g)
            out.append(["pass", ""])
        signal.alarm(0)
    except AssertionError:
        signal.alarm(0); out.append(["fail", "assertion"])
    except BaseException as e:
        signal.alarm(0); out.append(["error", type(e).__name__ + ": " + str(e)[:120]])
print(json.dumps(out))
'''


def execute(src, tests, timeout=10):
    try:
        r = subprocess.run([sys.executable, "-c", RUNNER], input=json.dumps([src, tests]), capture_output=True,
                           text=True, timeout=timeout)
        return [tuple(x) for x in json.loads(r.stdout.strip().splitlines()[-1])]
    except Exception as e:
        return [("error", "runner: " + type(e).__name__)] * len(tests)


def extract_code(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    if m:
        return m[0].strip()
    m = re.search(r"(def .*)", text, re.S)
    return m.group(1).strip() if m else text.strip()


class State:
    __slots__ = ("src", "behaviour", "passed")

    def __init__(self, src=None, behaviour=None):
        self.src, self.behaviour = src, behaviour
        self.passed = sum(b[0] == "pass" for b in behaviour) if behaviour else 0


def feedback(p, s):
    lines = []
    for t, (status, detail) in zip(p["test_list"], s.behaviour):
        if status == "pass":
            lines.append(f"PASS  {t}")
        elif status == "fail":
            lines.append(f"FAIL  {t}\n      your function returned {detail}")
        else:
            lines.append(f"ERROR {t}\n      {detail}")
    return "\n".join(lines)


class Code:
    def __init__(self, model, tok, k=2, fresh_k=4, max_tokens=320, temp=0.8):
        from mlx_lm.sample_utils import make_sampler
        self.model, self.tok, self.k, self.fresh_k, self.max_tokens = model, tok, k, fresh_k, max_tokens
        self.sampler = make_sampler(temp=temp, top_p=0.95, top_k=20)

    # prompts
    def fresh_prompt(self, p):
        msg = (f"{p['prompt']}\nYour function must pass this test:\n{p['test_list'][0]}\n"
               "Reply with only the Python code in one ```python block.")
        return self.tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True,
                                            tokenize=True, enable_thinking=False)

    def revise_prompt(self, p, s):
        msg = (f"{p['prompt']}\nCurrent code:\n```python\n{s.src}\n```\nTest results:\n{feedback(p, s)}\n"
               "Fix the code so every test passes. Reply with only the corrected Python code in one ```python block.")
        return self.tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True,
                                            tokenize=True, enable_thinking=False)

    def generate(self, prompts):
        from mlx_lm import batch_generate
        if not prompts:
            return [], 0
        r = batch_generate(self.model, self.tok, prompts, max_tokens=self.max_tokens, sampler=self.sampler,
                           verbose=False)
        return r.texts, sum(len(self.tok.encode(t)) for t in r.texts)

    # domain interface
    def start(self, p):
        return State()

    def expand_prompts(self, states, p):
        """(prompt, parent index) pairs for one round."""
        out = []
        for i, s in enumerate(states):
            n = self.fresh_k if s.src is None else self.k
            pr = self.fresh_prompt(p) if s.src is None else self.revise_prompt(p, s)
            out += [(pr, i)] * n
        return out

    def to_states(self, texts, p):
        res = []
        for t in texts:
            src = extract_code(t)
            res.append(State(src, execute(src, p["test_list"])))
        return res

    def key(self, s):
        return None if s.behaviour is None else tuple(s.behaviour)

    def judge(self, states, p):
        return [s.passed for s in states], 0

    def is_goal(self, s, p):
        return s.behaviour is not None and s.passed == len(p["test_list"])

    def is_dead(self, s, p):
        return False
