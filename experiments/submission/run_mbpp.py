"""Spark-only MBPP with strict token caps, fixed problems, raw prompts and outputs."""
import argparse
import ast
import hashlib
import json
import random
import time
from pathlib import Path

import torch

from interference_search.program_domain import extract_code
from experiments.submission.common import append, digest, environment, read_rows, write_json
from experiments.submission.spark_qwen import SparkQwen
from experiments.submission.spark_sandbox import execute, preflight


def fresh(problem):
    return (f"{problem['prompt']}\nYour function must pass this test:\n{problem['test_list'][0]}\n"
            "Reply with only the Python code in one ```python block.")


def revision(problem,state):
    feedback="\n".join(f"{status.upper()} {test}\n{detail}" for test,(status,detail) in zip(problem["test_list"],state["tests"]))
    return (f"{problem['prompt']}\nCurrent code:\n```python\n{state['source']}\n```\nTest results:\n{feedback}\n"
            "Fix the code so every test passes. Reply with only the corrected Python code in one ```python block.")


def key(state,legacy=False):
    behaviour=tuple(tuple(v) for v in state["tests"])
    if legacy:
        return behaviour
    try:
        source=ast.dump(ast.parse(state["source"]),include_attributes=False)
    except SyntaxError:
        source=state["source"]
    return source,behaviour


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True)
    ap.add_argument("--model",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--n",type=int,default=200)
    ap.add_argument("--seeds",default="0,1,2,3,4")
    ap.add_argument("--budgets",default="1500,4000")
    ap.add_argument("--budget-unit",choices=("total_tokens","generated_tokens"),default="total_tokens")
    ap.add_argument("--arms",default="independent_sampling,transcript,linear_state,frontier_no_merge,interference_search,legacy_behaviour_merge")
    ap.add_argument("--width",type=int,default=3)
    ap.add_argument("--batch-size",type=int,default=8)
    ap.add_argument("--max-new-tokens",type=int,default=320)
    ap.add_argument("--temperature",type=float,default=0.8)
    ap.add_argument("--wall-hours",type=float,default=24)
    args=ap.parse_args()
    torch.set_num_threads(2)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    all_data=json.loads(Path(args.data).read_text())
    ids=sorted(range(len(all_data)),key=lambda i:hashlib.sha256(f"mbpp-submission-20261003:{all_data[i]['task_id']}".encode()).hexdigest())[:args.n]
    problems=[all_data[i] for i in ids]
    write_json(out/"problems.json",problems)
    budgets=list(map(int,args.budgets.split(",")))
    seeds=list(map(int,args.seeds.split(",")))
    arms=args.arms.split(",")
    config={"problem_ids":[p["task_id"] for p in problems],"data_hash":digest(args.data),
            "seeds":seeds,"budgets":budgets,"budget_unit":args.budget_unit,"arms":arms,
            "width":args.width,"temperature":args.temperature,"top_p":0.95,"top_k":20,
            "enable_thinking":False,"max_new_tokens":args.max_new_tokens,"batch_size":args.batch_size,
            "selection":"fixed SHA256 order, no screening on model success",
            "source_hashes":{name:digest(Path(__file__).with_name(name)) for name in
                ("run_mbpp.py","spark_qwen.py","spark_sandbox.py")}}
    if (out/"config.json").exists() and json.loads((out/"config.json").read_text())!=config:
        raise ValueError("changed immutable MBPP configuration")
    write_json(out/"config.json",config)
    write_json(out/"environment.json",environment())
    write_json(out/"sandbox_preflight.json",preflight())
    backend=SparkQwen(args.model,out/"model_receipt.json")
    done={(r["problem_id"],r["method"],r["seed"],r["budget"]) for r in read_rows(out/"raw.jsonl")}
    started=time.perf_counter()
    for budget in budgets:
        for seed in seeds:
            for arm in arms:
                indices=[i for i,p in enumerate(problems) if (p["task_id"],arm,seed,budget) not in done]
                states={i:{"live":[None],"seen":set(),"chat":None,"generated_tokens":0,"prompt_tokens":0,
                           "sequential_rounds":0,"states_proposed":0,"states_merged":0,"states_judged":0,
                           "states_pruned":0,"states_expanded":0,"solved":False,"budget_exhausted":False,
                           "batch_wall_seconds_allocated":0.0,"execution_wall_seconds":0.0,
                           "round_trace":[],"solution":None} for i in indices}
                round_no=0
                trial_started=time.perf_counter()
                while True:
                    active=[i for i in indices if not states[i]["solved"] and not states[i]["budget_exhausted"]]
                    if not active:
                        break
                    if time.perf_counter()-started>args.wall_hours*3600:
                        write_json(out/"paused_at_wall_cap.json",{"wall_hours":args.wall_hours,"elapsed":time.perf_counter()-started,
                                   "completed_trials":len(done),"reason":"resume exact config to continue; incomplete trials retained in candidate outputs"})
                        return
                    round_no+=1
                    jobs=[]
                    for i in active:
                        p,s=problems[i],states[i]
                        s["sequential_rounds"]+=1
                        parents=[None] if arm=="independent_sampling" else s["live"]
                        if arm=="transcript":
                            if s["chat"] is None:
                                s["chat"]=[{"role":"user","content":fresh(p)}]
                            parents=[None]
                        reserved=0
                        for parent_index,parent in enumerate(parents):
                            s["states_expanded"]+=1
                            messages=s["chat"] if arm=="transcript" else [{"role":"user","content":fresh(p) if parent is None else revision(p,parent)}]
                            prompt=backend.prompt(messages)
                            prompt_tokens=len(backend.tokenizer.encode(prompt,add_special_tokens=False))
                            used=s["generated_tokens"]+(s["prompt_tokens"] if args.budget_unit=="total_tokens" else 0)
                            n=4 if arm=="independent_sampling" or (parent is None and arm not in ("transcript","linear_state")) else 1 if arm in ("transcript","linear_state") else 2
                            for _ in range(n):
                                read_cost=prompt_tokens if args.budget_unit=="total_tokens" else 0
                                cap=min(args.max_new_tokens,budget-used-reserved-read_cost)
                                if cap<=0:
                                    break
                                # Reserve every concurrent request's maximum completion
                                # and prompt. Actual shorter completions free budget only
                                # for the next round.
                                jobs.append((i,parent_index,prompt,cap))
                                reserved+=cap+read_cost
                        if reserved==0:
                            s["budget_exhausted"]=True
                    if not jobs:
                        continue
                    pools={i:[] for i in active}
                    before={i:(states[i]["states_proposed"],states[i]["states_merged"]) for i in active}
                    for cap in sorted({job[3] for job in jobs},reverse=True):
                        cap_jobs=[job for job in jobs if job[3]==cap]
                        for begin in range(0,len(cap_jobs),args.batch_size):
                            batch=cap_jobs[begin:begin+args.batch_size]
                            generation_seed=seed*10000000+round_no*10000+cap*10+begin
                            outputs=backend.generate([j[2] for j in batch],cap,generation_seed,args.temperature)
                            for job,output in zip(batch,outputs):
                                i,parent_index,prompt,_=job
                                p,s=problems[i],states[i]
                                s["generated_tokens"]+=output["generated_tokens"]
                                s["prompt_tokens"]+=output["prompt_tokens"]
                                assert s["generated_tokens"]+(s["prompt_tokens"] if args.budget_unit=="total_tokens" else 0)<=budget
                                s["batch_wall_seconds_allocated"]+=output["batch_wall_seconds"]/output["batch_size"]
                                code=extract_code(output["text"])
                                executed=time.perf_counter()
                                tests=execute(code,p["test_list"],p.get("test_imports",[]))
                                execution_seconds=time.perf_counter()-executed
                                s["execution_wall_seconds"]+=execution_seconds
                                s["states_proposed"]+=1
                                s["states_judged"]+=1
                                candidate={"source":code,"tests":tests,"passed":sum(t[0]=="pass" for t in tests)}
                                candidate_key=key(candidate,arm=="legacy_behaviour_merge")
                                if arm in ("interference_search","legacy_behaviour_merge") and candidate_key in s["seen"]:
                                    s["states_merged"]+=1
                                else:
                                    pools[i].append(candidate)
                                    s["seen"].add(candidate_key)
                                if candidate["passed"]==len(p["test_list"]):
                                    s["solved"]=True
                                    s["solution"]=code
                                if arm=="transcript":
                                    s["chat"] += [{"role":"assistant","content":output["text"]},
                                                  {"role":"user","content":revision(p,candidate)}]
                                append(out/"candidates.jsonl",{**output,"problem_id":p["task_id"],"method":arm,"seed":seed,
                                       "budget":budget,"round":round_no,"generation_seed":generation_seed,"prompt":prompt,
                                       "parent":parent_index,"code":code,"tests":tests,"execution_wall_seconds":execution_seconds,
                                       "total_generated_tokens":s["generated_tokens"],"total_prompt_tokens":s["prompt_tokens"]})
                    for i in active:
                        s=states[i]
                        pool=pools.get(i,[])
                        if arm=="linear_state":
                            if pool:
                                s["live"]=[pool[-1]]
                        elif arm in ("interference_search","legacy_behaviour_merge","frontier_no_merge"):
                            pool.sort(key=lambda v:-v["passed"])
                            s["live"]=pool[:args.width] or [None]
                        raw_count=s["states_proposed"]-before[i][0]
                        merged_count=s["states_merged"]-before[i][1]
                        s["round_trace"].append({"round":round_no,"raw_candidates":raw_count,
                                                 "unique_after_merging":raw_count-merged_count,
                                                 "frontier":len(s["live"]),"merged":merged_count})
                    print(f"MBPP {arm} seed={seed} budget={budget} round={round_no}: solved {sum(s['solved'] for s in states.values())}/{len(states)}",flush=True)
                for i in indices:
                    s=states[i]
                    row={k:v for k,v in s.items() if k not in ("live","seen","chat")}
                    row.update(problem_id=problems[i]["task_id"],method=arm,seed=seed,budget=budget,
                               budget_unit=args.budget_unit,wall_seconds=s["batch_wall_seconds_allocated"]+s["execution_wall_seconds"],
                               timing_scope="batch generation time divided by batch size plus execution; suite wall recorded separately",
                               failure_reason=None if s["solved"] else "token_budget",model_hash=backend.receipt["source_safetensors_sha256"])
                    append(out/"raw.jsonl",row)
                    done.add((problems[i]["task_id"],arm,seed,budget))
                append(out/"suite_timings.jsonl",{"method":arm,"seed":seed,"budget":budget,"wall_seconds":time.perf_counter()-trial_started})
    write_json(out/"complete.json",{"expected_rows":len(problems)*len(arms)*len(seeds)*len(budgets),"actual_rows":len(done),"wall_seconds":time.perf_counter()-started})


if __name__=="__main__":
    main()
