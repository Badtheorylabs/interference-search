"""Five independent initialization/training seeds on the frozen original 4/5 sources."""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn

from interference_search.countdown import reachable_states,solver
from interference_search.judge import Judge,encode
from experiments.submission.common import append,digest,environment,write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--seeds",default="0,1,2,3,4")
    ap.add_argument("--epochs",type=int,default=6)
    args=ap.parse_args()
    torch.set_num_threads(2)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path(args.manifest).read_text())
    sources=manifest["training_sources"]
    assert {len(p["numbers"]) for p in sources}=={4,5}
    config={"manifest_hash":digest(args.manifest),"source_ids":[p["problem_id"] for p in sources],
            "seeds":list(map(int,args.seeds.split(","))),"epochs":args.epochs,
            "optimizer":"Adam","lr":0.001,"batch_size":512,"shuffle_seed":1,
            "model_selection":"fixed final epoch, no test-set selection","source_hash":digest(__file__)}
    if (out/"config.json").exists() and json.loads((out/"config.json").read_text())!=config:
        raise ValueError("changed frozen training config")
    write_json(out/"config.json",config)
    write_json(out/"environment.json",environment())
    cache=out/"training_tensors.pt"
    if not cache.exists():
        rows=[]
        for i,p in enumerate(sources):
            alive=solver(p["target"])
            rows += [(s,p["target"],float(alive(s))) for s in sorted(reachable_states(p["numbers"]))]
            if (i+1)%100==0:
                print(f"training corpus {i+1}/{len(sources)}, {len(rows)} labelled states",flush=True)
        random.Random(1).shuffle(rows)
        X,M=encode([r[0] for r in rows],[r[1] for r in rows])
        Y=torch.tensor([r[2] for r in rows])
        torch.save({"X":X,"M":M,"Y":Y},cache)
        write_json(out/"training_data_receipt.json",{"sources":len(sources),"states":len(rows),
                   "alive":int(Y.sum()),"sizes":[4,5],"tensor_sha256":digest(cache),
                   "order":"sorted exact states per source, then fixed seed-1 shuffle; differs from historical unsorted state iteration"})
    data=torch.load(cache,map_location="cpu",weights_only=True)
    X,M,Y=data["X"],data["M"],data["Y"]
    positive=Y.sum()
    pos_weight=(len(Y)-positive)/positive.clamp(min=1)
    for seed in config["seeds"]:
        destination=out/f"judge_s{seed}.pt"
        if destination.exists() and destination.with_suffix(".receipt.json").exists():
            continue
        torch.manual_seed(seed)
        model=Judge()
        optimizer=torch.optim.Adam(model.parameters(),lr=config["lr"])
        started=time.perf_counter()
        for epoch in range(args.epochs):
            total=0.0
            model.train()
            for begin in range(0,len(Y),512):
                logits=model(X[begin:begin+512],M[begin:begin+512])
                loss=nn.functional.binary_cross_entropy_with_logits(logits,Y[begin:begin+512],pos_weight=pos_weight)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total+=loss.item()*len(logits)
            append(out/"training.jsonl",{"seed":seed,"epoch":epoch,"loss":total/len(Y),"wall_seconds":time.perf_counter()-started})
            print(f"judge seed {seed} epoch {epoch+1}/{args.epochs} loss {total/len(Y):.5f}; {time.perf_counter()-started:.0f}s",flush=True)
        model.eval()
        torch.save(model.state_dict(),destination)
        write_json(destination.with_suffix(".receipt.json"),{"seed":seed,"weights_sha256":digest(destination),
                   "epochs":args.epochs,"training_tensor_sha256":digest(cache),"sizes":[4,5],"wall_seconds":time.perf_counter()-started})
    write_json(out/"complete.json",{"seeds":config["seeds"],"epochs":args.epochs})


if __name__=="__main__":
    main()
