"""Index and copy historical negative results without inventing missing trial rows."""
import argparse
import shutil
from pathlib import Path

from experiments.submission.common import digest,write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo",required=True)
    ap.add_argument("--original",help="optional original research directory")
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    sources=[]
    for root,label in ((Path(args.repo),"release"),(Path(args.original),"original") if args.original else (None,None)):
        if root is None:
            continue
        for folder in ("results/llm","results/invalid","results/code","results/toy","results/frontier","runs","toy/runs2"):
            path=root/folder
            if not path.exists():
                continue
            for source in sorted(path.rglob("*")):
                if source.is_file() and source.suffix in (".json",".jsonl",".log",".txt"):
                    relative=Path(label)/source.relative_to(root)
                    dest=out/relative
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(source,dest)
                    sources.append({"path":str(relative),"sha256":digest(dest),"bytes":dest.stat().st_size})
        for name in ("docs/EXPERIMENT_DESIGN.md","docs/RESEARCH_LOG.md","EXPERIMENT_DESIGN.md","REVIEW_BRIEF.md"):
            source=root/name
            if source.exists():
                relative=Path(label)/name
                dest=out/relative
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,dest)
                sources.append({"path":str(relative),"sha256":digest(dest),"bytes":dest.stat().st_size})
    write_json(out/"index.json",{"files":sources,"policy":"unchanged originals; failed/interrupted experiments retained",
               "limitations":"historical MBPP aggregate 9/30 versus 8/30 has no recoverable per-problem outcomes; no paired statistics are manufactured",
               "experiments":["textual refutation ledger","self rewind","cross rewind","explicit state prompting","cross-stream visibility and signed/unsigned toy studies","MLX OOM and interrupted runs"]})
    print(f"archived {len(sources)} historical files",flush=True)


if __name__=="__main__":
    main()
