# scripts/upload_model_artifact.py
import os
import wandb
from datetime import datetime
import argparse

def push_model_artifact(model_name, files, aliases=None, metadata=None, project=None, entity=None):
    aliases = aliases or ["latest"]
    metadata = metadata or {}
    # init run
    run = wandb.init(project=project, entity=entity, job_type="push-model", reinit=True)
    # add push timestamp
    metadata.setdefault("pushed_at", datetime.utcnow().isoformat() + "Z")
    artifact = wandb.Artifact(name=model_name, type="model", metadata=metadata)

    # Add files or directories to artifact
    for p in files:
        if os.path.isdir(p):
            artifact.add_dir(p)
        elif os.path.exists(p):
            artifact.add_file(p)
        else:
            print(f"WARNING: file not found: {p}")

    logged = run.log_artifact(artifact, aliases=aliases)
    logged.wait()  # wait for upload to finish
    print(f"Uploaded artifact {model_name} with aliases {aliases}")
    run.finish()
    return logged

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="artifact name, e.g. viktorifit-meal-model")
    parser.add_argument("--files", nargs="+", required=True, help="list of files or dirs to include")
    parser.add_argument("--alias", nargs="*", default=["latest"], help="aliases for this version")
    parser.add_argument("--project", default=os.getenv("WANDB_PROJECT"), help="wandb project")
    parser.add_argument("--entity", default=os.getenv("WANDB_ENTITY"), help="wandb entity")
    parser.add_argument("--meta", type=str, help="optional JSON string for metadata")
    args = parser.parse_args()

    meta = {}
    if args.meta:
        import json
        meta = json.loads(args.meta)

    push_model_artifact(
        model_name=args.name,
        files=args.files,
        aliases=args.alias,
        metadata=meta,
        project=args.project,
        entity=args.entity,
    )

if __name__ == "__main__":
    main()