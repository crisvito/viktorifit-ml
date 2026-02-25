import os
import wandb
from datetime import datetime

def get_next_version(entity, project, artifact_name):
    api = wandb.Api()
    try:
        # get all versions
        collection = api.artifact_collection(type_name="model", name=f"{entity}/{project}/{artifact_name}")
        versions = collection.artifacts()
        if not versions:
            return 1
        latest_version = max(int(a.version[1:]) for a in versions)
        return latest_version + 1
    except:
        # artifact belum ada
        return 1

def push_model_artifact_auto(model_name, files, project, entity, metadata=None):
    metadata = metadata or {}

    run = wandb.init(project=project, entity=entity, job_type="push-model", reinit=True)

    # 🔥 Auto version
    next_version = get_next_version(entity, project, model_name)
    version_alias = f"v{next_version}"

    # 🔥 Timestamp alias
    timestamp_alias = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    metadata["auto_version"] = version_alias
    metadata["timestamp"] = timestamp_alias

    artifact = wandb.Artifact(
        name=model_name,
        type="model",
        metadata=metadata
    )

    for f in files:
        if os.path.isdir(f):
            artifact.add_dir(f)
        else:
            artifact.add_file(f)

    run.log_artifact(
        artifact,
        aliases=[version_alias, timestamp_alias, "latest"]
    )

    run.finish()

    print(f"Uploaded {model_name}")
    print(f"Version: {version_alias}")
    print(f"Timestamp: {timestamp_alias}")

if __name__ == "__main__":
    push_model_artifact_auto(
        model_name="viktorifit-model",
        files=["models"], 
        project=os.environ.get("WANDB_PROJECT"),
        entity=os.environ.get("WANDB_ENTITY"),
        metadata={"framework": "sklearn"}
    )