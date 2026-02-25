import os
import wandb
from datetime import datetime

def get_next_version(entity, project, artifact_name):
    """
    Retrieves the next available version number for a Weights & Biases (W&B) artifact.

    This function queries the W&B API to find the latest version of a specified 
    model artifact and calculates the next incremental integer.

    Args:
        entity (str): The W&B entity (username or team name).
        project (str): The target W&B project name.
        artifact_name (str): The name of the artifact collection to query.

    Returns:
        int: The next incremental version number. Returns 1 if the artifact
             does not exist or if an API error occurs.
    """
    api = wandb.Api()
    try:
        # Retrieve the collection of artifacts matching the specified name
        collection = api.artifact_collection(type_name="model", name=f"{entity}/{project}/{artifact_name}")
        versions = collection.artifacts()
        
        if not versions:
            return 1
            
        # Extract the integer value from the latest version tag (e.g., 'v3' -> 3)
        latest_version = max(int(a.version[1:]) for a in versions)
        return latest_version + 1
        
    except:
        # Fallback: If the artifact collection doesn't exist yet, initialize as version 1
        return 1

def push_model_artifact_auto(model_name, files, project, entity, metadata=None):
    """
    Automatically versions and uploads machine learning model artifacts to W&B.

    This function initializes a W&B run, creates a model artifact with auto-incrementing
    versioning and timestamp aliases, and uploads the specified files/directories.

    Args:
        model_name (str): The base name for the model artifact.
        files (list): A list of file paths or directory paths to be included in the artifact.
        project (str): The target W&B project name.
        entity (str): The W&B entity (username or team name).
        metadata (dict, optional): Additional metadata dictionary to attach to the artifact. Defaults to None.
    """
    metadata = metadata or {}

    # Initialize a new W&B run specifically for logging the artifact
    run = wandb.init(project=project, entity=entity, job_type="push-model", reinit=True)

    # Generate an auto-incrementing version alias
    next_version = get_next_version(entity, project, model_name)
    version_alias = f"v{next_version}"

    # Generate a UTC timestamp alias for precise tracking
    timestamp_alias = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    # Append tracking aliases to the artifact's metadata
    metadata["auto_version"] = version_alias
    metadata["timestamp"] = timestamp_alias

    # Initialize the W&B Artifact object
    artifact = wandb.Artifact(
        name=model_name,
        type="model",
        metadata=metadata
    )

    # Iterate through the provided paths and append them to the artifact
    for f in files:
        if os.path.isdir(f):
            artifact.add_dir(f)
        else:
            artifact.add_file(f)

    # Upload the artifact to W&B with multiple identifiable tags
    run.log_artifact(
        artifact,
        aliases=[version_alias, timestamp_alias, "latest"]
    )

    # Gracefully terminate the W&B run
    run.finish()

    # Output execution summary to the console
    print(f"Uploaded {model_name}")
    print(f"Version: {version_alias}")
    print(f"Timestamp: {timestamp_alias}")

if __name__ == "__main__":
    # Execute the artifact push for the Viktorifit project
    push_model_artifact_auto(
        model_name="viktorifit-model",
        files=["models"], 
        project=os.environ.get("WANDB_PROJECT"),
        entity=os.environ.get("WANDB_ENTITY"),
        metadata={"framework": "sklearn"}
    )