#!/usr/bin/env python3
"""
Upload Models to W&B
* No timestamp aliases
* Let W&B handle internal versioning (v0 -> v1 -> v2)
* Always add "latest" alias (and optional "production")
* Wait for upload to finish
"""

import os
import wandb
from datetime import datetime
import glob

def push_model_artifact(
    model_name,
    files,
    project,
    entity,
    metadata=None,
    description=None,
    promote_to_production=False
):
    metadata = metadata or {}

    # Debug prints so we can see in CI logs
    print(">>> push_model_artifact called")
    print("  model_name:", model_name)
    print("  project:", project)
    print("  entity:", entity)
    print("  files to add:", files)
    print("  promote_to_production:", promote_to_production)

    run = wandb.init(project=project, entity=entity, job_type="upload-model", reinit=True)

    # Add upload timestamp to metadata (NOT as alias)
    metadata["uploaded_at"] = datetime.utcnow().isoformat() + "Z"

    artifact = wandb.Artifact(
        name=model_name,
        type="model",
        description=description or f"Model artifact for {model_name}",
        metadata=metadata
    )

    # Add files/dirs
    files_added = 0
    for file_path in files:
        if os.path.isdir(file_path):
            artifact.add_dir(file_path)
            # approximate count of files under dir (for debug)
            files_added += len(glob.glob(f"{file_path}/**/*", recursive=True))
        elif os.path.exists(file_path):
            artifact.add_file(file_path)
            files_added += 1
        else:
            print(f"⚠️  File not found: {file_path}")

    # aliases: let W&B handle numeric versions; add 'latest' and optional 'production'
    aliases = ["latest"]
    if promote_to_production:
        aliases.append("production")

    # Log artifact and wait for upload to complete
    logged = run.log_artifact(artifact, aliases=aliases)
    try:
        logged.wait()  # wait until upload completes
    except Exception as e:
        print("⚠️  Warning: waiting for artifact to finish raised:", e)

    run.finish()

    # Informative prints
    print(f"✅ Uploaded: {model_name}")
    print(f"   Aliases set: {aliases}")
    print(f"   Files (approx): {files_added}")
    print(f"   Metadata keys: {list(metadata.keys())}")
    print()

def upload_all_models_separately():
    project = os.environ.get("WANDB_PROJECT")
    entity = os.environ.get("WANDB_ENTITY")

    if not project or not entity:
        raise ValueError("WANDB_PROJECT and WANDB_ENTITY must be set in environment")

    print("="*70)
    print("UPLOADING MODELS TO W&B (SEPARATE ARTIFACTS)")
    print("="*70)
    print(f"Project: {project}")
    print(f"Entity: {entity}")
    print()

    # 1) MEAL
    print("📦 1/3: Meal Predictor")
    meal_path = "models/model_meal.pickle"
    if os.path.exists(meal_path):
        push_model_artifact(
            model_name="viktorifit-meal-model",
            files=[meal_path, "models/progress_encoders.pickle"] if os.path.exists("models/progress_encoders.pickle") else [meal_path],
            project=project, entity=entity,
            metadata={"model_type":"knn","purpose":"Meal recommendation"},
            description="KNN-based meal recommendation model",
            promote_to_production=False
        )
    else:
        print("⚠️  models/model_meal.pickle not found, skipping")

    # 2) WORKOUT
    print("📦 2/3: Workout Predictor")
    workout_path = "models/model_workout.pickle"
    if os.path.exists(workout_path):
        push_model_artifact(
            model_name="viktorifit-workout-model",
            files=[workout_path],
            project=project, entity=entity,
            metadata={"model_type":"weighted_knn","purpose":"Workout matching"},
            description="Weighted KNN for workout partner matching",
            promote_to_production=False
        )
    else:
        print("⚠️  models/model_workout.pickle not found, skipping")

    # 3) PROGRESS (ONNX folder)
    print("📦 3/3: Progress Predictor (ONNX)")
    onnx_dir = "models/onnx"
    if os.path.isdir(onnx_dir):
        onnx_files = glob.glob(f"{onnx_dir}/*.onnx")
        if onnx_files:
            push_model_artifact(
                model_name="viktorifit-progress-model",
                files=onnx_files,
                project=project, entity=entity,
                metadata={
                    "model_type":"xgboost_ensemble_onnx",
                    "n_models": len(onnx_files),
                    "targets": [os.path.basename(f).replace('.onnx','') for f in onnx_files],
                    "purpose":"Progress prediction"
                },
                description=f"XGBoost ensemble ({len(onnx_files)} ONNX models) for progress tracking",
                promote_to_production=False
            )
        else:
            print(f"⚠️  No ONNX files found in {onnx_dir}")
    else:
        print(f"⚠️  {onnx_dir} not found, skipping")

    print("="*70)
    print("✅ UPLOAD COMPLETE")
    print("="*70)
    print()
    print("View artifacts at:")
    print(f"https://wandb.ai/{entity}/{project}/artifacts")
    print()

if __name__ == "__main__":
    upload_all_models_separately()
