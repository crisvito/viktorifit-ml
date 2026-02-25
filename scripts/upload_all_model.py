"""
Viktorifit Model Registry Tool
Automated script to push Model Artifacts to Weights & Biases (W&B).
Features:
- Automated versioning (v0, v1, ...)
- GMT+7 (WIB) Timestamp Aliasing
- Metadata tracking
- Dedicated artifacts for Meal, Workout, and Progress models.
"""

import os
import wandb
import glob
from datetime import datetime, timedelta, timezone

def push_model_artifact(
    model_name,
    files,
    project,
    entity,
    metadata=None,
    description=None,
    promote_to_production=False
):
    """
    Push a set of files to W&B as a Model Artifact with GMT+7 timestamping.

    Args:
        model_name (str): The name of the artifact in W&B.
        files (list): List of file paths or directories to include.
        project (str): W&B project name.
        entity (str): W&B entity/username.
        metadata (dict, optional): Contextual data about the model.
        description (str, optional): Brief summary of the artifact.
        promote_to_production (bool): If True, adds the 'production' alias.
    """
    metadata = metadata or {}

    # Logging debug information for CI/CD environment
    print(f">>> Initializing upload for: {model_name}")
    print(f"    Target: {entity}/{project}")

    # Initialize W&B run for the upload task
    run = wandb.init(project=project, entity=entity, job_type="upload-model", reinit=True)

    # --- GMT+7 (WIB) TIMESTAMP LOGIC ---
    # Define Jakarta timezone (UTC+7)
    wib_tz = timezone(timedelta(hours=7))
    now_wib = datetime.now(wib_tz)
    
    # Format: YYYYMMDD-HHMMSS (Standard for easy sorting in W&B)
    timestamp_alias = now_wib.strftime("%Y%m%d-%H%M%S")
    iso_timestamp = now_wib.isoformat()

    # Add timestamp to metadata for traceability
    metadata["uploaded_at_wib"] = iso_timestamp

    # Initialize the Artifact object
    artifact = wandb.Artifact(
        name=model_name,
        type="model",
        description=description or f"Model artifact for {model_name}",
        metadata=metadata
    )

    # Process and add files/directories to the artifact
    files_added = 0
    for file_path in files:
        if os.path.isdir(file_path):
            artifact.add_dir(file_path)
            files_added += len(glob.glob(f"{file_path}/**/*", recursive=True))
        elif os.path.exists(file_path):
            artifact.add_file(file_path)
            files_added += 1
        else:
            print(f"⚠️  Warning: File not found and skipped: {file_path}")

    # --- ALIAS MANAGEMENT ---
    # 'latest' always points to the most recent upload
    # timestamp_alias allows us to track versions by WIB time
    aliases = ["latest", f"wib-{timestamp_alias}"]
    
    if promote_to_production:
        aliases.append("production")

    # Log the artifact and wait for the cloud sync to complete
    logged_artifact = run.log_artifact(artifact, aliases=aliases)
    
    try:
        # wait() ensures the script doesn't exit before the upload is finished
        logged_artifact.wait()
    except Exception as e:
        print(f"⚠️  Upload sync warning: {e}")

    # Close the W&B run
    run.finish()

    print(f"✅ Success: {model_name}")
    print(f"    Version: {logged_artifact.version}")
    print(f"    WIB Alias: wib-{timestamp_alias}")
    print(f"    Files: {files_added}")
    print("-" * 30)

def upload_all_models_separately():
    """
    Orchestrates the separate upload of Meal, Workout, and Progress models.
    Ensures each model type is stored in its own dedicated artifact collection.
    """
    project = os.environ.get("WANDB_PROJECT")
    entity = os.environ.get("WANDB_ENTITY")

    if not project or not entity:
        raise ValueError("Environment variables WANDB_PROJECT and WANDB_ENTITY are required.")

    print("="*70)
    print("VIKTORIFIT MLOPS: INDIVIDUAL MODEL UPLOAD (GMT+7)")
    print("="*70)

    # 1. MEAL MODEL UPLOAD
    print("📦 [1/3] Processing Meal Predictor...")
    meal_path = "models/model_meal.pickle"
    if os.path.exists(meal_path):
        # Encoders are often required alongside the model
        model_files = [meal_path]
        if os.path.exists("models/progress_encoders.pickle"):
            model_files.append("models/progress_encoders.pickle")
            
        push_model_artifact(
            model_name="viktorifit-meal-model",
            files=model_files,
            project=project, entity=entity,
            metadata={"model_arch": "knn", "domain": "nutrition"},
            description="KNN model for meal recommendations",
        )
    else:
        print("⏭️  Skipping Meal Model: File not found.")

    # 2. WORKOUT MODEL UPLOAD
    print("📦 [2/3] Processing Workout Predictor...")
    workout_path = "models/model_workout.pickle"
    if os.path.exists(workout_path):
        push_model_artifact(
            model_name="viktorifit-workout-model",
            files=[workout_path],
            project=project, entity=entity,
            metadata={"model_arch": "weighted_knn", "domain": "fitness"},
            description="Weighted KNN for workout matching",
        )
    else:
        print("⏭️  Skipping Workout Model: File not found.")

    # 3. PROGRESS MODEL UPLOAD (ONNX ENSEMBLE)
    print("📦 [3/3] Processing Progress Predictor (ONNX)...")
    onnx_dir = "models/onnx"
    if os.path.isdir(onnx_dir):
        onnx_files = glob.glob(f"{onnx_dir}/*.onnx")
        if onnx_files:
            push_model_artifact(
                model_name="viktorifit-progress-model",
                files=onnx_files,
                project=project, entity=entity,
                metadata={
                    "model_arch": "onnx_ensemble",
                    "n_targets": len(onnx_files),
                    "domain": "biometrics"
                },
                description=f"Ensemble of {len(onnx_files)} ONNX models for progress tracking",
            )
        else:
            print(f"⚠️  No ONNX files found in {onnx_dir}")
    else:
        print("⏭️  Skipping Progress Model: Directory not found.")

    print("="*70)
    print("🏁 ALL UPLOADS COMPLETED SUCCESSFULLY")
    print(f"View your models at: https://wandb.ai/{entity}/{project}/artifacts")
    print("="*70)

if __name__ == "__main__":
    upload_all_models_separately()