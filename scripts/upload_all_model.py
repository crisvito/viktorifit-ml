#!/usr/bin/env python3
"""
Upload Models to W&B
=====================================
Uploads each model as a SEPARATE artifact for independent versioning.

This replaces upload_all_model.py which incorrectly bundled everything together.
"""

import os
import wandb
from datetime import datetime
import glob

def get_next_version(entity, project, artifact_name):
    """Get next version number for an artifact"""
    api = wandb.Api()
    try:
        collection = api.artifact_collection(
            type_name="model", 
            name=f"{entity}/{project}/{artifact_name}"
        )
        versions = collection.artifacts()
        
        if not versions:
            return 1
            
        latest_version = max(int(a.version[1:]) for a in versions)
        return latest_version + 1
        
    except:
        return 1

def push_model_artifact(
    model_name, 
    files, 
    project, 
    entity, 
    metadata=None,
    description=None
):
    """
    Upload a single model artifact to W&B
    
    Args:
        model_name: Artifact name (e.g., 'viktorifit-meal-model')
        files: List of file paths to include
        project: W&B project name
        entity: W&B entity/username
        metadata: Dict of metadata to attach
        description: Human-readable description
    """
    metadata = metadata or {}

    # Initialize W&B run
    run = wandb.init(
        project=project, 
        entity=entity, 
        job_type="upload-model", 
        reinit=True
    )

    # Generate version
    next_version = get_next_version(entity, project, model_name)
    version_alias = f"v{next_version}"
    timestamp_alias = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    # Add tracking info to metadata
    metadata.update({
        "version": version_alias,
        "timestamp": timestamp_alias,
        "uploaded_at": datetime.utcnow().isoformat() + "Z"
    })

    # Create artifact
    artifact = wandb.Artifact(
        name=model_name,
        type="model",
        description=description or f"Model artifact for {model_name}",
        metadata=metadata
    )

    # Add files
    files_added = 0
    for file_path in files:
        if os.path.isdir(file_path):
            artifact.add_dir(file_path)
            files_added += len(glob.glob(f"{file_path}/**/*", recursive=True))
        elif os.path.exists(file_path):
            artifact.add_file(file_path)
            files_added += 1
        else:
            print(f"⚠️  File not found: {file_path}")

    # Log artifact with aliases
    run.log_artifact(
        artifact,
        aliases=[version_alias, timestamp_alias, "latest", "production"]
    )

    run.finish()

    # Print summary
    print(f"✅ Uploaded: {model_name}")
    print(f"   Version: {version_alias}")
    print(f"   Files: {files_added}")
    print(f"   Timestamp: {timestamp_alias}")
    print()

def upload_all_models_separately():
    """
    Upload each model as a SEPARATE artifact
    """
    
    project = os.environ.get("WANDB_PROJECT")
    entity = os.environ.get("WANDB_ENTITY")
    
    if not project or not entity:
        raise ValueError("WANDB_PROJECT and WANDB_ENTITY must be set")
    
    print("="*70)
    print("UPLOADING MODELS TO W&B (SEPARATE ARTIFACTS)")
    print("="*70)
    print(f"Project: {project}")
    print(f"Entity: {entity}")
    print()
    
    # ========================================
    # 1. MEAL PREDICTOR
    # ========================================
    print("📦 1/3: Meal Predictor")
    
    if os.path.exists("models/model_meal.pickle"):
        push_model_artifact(
            model_name="viktorifit-meal-model",
            files=["models/model_meal.pickle"],
            project=project,
            entity=entity,
            metadata={
                "model_type": "knn",
                "algorithm": "NearestNeighbors",
                "n_neighbors": 3,
                "features": ["Energy", "Protein", "Carbs"],
                "purpose": "Food recommendation system"
            },
            description="KNN-based meal recommendation model"
        )
    else:
        print("⚠️  models/model_meal.pickle not found, skipping")
    
    # ========================================
    # 2. WORKOUT PREDICTOR
    # ========================================
    print("📦 2/3: Workout Predictor")
    
    if os.path.exists("models/model_workout.pickle"):
        push_model_artifact(
            model_name="viktorifit-workout-model",
            files=["models/model_workout.pickle"],
            project=project,
            entity=entity,
            metadata={
                "model_type": "weighted_knn",
                "algorithm": "NearestNeighbors (Weighted)",
                "n_neighbors": 1,
                "features": [
                    "Goal_Encoded", "level_Encoded", "Workout_Frequency_x",
                    "environment_Encoded", "Gender_Encoded", "Age_x",
                    "Initial_Weight_kg_x", "Badminton", "Football",
                    "Basketball", "Volleyball", "Swim"
                ],
                "purpose": "Workout buddy matching"
            },
            description="Weighted KNN for workout partner matching"
        )
    else:
        print("⚠️  models/model_workout.pickle not found, skipping")
    
    # ========================================
    # 3. PROGRESS PREDICTOR (ONNX)
    # ========================================
    print("📦 3/3: Progress Predictor (ONNX)")
    
    onnx_dir = "models/onnx"
    
    if os.path.exists(onnx_dir):
        # Get all ONNX files
        onnx_files = glob.glob(f"{onnx_dir}/*.onnx")
        
        if onnx_files:
            push_model_artifact(
                model_name="viktorifit-progress-model",
                files=onnx_files,
                project=project,
                entity=entity,
                metadata={
                    "model_type": "xgboost_ensemble_onnx",
                    "algorithm": "XGBoost Regressor (ONNX)",
                    "n_models": len(onnx_files),
                    "targets": [
                        os.path.basename(f).replace('.onnx', '') 
                        for f in onnx_files
                    ],
                    "features": [
                        "Age", "Gender_Encoded", "Height_cm", "Initial_Weight_kg",
                        "BMI_Category_x_Encoded", "Body_Fat_Category",
                        "Goal_Encoded", "Workout_Frequency", "Average_Duration_Minutes",
                        "level_Encoded", "Badminton", "Football", "Basketball",
                        "Volleyball", "Swim", "Week"
                    ],
                    "purpose": "User progress prediction"
                },
                description=f"XGBoost ensemble ({len(onnx_files)} ONNX models) for progress tracking"
            )
        else:
            print(f"⚠️  No ONNX files found in {onnx_dir}")
    else:
        print(f"⚠️  {onnx_dir} not found, skipping")
    
    # ========================================
    # SUMMARY
    # ========================================
    print("="*70)
    print("✅ UPLOAD COMPLETE")
    print("="*70)
    print()
    print("View artifacts at:")
    print(f"https://wandb.ai/{entity}/{project}/artifacts")
    print()
    print("You should now see 3 separate artifacts:")
    print("  • viktorifit-meal-model")
    print("  • viktorifit-workout-model")
    print("  • viktorifit-progress-model")

if __name__ == "__main__":
    upload_all_models_separately()