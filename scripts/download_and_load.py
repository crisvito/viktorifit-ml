#!/usr/bin/env python3
"""
Download Models from W&B
=========================================
Downloads specific model artifacts from W&B.

Usage:
    # Download specific model
    python download_models.py --model meal
    python download_models.py --model workout
    python download_models.py --model progress
    
    # Download specific version
    python download_models.py --model meal --version v2
    
    # Download all models
    python download_models.py --all
"""

import os
import wandb
import pickle
import onnxruntime
import argparse

# Model artifact mapping
ARTIFACT_NAMES = {
    'meal': 'viktorifit-meal-model',
    'workout': 'viktorifit-workout-model',
    'progress': 'viktorifit-progress-model'
}

def download_artifact(entity, project, artifact_name, alias="latest", dest=None):
    """
    Download an artifact from W&B
    
    Args:
        entity: W&B entity/username
        project: W&B project name
        artifact_name: Name of the artifact (e.g., 'viktorifit-meal-model')
        alias: Version alias (default: 'latest')
        dest: Destination directory (default: auto)
    
    Returns:
        Path to downloaded directory
    """
    api = wandb.Api()
    ref = f"{entity}/{project}/{artifact_name}:{alias}"
    
    print(f"📥 Downloading: {ref}")
    
    try:
        art = api.artifact(ref)
        dest_dir = art.download(root=dest) if dest else art.download()
        
        print(f"✅ Downloaded to: {dest_dir}")
        
        # Print metadata
        if art.metadata:
            print(f"\n📊 Metadata:")
            for key, value in art.metadata.items():
                if isinstance(value, list) and len(value) > 5:
                    print(f"   {key}: {value[:5]}... ({len(value)} items)")
                else:
                    print(f"   {key}: {value}")
        
        return dest_dir
        
    except Exception as e:
        print(f"❌ Error downloading {artifact_name}: {e}")
        return None

def load_pickle(path):
    """Load a pickle file"""
    with open(path, "rb") as f:
        return pickle.load(f)

def load_onnx_session(path):
    """Load an ONNX model as inference session"""
    return onnxruntime.InferenceSession(path)

def test_model_files(directory):
    """
    Test loading all model files in a directory
    
    Args:
        directory: Path to directory containing model files
    """
    print(f"\n🔍 Testing model files in: {directory}")
    print("-" * 70)
    
    found_files = False
    
    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            
            if filename.endswith(".pickle"):
                found_files = True
                try:
                    model_data = load_pickle(filepath)
                    print(f"✅ Pickle: {filename}")
                    print(f"   Type: {type(model_data)}")
                    
                    if isinstance(model_data, dict):
                        print(f"   Keys: {list(model_data.keys())}")
                        
                        # Check for common model components
                        if 'knn_model' in model_data:
                            knn = model_data['knn_model']
                            print(f"   KNN: {type(knn).__name__}")
                        
                        if 'features' in model_data:
                            features = model_data['features']
                            print(f"   Features ({len(features)}): {features[:3]}...")
                    
                    print()
                    
                except Exception as e:
                    print(f"❌ Failed to load pickle {filename}: {e}")
                    print()
            
            elif filename.endswith(".onnx"):
                found_files = True
                try:
                    sess = load_onnx_session(filepath)
                    print(f"✅ ONNX: {filename}")
                    
                    inputs = sess.get_inputs()
                    outputs = sess.get_outputs()
                    
                    print(f"   Inputs: {[i.name for i in inputs]}")
                    print(f"   Input shape: {inputs[0].shape if inputs else 'N/A'}")
                    print(f"   Outputs: {[o.name for o in outputs]}")
                    print()
                    
                except Exception as e:
                    print(f"❌ Failed to load ONNX {filename}: {e}")
                    print()
    
    if not found_files:
        print("⚠️  No model files found")

def download_model(model_name, entity, project, version="latest", test=True):
    """
    Download a specific model and optionally test it
    
    Args:
        model_name: Short name (meal, workout, progress)
        entity: W&B entity
        project: W&B project
        version: Version alias
        test: Whether to test loading the model
    """
    if model_name not in ARTIFACT_NAMES:
        print(f"❌ Unknown model: {model_name}")
        print(f"Available models: {list(ARTIFACT_NAMES.keys())}")
        return None
    
    artifact_name = ARTIFACT_NAMES[model_name]
    
    dest_dir = download_artifact(
        entity=entity,
        project=project,
        artifact_name=artifact_name,
        alias=version
    )
    
    if dest_dir and test:
        test_model_files(dest_dir)
    
    return dest_dir

def download_all_models(entity, project, version="latest"):
    """Download all model artifacts"""
    print("="*70)
    print("DOWNLOADING ALL MODELS")
    print("="*70)
    print()
    
    results = {}
    
    for short_name, artifact_name in ARTIFACT_NAMES.items():
        print(f"\n📦 {short_name.upper()} MODEL")
        print("-" * 70)
        
        dest = download_artifact(
            entity=entity,
            project=project,
            artifact_name=artifact_name,
            alias=version
        )
        
        results[short_name] = dest
        
        if dest:
            test_model_files(dest)
    
    print("\n" + "="*70)
    print("✅ DOWNLOAD COMPLETE")
    print("="*70)
    print("\nDownloaded models:")
    for name, path in results.items():
        status = "✅" if path else "❌"
        print(f"  {status} {name}: {path or 'failed'}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Download model artifacts from W&B"
    )
    
    parser.add_argument(
        "--model",
        choices=['meal', 'workout', 'progress'],
        help="Model to download (meal, workout, or progress)"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all models"
    )
    
    parser.add_argument(
        "--version",
        default="latest",
        help="Version alias (default: latest). Can also use: production, v0, v1, etc."
    )
    
    parser.add_argument(
        "--entity",
        default=os.getenv("WANDB_ENTITY"),
        help="W&B entity (default: from WANDB_ENTITY env var)"
    )
    
    parser.add_argument(
        "--project",
        default=os.getenv("WANDB_PROJECT", "viktorifit-ml"),
        help="W&B project (default: from WANDB_PROJECT env var or 'viktorifit-ml')"
    )
    
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip testing model files after download"
    )
    
    args = parser.parse_args()
    
    if not args.entity:
        print("❌ Error: WANDB_ENTITY not set")
        print("Set via: export WANDB_ENTITY=your-username")
        print("Or use: --entity your-username")
        return
    
    if args.all:
        download_all_models(
            entity=args.entity,
            project=args.project,
            version=args.version
        )
    elif args.model:
        download_model(
            model_name=args.model,
            entity=args.entity,
            project=args.project,
            version=args.version,
            test=not args.no_test
        )
    else:
        print("❌ Error: Specify --model or --all")
        parser.print_help()

if __name__ == "__main__":
    main()