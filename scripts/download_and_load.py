# scripts/download_and_load.py
import os
import wandb
import pickle
import onnxruntime
import argparse

def download_artifact(entity, project, artifact_name, alias="latest", dest=None):
    api = wandb.Api()
    ref = f"{entity}/{project}/{artifact_name}:{alias}"
    print("Downloading", ref)
    art = api.artifact(ref)
    dest_dir = art.download(root=dest) if dest else art.download()
    print("Downloaded to", dest_dir)
    return dest_dir

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_onnx_session(path):
    return onnxruntime.InferenceSession(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, help="artifact name, e.g. viktorifit-meal-model")
    parser.add_argument("--alias", default="latest")
    parser.add_argument("--entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--project", default=os.getenv("WANDB_PROJECT"))
    args = parser.parse_args()

    d = download_artifact(args.entity, args.project, args.artifact, alias=args.alias)
    # try loading any .pickle or .onnx inside
    for root, _, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f)
            if f.endswith(".pickle"):
                try:
                    model = load_pickle(p)
                    print("Loaded pickle:", p, "type:", type(model))
                except Exception as e:
                    print("Failed to load pickle", p, e)
            if f.endswith(".onnx"):
                try:
                    sess = load_onnx_session(p)
                    print("Loaded onnx session:", p, "inputs:", [i.name for i in sess.get_inputs()])
                except Exception as e:
                    print("Failed to load onnx", p, e)

if __name__ == "__main__":
    main()