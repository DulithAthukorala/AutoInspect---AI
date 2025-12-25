import os
from pathlib import Path
import mlflow
import mlflow.artifacts
from ultralytics import YOLO

def train_yolo_with_mlflow(
    data_yaml: str,
    model_ckpt: str = "yolov8s-seg.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/segment",
    name: str = "cardd_seg_mlflow",
    seed: int = 42,
):
    # Keep runs local inside your repo (professional + portable)
    mlruns_dir = Path("mlruns")
    mlruns_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"file:{mlruns_dir.resolve()}")

    mlflow.set_experiment("AutoInspect-CarDD-YOLOv8Seg")

    with mlflow.start_run(run_name=name):
        # log config (this is what pros do)
        mlflow.log_params({
            "model_ckpt": model_ckpt,
            "data_yaml": data_yaml,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "seed": seed,
            "project": project,
            "name": name,
        })

        model = YOLO(model_ckpt)

        results = model.train(
            task="segment",
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=name,
            seed=seed,
            deterministic=True,
            plots=True,
            save=True,
        )

        # Ultralytics puts outputs here:
        run_dir = Path(project) / name
        # log artifacts: weights, plots, results.csv, args.yaml, etc.
        if run_dir.exists():
            mlflow.log_artifacts(str(run_dir), artifact_path="yolo_run")

        # Log key metrics from results.csv if available (very important)
        csv_path = run_dir / "results.csv"
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            # log last epoch metrics (simple + useful)
            last = df.iloc[-1].to_dict()
            for k, v in last.items():
                if k == "epoch":
                    continue
                try:
                    mlflow.log_metric(k, float(v))
                except Exception:
                    pass

        print(f"✅ MLflow run logged. YOLO outputs: {run_dir}")
        return results


if __name__ == "__main__":
    # Example:
    # python src/train_with_mlflow.py yolo_data/data.yaml
    import sys
    if len(sys.argv) != 2:
        print("Usage: python src/train_with_mlflow.py <path_to_data.yaml>")
        raise SystemExit(1)

    data_yaml = sys.argv[1]
    train_yolo_with_mlflow(data_yaml=data_yaml)
