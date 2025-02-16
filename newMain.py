import os
import glob
import shutil
from ultralytics import YOLO

def main():
    # Load the YOLO model (make sure the weights path is correct).
    model = YOLO("yolo11s.pt")
    
    # Start training. Adjust the data path, image size, epochs, and batch size as needed.
    results = model.train(
        data="E:/BioM/archive/BrainTumor/BrainTumorYolov11/data.yaml",
        imgsz=64,
        epochs=100,   # Change to desired number of epochs.
        batch=256,
    )

    # After training, the ultralytics training process saves a results CSV file
    # (which includes metrics like precision, recall, mAP50, mAP50-95, and losses)
    # in the run directory. We need to locate that CSV file.

    # If the results object has a 'path' attribute, use it.
    if hasattr(results, "path"):
        run_dir = results.path
    else:
        # Otherwise, search for the latest run directory in 'runs/detect/train*'
        run_dirs = sorted(glob.glob(os.path.join("runs", "detect", "train*")), key=os.path.getmtime)
        if run_dirs:
            run_dir = run_dirs[-1]
        else:
            print("No training run directory found.")
            return

    # The training run directory is expected to contain a 'results.csv' file.
    csv_source = os.path.join(run_dir, "results.csv")
    csv_destination = "yolo.csv"
    if os.path.exists(csv_source):
        shutil.copy(csv_source, csv_destination)
        print(f"Training metrics saved to {csv_destination}")
    else:
        print(f"Could not find 'results.csv' in {run_dir}.")

if __name__ == '__main__':
    main()
