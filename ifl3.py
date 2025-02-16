import os
import glob
import shutil
import copy
import torch
from ultralytics import YOLO

def average_state_dicts(state_dicts):
    avg_state = {}
    for key in state_dicts[0].keys():
        avg_tensor = sum(client_sd[key] for client_sd in state_dicts) / len(state_dicts)
        avg_state[key] = avg_tensor
    return avg_state

def federated_training(global_model, data_path, imgsz, local_epochs, num_clients, global_epochs, batch):
    for global_epoch in range(global_epochs):
        print(f"\n--- Global Epoch {global_epoch + 1}/{global_epochs} ---")
        client_state_dicts = []
        
        for client in range(num_clients):
            print(f"Client {client + 1}/{num_clients} training for {local_epochs} epoch(s)...")
            client_model = copy.deepcopy(global_model)
            client_model.train(
                data=data_path,
                imgsz=imgsz,
                epochs=local_epochs,
                batch=batch,
            )
            client_state = client_model.model.state_dict()
            client_state_dicts.append(client_state)
        
        new_state_dict = average_state_dicts(client_state_dicts)
        # Load the aggregated weights with strict=True (after ensuring architectures match)
        global_model.model.load_state_dict(new_state_dict)
        print("Aggregated global model updated.")
    
    return global_model

def main():
    # Load the YOLO model (pre-trained on COCO with 80 classes)
    global_model = YOLO("yolo11s.pt")
    
    # Suppose your dataset has 3 classes.
    # Ensure the model architecture matches your dataset. Check your data.yaml for nc: 3.
    # The following is an example and might require adjustments per ultralytics' API.
    global_model.overrides = {"nc": 3}
    global_model.model.nc = 3
    if hasattr(global_model.model, 'initialize_heads'):
        global_model.model.initialize_heads()
    
    data_path = "E:/BioM/archive/BrainTumor/BrainTumorYolov11/data.yaml"
    imgsz = 32
    local_epochs = 2
    num_clients = 3
    global_epochs = 3
    batch = 1048
    
    global_model = federated_training(global_model, data_path, imgsz, local_epochs, num_clients, global_epochs, batch)
    
    torch.save(global_model.model.state_dict(), "global_model.pt")
    print("\nFinal global model saved to 'global_model.pt'.")
    
    run_dirs = sorted(glob.glob(os.path.join("runs", "detect", "train*")), key=os.path.getmtime)
    if run_dirs:
        run_dir = run_dirs[-1]
        csv_source = os.path.join(run_dir, "results.csv")
        csv_destination = "yolo.csv"
        if os.path.exists(csv_source):
            shutil.copy(csv_source, csv_destination)
            print(f"Training metrics saved to '{csv_destination}'.")
        else:
            print(f"Could not find 'results.csv' in {run_dir}.")
    else:
        print("No training run directory found.")

if __name__ == '__main__':
    main()



