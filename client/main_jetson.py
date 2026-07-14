import argparse
import os
import sys
import torch
import flwr as fl
from collections import OrderedDict
import psutil
import time
import subprocess

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
YOLO_DIR = os.path.join(BASE_DIR, "..", "yolov5") if os.path.exists(os.path.join(BASE_DIR, "..", "yolov5")) else "/home/js1/yolov5"

# Détection dynamique du dossier utilisateur pour s'adapter à js1 et comvis
if "comvis" in BASE_DIR:
    YOLO_DIR = "/home/comvis/yolov5"
    BASE_DIR = "/home/comvis/workspace"
elif "js1" in BASE_DIR:
    YOLO_DIR = "/home/js1/yolov5"
    BASE_DIR = "/home/js1/workspace"

sys.path.append(YOLO_DIR)

try:
    from models.yolo import Model
    import val as yolov5_val
except ImportError:
    print(f"Erreur : Impossible de charger YOLOv5 depuis {YOLO_DIR}")
    sys.exit(1)

class YOLOv5Client(fl.client.NumPyClient):
    def __init__(self, data_path, device, device_type, device_id):
        self.data_path = data_path
        self.device = device
        self.device_type = device_type
        self.device_id = device_id
        self.model = self.load_base_architecture()

    def load_base_architecture(self):
        """Crée la structure d'entraînement native v6.1 (270 couches, nc=8) avec les poids valides."""
        model = Model(os.path.join(YOLO_DIR, "models/yolov5s.yaml"), ch=3, nc=8).to(self.device)
        weights_path = os.path.join(YOLO_DIR, "yolov5s.pt")
        if os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
            ckpt_state = ckpt['model'].float().state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']
            intersect_state = {k: v for k, v in ckpt_state.items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
            model.load_state_dict(intersect_state, strict=False)
        return model

    def get_parameters(self, config=None):
        # On exclut les buffers d'ancres non-entraînables
        return [val.cpu().numpy() for name, val in self.model.state_dict().items() if "anchor" not in name]

    def set_parameters(self, parameters):
        # On filtre les clés de la même manière
        keys = [k for k in self.model.state_dict().keys() if "anchor" not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # strict=False permet d'ignorer les buffers statiques comme anchor_grid
        self.model.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # --- EXTRACTION DE LA CONFIGURATION DU SERVEUR ---
        epochs = "1"
        batch_size = "4"
        workers = "2"
        
        if "client_configs" in config:
            import json
            try:
                all_configs = json.loads(config["client_configs"])
                # Cherche l'ID de la carte (ex: js1), sinon utilise "default"
                client_config = all_configs.get(self.device_id, all_configs.get("default", {}))
                
                epochs = str(client_config.get("epochs", 1))
                batch_size = str(client_config.get("batch_size", 4))
                workers = str(client_config.get("workers", 2))
                
                print(f"\n[CONFIG SERVEUR REÇUE] Appareil: {self.device_id} | Epochs: {epochs} | Batch: {batch_size} | Workers: {workers}")
            except Exception as e:
                print(f"[ATTENTION] Erreur lors de la lecture de la config serveur : {e}. Valeurs par défaut appliquées.")

        temp_weight_path = os.path.join(BASE_DIR, "temp_flower_weights.pt")
        torch.save({
            'model': self.model,
            'optimizer': None,
            'epoch': -1,
            'wandb_id': None
        }, temp_weight_path)
        
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(2) 
        
        print("\n" + "="*60)
        print(f"    [Round Flower] ENTRAÎNEMENT LOCAL - APPAREIL: {self.device_id}")
        print("="*60 + "\n")
        
        start_time = time.time()
        cpu_start = psutil.cpu_percent(interval=None)
        ram_start = psutil.virtual_memory().used / (1024 ** 2)

        # Les variables dynamiques sont maintenant injectées ici :
        cmd = [
            "python3", os.path.join(YOLO_DIR, "train.py"),
            "--weights", temp_weight_path,
            "--data", self.data_path,
            "--epochs", epochs,          # <-- DYNAMIQUE
            "--batch-size", batch_size,  # <-- DYNAMIQUE
            "--imgsz", "320",
            "--device", "0" if torch.cuda.is_available() else "cpu",
            "--project", os.path.join(YOLO_DIR, "runs/train"),
            "--name", "flower_local",
            "--exist-ok",
            "--workers", workers         # <-- DYNAMIQUE
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Erreur critique durant l'entraînement YOLOv5 : {e}")
            self.model = self.load_base_architecture()
            return self.get_parameters(config={}), 0, {}

        # Extraction de la Loss locale calculée
        local_loss = 0.0
        results_csv = os.path.join(YOLO_DIR, "runs", "train", "flower_local", "results.csv")
        if os.path.exists(results_csv):
            try:
                with open(results_csv, "r") as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                    if len(lines) > 1:
                        last_line = lines[-1].split(',')
                        if len(last_line) >= 4:
                            local_loss = float(last_line[1]) + float(last_line[2]) + float(last_line[3])
                        else:
                            print(f"[ATTENTION] Ligne de résultats CSV incomplète : {last_line}")
            except Exception as e:
                print(f"[ERREUR] Impossible de récupérer la Loss depuis le CSV : {e}")
                local_loss = 0.0

        trained_weights_path = os.path.join(YOLO_DIR, "runs", "train", "flower_local", "weights", "last.pt")
        if os.path.exists(trained_weights_path):
            checkpoint = torch.load(trained_weights_path, map_location=self.device, weights_only=False)
            self.model = checkpoint['model'].to(self.device)
        else:
            self.model = self.load_base_architecture()

        duration = time.time() - start_time
        cpu_end = psutil.cpu_percent(interval=None)
        ram_end = psutil.virtual_memory().used / (1024 ** 2)

        print("[YOLOv5] Calcul de la précision locale...")
        try:
            results, _, _ = yolov5_val.run(
                data=self.data_path, 
                weights=trained_weights_path if os.path.exists(trained_weights_path) else temp_weight_path,
                batch_size=int(batch_size), imgsz=320, device=self.device, workers=0, plots=False
            )
            map50 = results[2]
            map50_95 = results[3]
        except Exception as e:
            print(f"Erreur lors de la validation : {e}")
            map50, map50_95 = 0.0, 0.0

        if os.path.exists(temp_weight_path):
            os.remove(temp_weight_path)

        parameters_to_return = self.get_parameters(config={})
        bytes_received = sum(p.nbytes for p in parameters)
        bytes_sent = sum(p.nbytes for p in parameters_to_return)

        metrics = {
            "device_type": str(self.device_type),
            "device_id": str(self.device_id),
            "accuracy_map50": float(map50),
            "accuracy_map50_95": float(map50_95),
            "training_duration_sec": float(duration),
            "cpu_usage_percent": float((cpu_start + cpu_end) / 2),
            "ram_usage_mb": float(ram_end - ram_start if ram_end > ram_start else ram_end),
            "train_loss": float(local_loss),
            "network_received_mb": float(bytes_received / (1024 ** 2)),
            "network_sent_mb": float(bytes_sent / (1024 ** 2))
        }

        if not hasattr(self, 'model') or self.model is None:
            if os.path.exists(trained_weights_path):
                checkpoint = torch.load(trained_weights_path, map_location=self.device, weights_only=False)
                self.model = checkpoint['model'].to(self.device)
            else:
                self.model = self.load_base_architecture()

        return parameters_to_return, 1, metrics
        

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, 1, {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Chemin absolu vers data.yaml")
    parser.add_argument("--server", type=str, required=True, help="IP:Port du serveur Flower")
    parser.add_argument("--device-type", type=str, required=True, help="jetson ou raspberry")
    parser.add_argument("--device-id", type=str, required=True, help="Nom unique de la carte")
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        print(f"[ERREUR] Fichier data.yaml introuvable : {args.data}")
        print("Corrigez --data avec le chemin absolu vers le vrai fichier data.yaml de votre dataset YOLO.")
        sys.exit(1)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Démarrage de l'appareil {args.device_id} ({args.device_type}) sur {device}")
    
    fl.client.start_numpy_client(
        server_address=args.server, 
        client=YOLOv5Client(data_path=args.data, device=device, device_type=args.device_type, device_id=args.device_id)
    )

if __name__ == "__main__":
    main()

    
