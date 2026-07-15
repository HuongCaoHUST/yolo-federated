import flwr as fl
import numpy as np
if int(np.__version__.split(".")[0]) >= 2:
    raise RuntimeError(
        f"Server yêu cầu NumPy 1.x với torch 2.2.2, nhưng image đang dùng NumPy {np.__version__}. "
        "Hãy rebuild image bằng: docker compose build --no-cache"
    )
import torch
import sys
import os
import csv
import time
from collections import OrderedDict
from typing import List, Tuple, Dict
from flwr.common import Metrics, Scalar
import traceback
import json
from pathlib import Path
import yaml
 
# Configuration des chemins
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
YOLO_DIR = os.environ.get("YOLO_DIR", os.path.join(BASE_DIR, "yolov5"))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, YOLO_DIR)

CACHE_DIR = os.environ.get("FL_CACHE_DIR", "/tmp/yolo-fl-cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(CACHE_DIR, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", CACHE_DIR)
os.environ.setdefault("YOLOV5_CONFIG_DIR", os.path.join(CACHE_DIR, "Ultralytics"))
os.makedirs(os.environ["YOLOV5_CONFIG_DIR"], exist_ok=True)
 
# On définit le chemin absolu du fichier une bonne fois pour toutes
CONFIG_PATH = os.environ.get(
    "FL_CONFIG_PATH", os.path.join(os.path.dirname(__file__), "config_server.json")
)

OUTPUT_DIR = os.environ.get("FL_OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "output"))
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
# La fonction utilise directement cette variable globale
def load_server_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
    
config = load_server_config()
 
# CSV file pour enregistrer toutes les données
csv_file = os.path.join(OUTPUT_DIR, "rapport_federated_learning.csv")
 
# Fonction Flower appelée à chaque début de round pour envoyer les configs aux clients
def fit_config(server_round: int) -> Dict[str, Scalar]:
    current_config = load_server_config()
    return {
        "server_round": server_round,
        "client_configs": json.dumps(current_config["clients"])
    }
 
try:
    from models.yolo import Model
    import val as yolov5_val
    from utils.dataloaders import create_dataloader
    from utils.general import check_dataset
    from utils.loss import ComputeLoss
except ImportError as e:
    print("\n--- ERREUR D'IMPORTATION DETECTEE ---")
    traceback.print_exc()
    print(f"YOLO_DIR attendu : {YOLO_DIR}")
    print(f"models/yolo.py existe : {os.path.isfile(os.path.join(YOLO_DIR, 'models', 'yolo.py'))}")
    print("-------------------------------------------\n")
    sys.exit(1)
 
 
def aggregate_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if not metrics:
        return {}
 
    # CORRECTION 1 : La vérification doit se faire à CHAQUE appel
    file_exists = os.path.exists(csv_file)
 
    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
 
        if not file_exists:
            writer.writerow(["Horodatage", "Device_ID", "Type_Appareil", "mAP50", "mAP50-95",
                             "Temps_Entrainement_Sec", "CPU_Moyen_Pourcent",
                             "RAM_Delta_Mo", "Loss_Entrainement", "Donnees_Recues_Mo", "Donnees_Envoyees_Mo"])
 
        for client_num, (num_examples, m) in enumerate(metrics):
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                m.get('device_id', f"Inconnu_{client_num}"),
                m.get('device_type', 'Inconnu'),
                m.get('accuracy_map50', 0.0),
                m.get('accuracy_map50_95', 0.0),
                m.get('training_duration_sec', 0.0),
                m.get('cpu_usage_percent', 0.0),
                m.get('ram_usage_mb', 0.0),
                m.get('train_loss', 0.0),
                m.get('network_received_mb', 0.0),
                m.get('network_sent_mb', 0.0)
            ])
 
    return {}
 
 
class YOLOv5AggregateStrategy(fl.server.strategy.FedAvg):
 
    def __init__(self, template_cfg_path="yolov5/models/yolov5s.yaml", output_model_path="model_federated_round_3.pt", **kwargs):
        super().__init__(**kwargs)
        self.template_cfg_path = template_cfg_path
        self.output_model_path = output_model_path

    @staticmethod
    def _dataset_metadata(data_path):
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Không tìm thấy dataset YAML: {data_path}")
        with open(data_path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        names = data.get("names")
        num_classes = int(data.get("nc", len(names) if names is not None else 0))
        if num_classes < 1:
            raise ValueError(f"Không xác định được số lớp từ {data_path}")
        return num_classes, names

    @staticmethod
    def _array_to_tensor(value):
        """Convert ndarray through the buffer protocol, without torch's NumPy bridge."""
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "int8": torch.int8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64,
            "uint8": torch.uint8,
            "bool": torch.bool,
        }
        dtype_name = str(value.dtype)
        if dtype_name not in dtype_map:
            raise TypeError(f"Kiểu tensor global chưa được hỗ trợ: {dtype_name}")
        # bytearray owns writable memory, so the resulting tensor remains valid.
        raw = bytearray(value.tobytes(order="C"))
        return torch.frombuffer(raw, dtype=dtype_map[dtype_name]).reshape(value.shape).clone()

    def _build_global_model(self, aggregated_parameters, data_path):
        params_np = fl.common.parameters_to_ndarrays(aggregated_parameters)
        yaml_cfg = os.path.join(BASE_DIR, self.template_cfg_path)
        num_classes, names = self._dataset_metadata(data_path)
        model = Model(cfg=yaml_cfg, ch=3, nc=num_classes).cpu()
        if names is not None:
            model.names = names
        keys = [key for key in model.state_dict().keys() if "anchor" not in key]
        if len(keys) != len(params_np):
            raise ValueError(f"Số tensor global ({len(params_np)}) không khớp model ({len(keys)})")
        state_dict = OrderedDict((key, self._array_to_tensor(value)) for key, value in zip(keys, params_np))
        model.load_state_dict(state_dict, strict=False)
        return model

    @staticmethod
    def _write_json(path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)

    def _validate_global_model(self, server_round, model, validation_config):
        data_path = validation_config.get("data", "/data/global_val/data.yaml")
        status_path = Path(OUTPUT_DIR) / "validation_status.json"
        metrics_path = Path(OUTPUT_DIR) / "global_metrics.json"
        metrics_csv = Path(OUTPUT_DIR) / "global_metrics.csv"
        started_at = time.time()

        self._write_json(status_path, {
            "running": True, "round": server_round, "current_batch": 0,
            "total_batches": 0, "percent": 0, "message": "Đang chuẩn bị dữ liệu validation",
        })
        print(f"[GLOBAL VAL][START] Round {server_round} | data={data_path}", flush=True)

        try:
            if not os.path.isfile(data_path):
                raise FileNotFoundError(f"Không tìm thấy global validation YAML: {data_path}")
            data = check_dataset(data_path)
            model_classes = int(model.model[-1].nc)
            if int(data["nc"]) != model_classes:
                raise ValueError(
                    f"Dataset validation có {data['nc']} lớp nhưng global model có {model_classes} lớp"
                )

            model.names = data["names"]
            hyp_path = os.path.join(YOLO_DIR, "data", "hyps", "hyp.scratch-low.yaml")
            with open(hyp_path, encoding="utf-8") as handle:
                model.hyp = yaml.safe_load(handle)
            model.eval()

            imgsz = int(validation_config.get("imgsz", 320))
            batch_size = int(validation_config.get("batch_size", 8))
            workers = int(validation_config.get("workers", 2))
            stride = max(int(model.stride.max()), 32)
            dataloader = create_dataloader(
                data["val"], imgsz, batch_size, stride, False,
                pad=0.5, rect=True, workers=workers, prefix="global-val: ",
            )[0]
            total_batches = len(dataloader)
            self._write_json(status_path, {
                "running": True, "round": server_round, "current_batch": 0,
                "total_batches": total_batches, "percent": 0,
                "message": "Đang validation global model",
            })

            strategy = self
            class TrackedLoader:
                def __init__(self, loader):
                    self.loader = loader
                    self.dataset = loader.dataset

                def __len__(self):
                    return len(self.loader)

                def __getattr__(self, name):
                    return getattr(self.loader, name)

                def __iter__(self):
                    for index, batch in enumerate(self.loader, start=1):
                        yield batch
                        percent = round(index / max(len(self.loader), 1) * 100)
                        strategy._write_json(status_path, {
                            "running": True, "round": server_round,
                            "current_batch": index, "total_batches": len(self.loader),
                            "percent": percent, "message": "Đang validation global model",
                        })

            save_dir = Path(OUTPUT_DIR) / "validation" / f"round_{server_round:03d}"
            save_dir.mkdir(parents=True, exist_ok=True)
            results, _, _ = yolov5_val.run(
                data=data,
                model=model,
                dataloader=TrackedLoader(dataloader),
                batch_size=batch_size,
                imgsz=imgsz,
                half=False,
                plots=False,
                save_dir=save_dir,
                compute_loss=ComputeLoss(model),
            )
            precision, recall, map50, map5095, box_loss, obj_loss, cls_loss = map(float, results)
            record = {
                "round": server_round,
                "precision": precision,
                "recall": recall,
                "map50": map50,
                "map50_95": map5095,
                "box_loss": box_loss,
                "obj_loss": obj_loss,
                "cls_loss": cls_loss,
                "total_loss": box_loss + obj_loss + cls_loss,
                "duration_sec": round(time.time() - started_at, 3),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            history = []
            if metrics_path.exists():
                history = json.loads(metrics_path.read_text(encoding="utf-8"))
            history = [item for item in history if item.get("round") != server_round]
            history.append(record)
            history.sort(key=lambda item: item["round"])
            self._write_json(metrics_path, history)

            with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(record.keys()))
                writer.writeheader()
                writer.writerows(history)

            self._write_json(status_path, {
                "running": False, "round": server_round,
                "current_batch": total_batches, "total_batches": total_batches,
                "percent": 100, "message": "Validation hoàn tất", "metrics": record,
            })
            print(
                f"[GLOBAL VAL][DONE] Round {server_round} | P={precision:.4f} R={recall:.4f} "
                f"mAP50={map50:.4f} mAP50-95={map5095:.4f} loss={record['total_loss']:.4f}",
                flush=True,
            )
            return record
        except Exception as exc:
            self._write_json(status_path, {
                "running": False, "round": server_round, "percent": 0,
                "error": str(exc), "message": "Validation thất bại",
            })
            print(f"[GLOBAL VAL][ERROR] Round {server_round}: {exc}", flush=True)
            traceback.print_exc()
            return None
 
    def aggregate_fit(self, server_round, results, failures):
        if results:
            metrics_to_process = [(res.num_examples, res.metrics) for _, res in results]
            aggregate_fit_metrics(metrics_to_process)

        valid_results = [(client, res) for client, res in results if res.num_examples > 0]
        skipped_results = len(results) - len(valid_results)

        if skipped_results:
            print(f"[ROUND {server_round}] {skipped_results} client(s) ignoré(s) car aucun exemple valide n'a été entraîné.")

        if not valid_results:
            print(f"[ROUND {server_round}] Agrégation annulée : aucun client n'a terminé l'entraînement local.")
            return None, {}

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, valid_results, failures)
 
        if aggregated_parameters is not None:
            print(f"[ROUND {server_round}] Fusion globale des poids réussie.")
            current_config = load_server_config()
            validation_config = current_config.get("global_validation", {})
            try:
                model_data = validation_config.get("data", "/app/coco8.yaml")
                model = self._build_global_model(aggregated_parameters, model_data)
                if validation_config.get("enabled", False):
                    record = self._validate_global_model(server_round, model, validation_config)
                    if record:
                        aggregated_metrics.update({
                            "global_precision": record["precision"],
                            "global_recall": record["recall"],
                            "global_map50": record["map50"],
                            "global_map50_95": record["map50_95"],
                            "global_total_loss": record["total_loss"],
                        })
                if server_round == current_config["num_rounds"]:
                    torch.save({'model': model}, self.output_model_path)
                    print(f"-> Modèle final sauvegardé au Round {server_round} : '{self.output_model_path}'")
            except Exception as e:
                print(f"Erreur reconstruction/validation modèle : {e}")
                traceback.print_exc()
 
        return aggregated_parameters, aggregated_metrics
 
 
def run_server():
    current_config = load_server_config()
    NB_CARTES_ATTENDUES = current_config.get("min_clients_connected", 1)
 
    strategy = YOLOv5AggregateStrategy(
        template_cfg_path="yolov5/models/yolov5s.yaml",
        output_model_path=os.path.join(OUTPUT_DIR, "model_federated_final.pt"),
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NB_CARTES_ATTENDUES,
        min_evaluate_clients=NB_CARTES_ATTENDUES,
        min_available_clients=NB_CARTES_ATTENDUES,
        on_fit_config_fn=fit_config,
    )
    
    print(f"Démarrage du serveur Flower (Attente de {NB_CARTES_ATTENDUES} client(s))...")
 
    try:
        fl.server.start_server(
            server_address=os.environ.get("FL_SERVER_ADDRESS", "0.0.0.0:8080"),
            config=fl.server.ServerConfig(num_rounds=current_config["num_rounds"]),
            strategy=strategy,
        )
        print("\n[SUCCÈS] Apprentissage Fédéré terminé proprement.")
 
    except Exception as e:
        if "GRPCBridgeClosed" in str(e) or "iterating responses" in str(e).lower():
            print("\n[INFO] Serveur arrêté. Les clients se sont déconnectés.")
        else:
            print(f"\n[ERREUR] Le serveur a rencontré un problème : {e}")
 
if __name__ == "__main__":
    run_server()
