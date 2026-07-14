import flwr as fl
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
 
# Configuration des chemins
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'yolov5'))

CACHE_DIR = os.environ.get("FL_CACHE_DIR", "/tmp/yolo-fl-cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(CACHE_DIR, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", CACHE_DIR)
 
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
except ImportError as e:
    print("\n--- ERREUR D'IMPORTATION DETECTEE ---")
    traceback.print_exc()
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
 
            if server_round == current_config["num_rounds"]:
                try:
                    params_np = fl.common.parameters_to_ndarrays(aggregated_parameters)
                    yaml_cfg = os.path.join(BASE_DIR, self.template_cfg_path)
                    model = Model(cfg=yaml_cfg, ch=3, nc=8)
                    
                    # CORRECTION 2 : Filtrage des clés d'ancres YOLOv5
                    keys = [k for k in model.state_dict().keys() if "anchor" not in k]
                    
                    params_dict = zip(keys, params_np)
                    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})  
                    model.load_state_dict(state_dict, strict=False)
                    
                    torch.save({'model': model}, self.output_model_path)
                    print(f"-> Modèle final sauvegardé au Round {server_round} : '{self.output_model_path}'")
                except Exception as e:
                    print(f"Erreur reconstruction modèle : {e}")
 
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
