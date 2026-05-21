import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from collections import OrderedDict
from dataset_loader import load_client_data, load_client_data_image
import copy
import sys
import os
import atexit
from tqdm import tqdm
import json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ai_engines.audio_engine.cnn14_model import CNN14, freeze_model_blocks
from ai_engines.image_engine.resnet18_model import ResNet18, freeze_model_blocks as freeze_image_blocks
from fl_worker.api_client import api_client


def _safe_div(numer: float, denom: float) -> float:
    return float(numer / denom) if denom else 0.0


def _compute_binary_metrics(probabilities: list[float], labels: list[int]) -> dict:
    preds = [1 if p >= 0.5 else 0 for p in probabilities]
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    auc = None
    try:
        from sklearn.metrics import roc_auc_score

        if len(set(labels)) > 1:
            auc = float(roc_auc_score(labels, probabilities))
    except Exception:
        auc = None

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


class AdvancedPneumoniaClient(fl.client.NumPyClient):
    def __init__(self, client_id, model, trainloader, valloader, device, lr=1e-4, mu=0.001, local_epochs=2):
        self.client_id = client_id
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.mu = mu
        self.local_epochs = local_epochs
        self.current_round = 0

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        self.current_round += 1

        global_model = copy.deepcopy(self.model)
        global_model.eval()
        for param in global_model.parameters():
            param.requires_grad = False

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=1e-4)

        local_epochs = self.local_epochs
        total_steps = len(self.trainloader) * local_epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        api_client.update_training_state(
            current_round=self.current_round,
            current_epoch=0,
            total_epochs=local_epochs,
            status="training",
        )

        total_train_loss = 0.0
        total_train_samples = 0
        total_train_correct = 0
        for epoch in range(local_epochs):
            print(f"\n--- Epoch {epoch+1}/{local_epochs} ---")
            progress_bar = tqdm(self.trainloader, desc="Training", file=sys.stdout, dynamic_ncols=True)

            epoch_loss = 0.0
            epoch_samples = 0
            epoch_correct = 0

            for inputs, labels in progress_bar:
                inputs = inputs.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                if self.mu > 0:
                    proximal_term = 0.0
                    for param, global_param in zip(self.model.parameters(), global_model.parameters()):
                        if param.requires_grad:
                            proximal_term += ((param - global_param) ** 2).sum()
                    loss += (self.mu / 2) * proximal_term

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item() * inputs.size(0)
                total_train_samples += inputs.size(0)
                epoch_loss += loss.item() * inputs.size(0)
                epoch_samples += inputs.size(0)
                with torch.no_grad():
                    probs = torch.sigmoid(outputs)
                    preds = (probs >= 0.5).float()
                    epoch_correct += (preds == labels).sum().item()
                    total_train_correct += (preds == labels).sum().item()
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

            epoch_avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
            epoch_acc = epoch_correct / epoch_samples if epoch_samples > 0 else 0.0
            api_client.update_training_state(
                current_round=self.current_round,
                current_epoch=epoch + 1,
                total_epochs=local_epochs,
                train_loss=epoch_avg_loss,
                train_accuracy=epoch_acc,
                status="training",
            )

        avg_train_loss = total_train_loss / total_train_samples if total_train_samples > 0 else 0.0
        avg_train_acc = total_train_correct / total_train_samples if total_train_samples > 0 else 0.0

        # Quick evaluation after training for accuracy metric
        val_accuracy = None
        val_loss = None
        precision = None
        recall = None
        f1 = None
        auc = None
        if self.valloader and len(self.valloader.dataset) > 0:
            self.model.eval()
            total_correct, total_val = 0, 0
            total_val_loss = 0.0
            all_probs = []
            all_labels = []
            with torch.no_grad():
                for inputs, labels in self.valloader:
                    inputs = inputs.to(self.device)
                    labels = labels.float().unsqueeze(1).to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    total_val_loss += loss.item() * inputs.size(0)
                    probs = torch.sigmoid(outputs)
                    preds = (probs >= 0.5).float()
                    total_correct += (preds == labels).sum().item()
                    total_val += labels.size(0)
                    all_probs.extend(probs.squeeze(1).detach().cpu().tolist())
                    all_labels.extend(labels.squeeze(1).detach().cpu().int().tolist())
            val_accuracy = float(total_correct / total_val) if total_val > 0 else None
            val_loss = float(total_val_loss / total_val) if total_val > 0 else None
            metrics = _compute_binary_metrics(all_probs, all_labels)
            precision = metrics.get("precision")
            recall = metrics.get("recall")
            f1 = metrics.get("f1")
            auc = metrics.get("auc")

        api_client.update_training_state(
            current_round=self.current_round,
            current_epoch=local_epochs,
            total_epochs=local_epochs,
            train_loss=avg_train_loss,
            train_accuracy=avg_train_acc,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auc=auc,
            loss=val_loss if val_loss is not None else avg_train_loss,
            accuracy=val_accuracy if val_accuracy is not None else avg_train_acc,
            status="evaluating",
        )
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"loss": avg_train_loss, "accuracy": val_accuracy}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.valloader:
                inputs = inputs.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)

                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                all_probs.extend(probs.squeeze(1).detach().cpu().tolist())
                all_labels.extend(labels.squeeze(1).detach().cpu().int().tolist())

        accuracy = total_correct / total_samples
        avg_loss = float(total_loss / total_samples)

        metrics = _compute_binary_metrics(all_probs, all_labels)

        api_client.update_training_state(
            loss=avg_loss,
            accuracy=float(accuracy),
            val_loss=avg_loss,
            val_accuracy=float(accuracy),
            precision=metrics.get("precision"),
            recall=metrics.get("recall"),
            f1=metrics.get("f1"),
            auc=metrics.get("auc"),
            current_round=self.current_round,
            status="evaluating",
        )

        return avg_loss, len(self.valloader.dataset), {"accuracy": float(accuracy)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--server_address", type=str, default=None)
    parser.add_argument("--modality", type=str, choices=["audio", "image"], default="audio")
    parser.add_argument("--total_rounds", type=int, default=10, help="Expected total rounds (for dashboard progress)")
    parser.add_argument("--total_epochs", type=int, default=2, help="Local epochs per round")
    args = parser.parse_args()

    if args.server_address is None:
        port = 8080 if args.modality == "audio" else 8081
        args.server_address = f"127.0.0.1:{port}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Khoi dong Client {args.client_id} ({args.modality.upper()}) tren {device}...")

    # ---- FastAPI registration (idempotent by client_name) ----
    client_uuid = api_client.register(task_type=args.modality)
    if client_uuid:
        print(f"Registered with FastAPI: {client_uuid}")
    api_client.update_training_state(
        modality=args.modality,
        total_rounds=args.total_rounds,
        total_epochs=args.total_epochs,
        connected_to_flower=False,
    )

    # ---- Start heartbeat ----
    api_client.start_heartbeat()
    atexit.register(api_client.shutdown)

    # ---- Load model + data ----
    state_path = Path(__file__).resolve().parent.parent / "local_managers" / "fl_state.json"
    if state_path.exists():
        try:
            with state_path.open("r", encoding="utf-8") as f:
                state = json.load(f)
            current_batch = int(state.get("current_batch", 1))
            if current_batch <= 0:
                base_dir = "./fl_worker/fl_data"
            else:
                base_dir = os.path.join("./fl_worker", f"fl_data_{current_batch}")
        except (ValueError, json.JSONDecodeError, OSError):
            base_dir = "./fl_worker/fl_data"
    else:
        base_dir = "./fl_worker/fl_data"

    if args.modality == "audio":
        model = CNN14()
        model = freeze_model_blocks(model).to(device)
        audio_dir = os.path.join(base_dir, "fl_audio")
        train_csv = os.path.join(base_dir, "metadata", "audio_fl", f"client_{args.client_id}_train.csv")
        val_csv = os.path.join(base_dir, "metadata", "audio_fl", f"client_{args.client_id}_val.csv")
        print(f"[Debug] Audio base_dir: {base_dir}")
        print(f"[Debug] Audio dir: {audio_dir}")
        print(f"[Debug] Train CSV: {train_csv}")
        print(f"[Debug] Val CSV: {val_csv}")
        trainloader, valloader = load_client_data(args.client_id, base_dir=base_dir)
    else:
        model = ResNet18(in_channels=3)
        model = freeze_image_blocks(model).to(device)
        trainloader, valloader = load_client_data_image(args.client_id, base_dir=base_dir)

    # ---- Start Flower client (blocking) ----
    api_client.update_training_state(connected_to_flower=True, status="online")
    print(f"Ket noi den Flower server {args.server_address}...")

    try:
        fl.client.start_client(
            server_address=args.server_address,
            client=AdvancedPneumoniaClient(
                args.client_id,
                model,
                trainloader,
                valloader,
                device,
                local_epochs=args.total_epochs,
            ).to_client(),
            grpc_max_message_length=1024 * 1024 * 1024,
        )
    except KeyboardInterrupt:
        print("\nInterrupted. Shutting down...")
    finally:
        api_client.shutdown()
