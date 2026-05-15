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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ai_engines.audio_engine.cnn14_model import CNN14, freeze_model_blocks
from ai_engines.image_engine.resnet18_model import ResNet18, freeze_model_blocks as freeze_image_blocks
from fl_worker.api_client import api_client


class AdvancedPneumoniaClient(fl.client.NumPyClient):
    def __init__(self, client_id, model, trainloader, valloader, device, lr=1e-4, mu=0.001):
        self.client_id = client_id
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.mu = mu
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

        local_epochs = 2
        total_steps = len(self.trainloader) * local_epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        api_client.update_training_state(current_round=self.current_round, status="training")

        total_train_loss = 0.0
        total_train_samples = 0
        for epoch in range(local_epochs):
            print(f"\n--- Epoch {epoch+1}/{local_epochs} ---")
            progress_bar = tqdm(self.trainloader, desc="Training")

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
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / total_train_samples if total_train_samples > 0 else 0.0

        # Quick evaluation after training for accuracy metric
        val_accuracy = None
        if self.valloader and len(self.valloader.dataset) > 0:
            self.model.eval()
            total_correct, total_val = 0, 0
            with torch.no_grad():
                for inputs, labels in self.valloader:
                    inputs = inputs.to(self.device)
                    labels = labels.float().unsqueeze(1).to(self.device)
                    outputs = self.model(inputs)
                    probs = torch.sigmoid(outputs)
                    preds = (probs >= 0.5).float()
                    total_correct += (preds == labels).sum().item()
                    total_val += labels.size(0)
            val_accuracy = float(total_correct / total_val) if total_val > 0 else None

        api_client.update_training_state(
            current_round=self.current_round,
            loss=avg_train_loss,
            accuracy=val_accuracy,
        )
        return self.get_parameters(config={}), len(self.trainloader.dataset), {"loss": avg_train_loss, "accuracy": val_accuracy}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

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

        accuracy = total_correct / total_samples
        avg_loss = float(total_loss / total_samples)

        api_client.update_training_state(
            loss=avg_loss,
            accuracy=float(accuracy),
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
        connected_to_flower=False,
    )

    # ---- Start heartbeat ----
    api_client.start_heartbeat()
    atexit.register(api_client.shutdown)

    # ---- Load model + data ----
    if args.modality == "audio":
        model = CNN14()
        model = freeze_model_blocks(model).to(device)
        trainloader, valloader = load_client_data(args.client_id, base_dir="./fl_worker/fl_data")
    else:
        model = ResNet18(in_channels=3)
        model = freeze_image_blocks(model).to(device)
        trainloader, valloader = load_client_data_image(args.client_id, base_dir="./fl_worker/fl_data")

    # ---- Start Flower client (blocking) ----
    api_client.update_training_state(connected_to_flower=True, status="online")
    print(f"Ket noi den Flower server {args.server_address}...")

    try:
        fl.client.start_client(
            server_address=args.server_address,
            client=AdvancedPneumoniaClient(args.client_id, model, trainloader, valloader, device).to_client(),
            grpc_max_message_length=1024 * 1024 * 1024,
        )
    except KeyboardInterrupt:
        print("\nInterrupted. Shutting down...")
    finally:
        api_client.shutdown()
