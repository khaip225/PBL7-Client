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
import numpy as np
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add PBL7-Server path for shared modules
_server_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'PBL7-Server'))
if os.path.exists(_server_path) and _server_path not in sys.path:
    sys.path.insert(0, _server_path)

from ai_engines.audio_engine.ast_model import ASTMultiLabel, freeze_early_blocks
from ai_engines.image_engine.densenet121_model import DenseNet121MultiLabel, freeze_backbone
from fl_worker.api_client import api_client


def compute_auroc(all_probs, all_labels, num_classes):
    """Compute per-class and macro AUROC."""
    try:
        from sklearn.metrics import roc_auc_score
        per_class = {}
        for i in range(num_classes):
            if all_labels[:, i].sum() > 0 and (all_labels[:, i].sum() < len(all_labels)):
                per_class[str(i)] = float(roc_auc_score(all_labels[:, i], all_probs[:, i]))
            else:
                per_class[str(i)] = 0.5
        macro = float(np.mean(list(per_class.values())))
        return macro, per_class
    except ImportError:
        return 0.5, {str(i): 0.5 for i in range(num_classes)}


class AdvancedPneumoniaClient(fl.client.NumPyClient):
    def __init__(self, client_id, model, trainloader, valloader, device,
                 num_classes=1, lr=1e-4, mu=0.001):
        self.client_id = client_id
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_classes = num_classes
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

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr, weight_decay=1e-4
        )

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
                labels = labels.float().to(self.device)

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

        # Multi-label evaluation: compute per-class AUROC
        val_auroc_macro = None
        per_class_auroc = {}
        if self.valloader and len(self.valloader.dataset) > 0:
            self.model.eval()
            all_probs_list = []
            all_labels_list = []
            with torch.no_grad():
                for inputs, labels in self.valloader:
                    inputs = inputs.to(self.device)
                    labels = labels.float().to(self.device)
                    outputs = self.model(inputs)
                    probs = torch.sigmoid(outputs)
                    all_probs_list.append(probs.cpu().numpy())
                    all_labels_list.append(labels.cpu().numpy())

            all_probs = np.concatenate(all_probs_list, axis=0)
            all_labels_np = np.concatenate(all_labels_list, axis=0)
            val_auroc_macro, per_class_auroc = compute_auroc(all_probs, all_labels_np, self.num_classes)

        api_client.update_training_state(
            current_round=self.current_round,
            loss=avg_train_loss,
            accuracy=val_auroc_macro,
        )
        return self.get_parameters(config={}), len(self.trainloader.dataset), {
            "loss": avg_train_loss,
            "auroc_macro": val_auroc_macro,
            "per_class_auroc": per_class_auroc,
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_probs_list = []
        all_labels_list = []

        with torch.no_grad():
            for inputs, labels in self.valloader:
                inputs = inputs.to(self.device)
                labels = labels.float().to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)

                probs = torch.sigmoid(outputs)
                all_probs_list.append(probs.cpu().numpy())
                all_labels_list.append(labels.cpu().numpy())
                total_samples += labels.size(0)

        avg_loss = float(total_loss / total_samples)

        all_probs = np.concatenate(all_probs_list, axis=0)
        all_labels_np = np.concatenate(all_labels_list, axis=0)
        auroc_macro, per_class_auroc = compute_auroc(all_probs, all_labels_np, self.num_classes)

        api_client.update_training_state(
            loss=avg_loss,
            accuracy=auroc_macro,
            current_round=self.current_round,
            status="evaluating",
        )

        return avg_loss, len(self.valloader.dataset), {
            "auroc_macro": float(auroc_macro),
            "per_class_auroc": per_class_auroc,
        }


class PrototypeAlignmentClient(fl.client.NumPyClient):
    """FL client for prototype-only alignment training."""

    def __init__(self, client_id, prototype_model, image_loader, audio_loader, device, lr=1e-4):
        self.client_id = client_id
        self.prototype_model = prototype_model
        self.image_loader = image_loader
        self.audio_loader = audio_loader
        self.device = device
        self.lr = lr
        self.current_round = 0

        from shared.prototypes import (
            supervised_contrastive_loss, prototype_consistency_loss,
            ontology_regularization_loss, EmbeddingMemoryBank,
        )
        self.sc_loss = supervised_contrastive_loss
        self.pc_loss = prototype_consistency_loss
        self.or_loss = ontology_regularization_loss
        self.memory_bank = EmbeddingMemoryBank(dim=256, queue_size=512)

    def get_parameters(self, config):
        sd = self.prototype_model.shareable_state_dict()
        return [v.cpu().numpy() for v in sd.values()]

    def set_parameters(self, parameters):
        keys = list(self.prototype_model.shareable_state_dict().keys())
        sd = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        self.prototype_model.load_shareable_state_dict(sd)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.prototype_model.train()
        self.current_round += 1

        optimizer = optim.AdamW(
            self.prototype_model.shareable_parameters(), lr=self.lr, weight_decay=1e-4
        )
        temperature = 0.07

        total_loss = 0.0
        n_batches = 0

        for (img_batch, img_labels), (aud_batch, aud_labels) in zip(
            self.image_loader, self.audio_loader
        ):
            img_batch = img_batch.to(self.device)
            img_labels = img_labels.float().to(self.device)
            aud_batch = aud_batch.to(self.device)
            aud_labels = aud_labels.float().to(self.device)

            optimizer.zero_grad()

            img_emb, aud_emb = self.prototype_model.get_embeddings(img_batch, aud_batch)
            disease_protos = self.prototype_model.disease_protos()
            acoustic_protos = self.prototype_model.acoustic_protos()

            loss_img = self.sc_loss(img_emb, img_labels, disease_protos, temperature)
            loss_aud = self.sc_loss(aud_emb, aud_labels, acoustic_protos, temperature)
            loss_proto = self.pc_loss(disease_protos, acoustic_protos)
            loss_onto = self.or_loss(disease_protos, acoustic_protos)

            total = loss_img + loss_aud + 0.3 * loss_proto + 0.2 * loss_onto
            total.backward()
            optimizer.step()

            self.memory_bank.enqueue_image(img_emb, img_labels)
            self.memory_bank.enqueue_audio(aud_emb, aud_labels)

            total_loss += total.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        api_client.update_training_state(
            current_round=self.current_round,
            loss=avg_loss,
        )
        return self.get_parameters(config={}), n_batches, {
            "loss": avg_loss,
            "loss_img": loss_img.item(),
            "loss_aud": loss_aud.item(),
            "loss_proto": loss_proto.item(),
            "loss_onto": loss_onto.item(),
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, 1, {"loss": 0.0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--server_address", type=str, default=None)
    parser.add_argument("--modality", type=str, choices=["audio", "image", "alignment"], default="audio")
    parser.add_argument("--total_rounds", type=int, default=10)
    args = parser.parse_args()

    if args.server_address is None:
        port_map = {"audio": 8080, "image": 8081, "alignment": 8082}
        port = port_map.get(args.modality, 8080)
        args.server_address = f"127.0.0.1:{port}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Khoi dong Client {args.client_id} ({args.modality.upper()}) tren {device}...")

    api_client.update_training_state(
        modality=args.modality,
        total_rounds=args.total_rounds,
        connected_to_flower=False,
        current_round=0,
        status="connecting",
    )

    if args.modality == "alignment":
        from prototype_fl_model import FLPrototypeModel

        server_flower = os.path.join(_server_path, 'flower_server')
        prototype_model = FLPrototypeModel(
            image_pretrained_path=os.path.join(server_flower, 'pretrained_xray_multilabel.pth'),
            audio_pretrained_path=os.path.join(server_flower, 'pretrained_audio_multilabel.pth'),
        ).to(device)

        image_loader, _ = load_client_data_image(args.client_id, base_dir="./fl_worker/fl_data")
        audio_loader, _ = load_client_data(args.client_id, base_dir="./fl_worker/fl_data")

        client = PrototypeAlignmentClient(
            args.client_id, prototype_model, image_loader, audio_loader, device
        ).to_client()
    elif args.modality == "audio":
        model = ASTMultiLabel(num_classes=2)
        model = freeze_early_blocks(model).to(device)
        trainloader, valloader = load_client_data(args.client_id, base_dir="./fl_worker/fl_data")
        num_classes = 2
        client = AdvancedPneumoniaClient(
            args.client_id, model, trainloader, valloader, device, num_classes=num_classes
        ).to_client()
    else:
        model = DenseNet121MultiLabel(num_classes=3)
        model = freeze_backbone(model).to(device)
        trainloader, valloader = load_client_data_image(args.client_id, base_dir="./fl_worker/fl_data")
        num_classes = 3
        client = AdvancedPneumoniaClient(
            args.client_id, model, trainloader, valloader, device, num_classes=num_classes
        ).to_client()

    api_client.update_training_state(connected_to_flower=True, status="training")
    print(f"Ket noi den Flower server {args.server_address}...")

    try:
        fl.client.start_client(
            server_address=args.server_address,
            client=client,
            grpc_max_message_length=1024 * 1024 * 1024,
        )
    except KeyboardInterrupt:
        print("\nInterrupted. Shutting down...")
    finally:
        api_client.update_training_state(
            connected_to_flower=False,
            training_active=False,
            status="online",
        )
