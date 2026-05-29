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
import json
from pathlib import Path
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


def compute_auprc(all_probs, all_labels, num_classes):
    """Compute per-class and macro Average Precision (AUPRC)."""
    try:
        from sklearn.metrics import average_precision_score
        per_class = {}
        for i in range(num_classes):
            if all_labels[:, i].sum() > 0:
                per_class[str(i)] = float(average_precision_score(all_labels[:, i], all_probs[:, i]))
            else:
                per_class[str(i)] = 0.0
        macro = float(np.mean(list(per_class.values())))
        return macro, per_class
    except ImportError:
        return 0.0, {str(i): 0.0 for i in range(num_classes)}


def compute_f1_precision_recall(all_probs, all_labels, num_classes, threshold=0.5):
    """Compute F1-macro, precision, recall, and confusion matrix."""
    try:
        from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
        preds = (all_probs >= threshold).astype(int)

        per_class_f1 = {}
        per_class_precision = {}
        per_class_recall = {}
        for i in range(num_classes):
            per_class_f1[str(i)] = float(f1_score(all_labels[:, i], preds[:, i], zero_division=0))
            per_class_precision[str(i)] = float(precision_score(all_labels[:, i], preds[:, i], zero_division=0))
            per_class_recall[str(i)] = float(recall_score(all_labels[:, i], preds[:, i], zero_division=0))

        f1_macro = float(np.mean(list(per_class_f1.values())))
        precision_macro = float(np.mean(list(per_class_precision.values())))
        recall_macro = float(np.mean(list(per_class_recall.values())))

        # Confusion matrix for each class
        cm_per_class = {}
        for i in range(num_classes):
            cm = confusion_matrix(all_labels[:, i], preds[:, i])
            cm_per_class[str(i)] = cm.tolist()

        return {
            "f1_macro": f1_macro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "per_class_f1": per_class_f1,
            "per_class_precision": per_class_precision,
            "per_class_recall": per_class_recall,
            "confusion_matrix": cm_per_class,
        }
    except ImportError:
        return {
            "f1_macro": 0.0, "precision_macro": 0.0, "recall_macro": 0.0,
            "per_class_f1": {}, "per_class_precision": {}, "per_class_recall": {},
            "confusion_matrix": {},
        }


class AdvancedPneumoniaClient(fl.client.NumPyClient):
    def __init__(self, client_id, model, trainloader, valloader, device,
                 num_classes=1, lr=1e-4, mu=0.001, local_epochs=2,
                 log_dir: str | None = None):
        self.client_id = client_id
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_classes = num_classes
        self.criterion = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.mu = mu
        self.local_epochs = local_epochs
        self.current_round = 0

        # TensorBoard logging
        self.writer = None
        if log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                import os
                os.makedirs(log_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir=log_dir)
                print(f"[TensorBoard] Logging to {log_dir}")
            except ImportError:
                print("[TensorBoard] tensorboard not installed, skipping")

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

        # Multi-label evaluation: compute per-class AUROC + advanced metrics
        val_auroc_macro = None
        per_class_auroc = {}
        val_auprc_macro = None
        per_class_auprc = {}
        val_f1 = None
        val_precision = None
        val_recall = None
        val_loss = None
        if self.valloader and len(self.valloader.dataset) > 0:
            self.model.eval()
            all_probs_list = []
            all_labels_list = []
            total_val_loss = 0.0
            total_val = 0
            with torch.no_grad():
                for inputs, labels in self.valloader:
                    inputs = inputs.to(self.device)
                    labels = labels.float().to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    total_val_loss += loss.item() * inputs.size(0)
                    total_val += labels.size(0)
                    probs = torch.sigmoid(outputs)
                    all_probs_list.append(probs.cpu().numpy())
                    all_labels_list.append(labels.cpu().numpy())

            all_probs = np.concatenate(all_probs_list, axis=0)
            all_labels_np = np.concatenate(all_labels_list, axis=0)
            val_auroc_macro, per_class_auroc = compute_auroc(all_probs, all_labels_np, self.num_classes)
            val_auprc_macro, per_class_auprc = compute_auprc(all_probs, all_labels_np, self.num_classes)
            f1_data = compute_f1_precision_recall(all_probs, all_labels_np, self.num_classes)
            val_f1 = f1_data["f1_macro"]
            val_precision = f1_data["precision_macro"]
            val_recall = f1_data["recall_macro"]
            val_loss = float(total_val_loss / total_val) if total_val > 0 else None

        # TensorBoard logging
        global_step = self.current_round * self.local_epochs * len(self.trainloader)
        if self.writer is not None:
            self.writer.add_scalar("Loss/train", avg_train_loss, self.current_round)
            self.writer.add_scalar("Accuracy/train", avg_train_acc, self.current_round)
            if val_loss is not None:
                self.writer.add_scalar("Loss/val", val_loss, self.current_round)
            if val_auroc_macro is not None:
                self.writer.add_scalar("AUROC/macro", val_auroc_macro, self.current_round)
            if val_auprc_macro is not None:
                self.writer.add_scalar("AUPRC/macro", val_auprc_macro, self.current_round)
            if val_f1 is not None:
                self.writer.add_scalar("F1/macro", val_f1, self.current_round)
            if val_precision is not None:
                self.writer.add_scalar("Precision/macro", val_precision, self.current_round)
            if val_recall is not None:
                self.writer.add_scalar("Recall/macro", val_recall, self.current_round)
            # Per-class AUROC
            for cls_name, val in per_class_auroc.items():
                self.writer.add_scalar(f"AUROC/class_{cls_name}", val, self.current_round)
            # Global step-based logging
            self.writer.add_scalar("Loss/train_step", avg_train_loss, global_step)
            self.writer.flush()

        api_client.update_training_state(
            current_round=self.current_round,
            current_epoch=local_epochs,
            total_epochs=local_epochs,
            train_loss=avg_train_loss,
            train_accuracy=avg_train_acc,
            val_loss=val_loss,
            val_accuracy=val_auroc_macro,
            loss=val_loss if val_loss is not None else avg_train_loss,
            accuracy=val_auroc_macro if val_auroc_macro is not None else avg_train_acc,
            precision=val_precision,
            recall=val_recall,
            f1=val_f1,
            auc=val_auroc_macro,
            status="evaluating",
        )
        return self.get_parameters(config={}), len(self.trainloader.dataset), {
            "loss": avg_train_loss,
            "auroc_macro": val_auroc_macro,
            "per_class_auroc": per_class_auroc,
            "auprc_macro": val_auprc_macro,
            "per_class_auprc": per_class_auprc,
            "f1_macro": val_f1,
            "precision_macro": val_precision,
            "recall_macro": val_recall,
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
        auprc_macro, per_class_auprc = compute_auprc(all_probs, all_labels_np, self.num_classes)
        f1_data = compute_f1_precision_recall(all_probs, all_labels_np, self.num_classes)

        api_client.update_training_state(
            loss=avg_loss,
            accuracy=auroc_macro,
            val_loss=avg_loss,
            val_accuracy=float(auroc_macro),
            precision=f1_data["precision_macro"],
            recall=f1_data["recall_macro"],
            f1=f1_data["f1_macro"],
            auc=float(auroc_macro),
            current_round=self.current_round,
            status="evaluating",
        )

        return avg_loss, len(self.valloader.dataset), {
            "auroc_macro": float(auroc_macro),
            "per_class_auroc": per_class_auroc,
            "auprc_macro": float(auprc_macro),
            "per_class_auprc": per_class_auprc,
            "f1_macro": f1_data["f1_macro"],
            "precision_macro": f1_data["precision_macro"],
            "recall_macro": f1_data["recall_macro"],
            "confusion_matrix": f1_data["confusion_matrix"],
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


def _resolve_base_dir() -> str:
    """Resolve FL data directory based on current batch from fl_state.json."""
    state_path = Path(__file__).resolve().parent.parent / "local_managers" / "fl_state.json"
    if state_path.exists():
        try:
            with state_path.open("r", encoding="utf-8") as f:
                state = json.load(f)
            current_batch = int(state.get("current_batch", 1))
            if current_batch <= 0:
                return "./fl_worker/fl_data"
            else:
                return os.path.join("./fl_worker", f"fl_data_{current_batch}")
        except (ValueError, json.JSONDecodeError, OSError):
            pass
    return "./fl_worker/fl_data"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--server_address", type=str, default=None)
    parser.add_argument("--modality", type=str, choices=["audio", "image", "alignment"], default="audio")
    parser.add_argument("--total_rounds", type=int, default=10)
    parser.add_argument("--total_epochs", type=int, default=2, help="Local epochs per round")
    parser.add_argument("--log_dir", type=str, default=None, help="TensorBoard log directory")
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
        total_epochs=args.total_epochs,
        connected_to_flower=False,
        current_round=0,
        status="connecting",
    )

    base_dir = _resolve_base_dir()

    # TensorBoard log dir
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = os.path.join(base_dir, "tensorboard", f"client_{args.client_id}_{args.modality}")

    if args.modality == "alignment":
        from shared.prototype_fl_model import FLPrototypeModel

        server_flower = os.path.join(_server_path, 'flower_server')
        prototype_model = FLPrototypeModel(
            image_pretrained_path=os.path.join(server_flower, 'pretrained_xray_multilabel.pth'),
            audio_pretrained_path=os.path.join(server_flower, 'pretrained_audio_multilabel.pth'),
        ).to(device)

        image_loader, _ = load_client_data_image(args.client_id, base_dir=base_dir)
        audio_loader, _ = load_client_data(args.client_id, base_dir=base_dir)

        client = PrototypeAlignmentClient(
            args.client_id, prototype_model, image_loader, audio_loader, device
        ).to_client()
    elif args.modality == "audio":
        model = ASTMultiLabel(num_classes=2)
        model = freeze_early_blocks(model).to(device)
        audio_dir = os.path.join(base_dir, "fl_audio")
        train_csv = os.path.join(base_dir, "metadata", "audio_fl", f"client_{args.client_id}_train.csv")
        val_csv = os.path.join(base_dir, "metadata", "audio_fl", f"client_{args.client_id}_val.csv")
        print(f"[Debug] Audio base_dir: {base_dir}")
        print(f"[Debug] Audio dir: {audio_dir}")
        print(f"[Debug] Train CSV: {train_csv}")
        print(f"[Debug] Val CSV: {val_csv}")
        trainloader, valloader = load_client_data(args.client_id, base_dir=base_dir)
        num_classes = 2
        client = AdvancedPneumoniaClient(
            args.client_id, model, trainloader, valloader, device,
            num_classes=num_classes, local_epochs=args.total_epochs, log_dir=log_dir
        ).to_client()
    else:
        model = DenseNet121MultiLabel(num_classes=3)
        model = freeze_backbone(model).to(device)
        trainloader, valloader = load_client_data_image(args.client_id, base_dir=base_dir)
        num_classes = 3
        client = AdvancedPneumoniaClient(
            args.client_id, model, trainloader, valloader, device,
            num_classes=num_classes, local_epochs=args.total_epochs, log_dir=log_dir
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
