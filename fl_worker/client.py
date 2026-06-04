"""Prototype-Guided Federated Learning Client — Flower Integration.

Supports 3 modalities:
  - image:       DenseNet121 encoder + image prototypes
  - audio:       AST encoder + audio prototypes
  - multimodal:  both encoders + all prototypes

Shares only prototypes + projection heads (NOT backbone / raw data).
Uses selective FedAvg: image params aggregated from image/multimodal clients,
audio params from audio/multimodal clients, prototypes from all.

Design matched to notebook pbl7-fl.ipynb (class FederatedClient).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import flwr as fl

# Path setup
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Server path must come AFTER client path for correct shared module resolution
_server_path = os.path.abspath(os.path.join(_project_root, "..", "PBL7-Server"))
if os.path.exists(_server_path) and _server_path not in sys.path:
    sys.path.append(_server_path)  # append, not insert(0)

from shared.encoder_models import DenseNet121Encoder, ASTEncoder
from shared.momentum_prototype import MomentumPrototypeModule, MemoryBank
from shared.prototype_fl_model import (
    FL_CONFIG,
    contrastive_loss,
    proto_consistency_loss,
    ontology_range_loss,
    normal_bridge_loss,
    triplet_loss,
    cross_modal_loss,
)

from fl_worker.dataset_loader import (
    load_image_data,
    load_audio_data,
    load_multimodal_data,
    _resolve_fl_base_dir,
)

from fl_worker.api_client import api_client


# ═══════════════════════════════════════════════════════════════════════════
# Flower NumPyClient wrapper
# ═══════════════════════════════════════════════════════════════════════════

class PrototypeFlowerClient(fl.client.NumPyClient):
    """Flower client wrapping PrototypeFLClient logic.

    This is a thin wrapper that translates between Flower's NumPyClient
    interface and the PrototypeFLClient's state management.
    """

    def __init__(
        self,
        proto_client,           # PrototypeFLClient instance
        api_client=None,        # FastAPIClient for state reporting
        total_rounds: int = 10,
    ):
        self.pc = proto_client            # PrototypeFLClient
        self.api = api_client
        self.total_rounds = total_rounds
        self._round = 0

    def get_parameters(self, config: dict) -> list[np.ndarray]:
        """Return shareable params as list of numpy arrays."""
        state = self.pc.get_sync_state()
        return [v.cpu().numpy() for v in state.values()]

    def set_parameters(self, parameters: list[np.ndarray]):
        """Load aggregated params from server (key-based matching).

        When the server sends back parameters after selective FedAvg, the order
        matches the initial parameter keys (img_proj.*, aud_proj.*, proto.*).
        We use key-based matching so the set_weights mapping is always correct,
        regardless of whether an image-only, audio-only, or multimodal client
        calls this.
        """
        server_keys = list(self.pc.get_sync_state().keys())
        if len(parameters) != len(server_keys):
            print(f"[WARNING] Parameter count mismatch: got {len(parameters)}, expected {len(server_keys)}")
            # Fall back to position-based mapping with truncation
            min_len = min(len(parameters), len(server_keys))
            state = OrderedDict({k: torch.tensor(v) for k, v in zip(server_keys[:min_len], parameters[:min_len])})
        else:
            state = OrderedDict({k: torch.tensor(v) for k, v in zip(server_keys, parameters)})
        self.pc.set_weights(state)

    # --- fit: local training ---
    def fit(self, parameters: list[np.ndarray], config: dict) -> tuple:
        self._round += 1
        print(f"\n{'='*60}")
        print(f"🔄 [ROUND {self._round}/{self.total_rounds}] Client {self.pc.client_id} ({self.pc.modality.upper()}) — FIT")
        print(f"{'='*60}")

        self.set_parameters(parameters)

        # Extract config overrides from server
        local_epochs = int(config.get("local_epochs", 1))

        # Report to GUI
        if self.api:
            self.api.update_training_state(
                current_round=self._round,
                current_epoch=0,
                total_epochs=local_epochs,
                total_rounds=self.total_rounds,
                status="training",
            )

        t_start = time.perf_counter()
        avg_loss = self.pc.train(self.pc._global_params_cache if hasattr(self.pc, '_global_params_cache') else OrderedDict(), local_epochs=local_epochs)
        train_time = time.perf_counter() - t_start

        print(f"  ✅ Training complete — Loss: {avg_loss:.4f}, Time: {train_time:.1f}s")
        print(f"  📤 Sending {len(self.pc.get_sync_state())} tensors (prototypes + projection heads)")

        if self.api:
            self.api.update_training_state(
                current_round=self._round,
                current_epoch=local_epochs,
                total_epochs=local_epochs,
                loss=avg_loss,
                train_loss=avg_loss,
                status="evaluating",
            )

        # NOTE: return the CURRENT local weights, not the global state
        # This is what gets aggregated on the server
        params = self.get_parameters(config)
        return (
            params,
            self.pc.num_samples,
            {
                "loss": float(avg_loss),
                "modality": self.pc.modality,
                "client_id": self.pc.client_id,
                "num_samples": self.pc.num_samples,
                "train_time_s": round(train_time, 2),
            },
        )

    # --- evaluate: cross-modal retrieval evaluation ---
    def evaluate(self, parameters: list[np.ndarray], config: dict) -> tuple:
        """Simple evaluation — returns loss placeholder.

        Full evaluation (cross-modal retrieval mAP) is done server-side
        after aggregation for global metrics.
        """
        self.set_parameters(parameters)

        # Placeholder evaluation
        val_loss = 0.0
        val_samples = 1

        if self.api:
            self.api.update_training_state(
                val_loss=val_loss,
                status="evaluating",
            )

        return float(val_loss), val_samples, {"loss": val_loss}

    def get_properties(self, config: dict) -> dict:
        return {"modality": self.pc.modality}


# ═══════════════════════════════════════════════════════════════════════════
# Client builder
# ═══════════════════════════════════════════════════════════════════════════

def build_proto_client(
    client_id: str,
    modality: str,
    base_dir: str,
    device: str = "cpu",
    batch_size_img: int = 32,
    batch_size_aud: int = 16,
    lr: float = 5e-6,
) -> PrototypeFLClient:
    """Build PrototypeFLClient with appropriate data loaders for the modality."""
    from shared.prototype_fl_model import PrototypeFLClient

    print(f"\n  📦 Building {modality.upper()} client '{client_id}'...")

    if modality == "image":
        train_loader, val_loader = load_image_data(
            int(client_id), batch_size=batch_size_img, base_dir=base_dir,
        )
        proto_cl = PrototypeFLClient(
            client_id=client_id,
            modality="image",
            img_loader=train_loader,
            aud_loader=None,
            device=device,
            lr=lr,
        )

    elif modality == "audio":
        train_loader, val_loader = load_audio_data(
            int(client_id), batch_size=batch_size_aud, base_dir=base_dir,
        )
        proto_cl = PrototypeFLClient(
            client_id=client_id,
            modality="audio",
            img_loader=None,
            aud_loader=train_loader,
            device=device,
            lr=lr,
        )

    elif modality == "alignment" or modality == "multimodal":
        (img_train, img_val), (aud_train, aud_val) = load_multimodal_data(
            int(client_id), batch_size_img, batch_size_aud, base_dir=base_dir,
        )
        proto_cl = PrototypeFLClient(
            client_id=client_id,
            modality="multimodal",
            img_loader=img_train,
            aud_loader=aud_train,
            device=device,
            lr=lr,
        )
    else:
        raise ValueError(f"Unknown modality: {modality}")

    return proto_cl


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prototype-Guided FL Client (Flower)")
    parser.add_argument("--client_id", type=str, required=True,
                        help="Client identifier (e.g., '1', 'A')")
    parser.add_argument("--modality", type=str, default="image",
                        choices=["image", "audio", "alignment", "multimodal"])
    parser.add_argument("--server_address", type=str, default=None,
                        help="Flower server address (host:port)")
    parser.add_argument("--total_rounds", type=int, default=10)
    parser.add_argument("--total_epochs", type=int, default=1,
                        help="Local epochs per round")
    parser.add_argument("--batch_size_img", type=int, default=32)
    parser.add_argument("--batch_size_aud", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Override FL data directory")
    args = parser.parse_args()

    # --- Resolve server address ---
    if args.server_address is None:
        port_map = {
            "image":      8081,
            "audio":      8080,
            "alignment":  8082,
            "multimodal": 8082,
        }
        port = port_map.get(args.modality, 8081)
        args.server_address = f"127.0.0.1:{port}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Prototype FL Client {args.client_id} ({args.modality.upper()}) on {device}")

    # --- Resolve data directory ---
    base_dir = args.base_dir if args.base_dir else _resolve_fl_base_dir()

    # --- Report to API ---
    api_client.update_training_state(
        modality=args.modality,
        total_rounds=args.total_rounds,
        total_epochs=args.total_epochs,
        connected_to_flower=False,
        current_round=0,
        status="connecting",
    )
    # Register with server
    api_client.register(task_type=args.modality)
    api_client.start_heartbeat()

    # --- Build client ---
    try:
        proto_cl = build_proto_client(
            client_id=args.client_id,
            modality=args.modality if args.modality != "alignment" else "multimodal",
            base_dir=base_dir,
            device=str(device),
            batch_size_img=args.batch_size_img,
            batch_size_aud=args.batch_size_aud,
            lr=args.lr,
        )

        # Initialize global state (random init for first round)
        global_state = proto_cl.get_sync_state()
        proto_cl.set_weights(global_state)

        flower_client = PrototypeFlowerClient(
            proto_client=proto_cl,
            api_client=api_client,
            total_rounds=args.total_rounds,
        )

    except Exception as e:
        print(f"❌ Failed to build client: {e}")
        import traceback
        traceback.print_exc()
        api_client.update_training_state(status="error")
        sys.exit(1)

    # --- Start Flower ---
    api_client.update_training_state(connected_to_flower=True, status="training")
    print(f"\n🔗 Connecting to Flower server at {args.server_address}...")
    print(f"   Task: {args.modality.upper()}")
    print(f"   Rounds: {args.total_rounds}")
    print(f"   Local epochs/round: {args.total_epochs}")
    print(f"   Samples: {proto_cl.num_samples}")
    print(f"   Device: {device}")
    print(f"   Syncing: prototypes ({7} tensors) + projection heads")
    print(f"{'='*60}\n")

    try:
        fl.client.start_client(
            server_address=args.server_address,
            client=flower_client,
            grpc_max_message_length=1024 * 1024 * 1024,  # 1GB
        )
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Flower client error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        api_client.update_training_state(
            connected_to_flower=False,
            training_active=False,
            status="online",
        )
        print("👋 Client shutdown complete")
