"""Resume the pinned Flowers-102 comparison from a Spot checkpoint."""

import os
import random
from pathlib import Path

import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from upstream import flowers_comparison as upstream


SCHEMA = 1
SOURCE_REVISION = "9d317e629428d73d92b88dc501f0dce3b1017c86"
MODEL_REVISION = "8a59acfc5285e72257e8d322da51bf486c3ce060"
MODEL_NAMES = (
    "torchvision/resnet-18",
    "perforated-ai/resnet-18-perforated-cascor",
)
CHECKPOINT_NAME = "flowers-comparison.pt"


def run_config(args):
    return {
        "source_revision": SOURCE_REVISION,
        "model_revision": MODEL_REVISION,
        "epochs": args.epochs,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "test_batch_size": args.test_batch_size,
        "lr": args.lr,
        "dry_run": args.dry_run,
        "save_model": args.save_model,
    }


def checkpoint_path():
    directory = Path("/mnt/checkpoint")
    if not directory.is_dir() or not os.access(directory, os.W_OK):
        raise RuntimeError("Spot resume requires a writable /mnt/checkpoint mount")
    return directory / CHECKPOINT_NAME


def read_checkpoint(path, config):
    if not path.exists():
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise RuntimeError(f"Cannot read Spot checkpoint {path}") from error
    if not isinstance(state, dict) or state.get("schema") != SCHEMA:
        raise ValueError("Spot checkpoint schema is invalid")
    if state.get("config") != config:
        raise ValueError("Spot checkpoint does not match this comparison")
    model_name = state.get("model_name")
    epoch = state.get("epoch")
    history = state.get("accuracy")
    if (
        model_name not in MODEL_NAMES
        or not isinstance(epoch, int)
        or not 1 <= epoch <= config["epochs"]
        or not isinstance(history, list)
        or len(history) != epoch
    ):
        raise ValueError("Spot checkpoint progress is invalid")
    if model_name == MODEL_NAMES[1]:
        baseline = state.get("completed", {}).get(MODEL_NAMES[0], {})
        if len(baseline.get("accuracy", [])) != config["epochs"]:
            raise ValueError("Spot checkpoint is missing the baseline result")
    for key in (
        "last_loss",
        "completed",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "python_rng",
        "torch_rng",
        "cuda_rng",
    ):
        if key not in state:
            raise ValueError(f"Spot checkpoint is missing {key}")
    return state


def write_checkpoint(path, state):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as output:
        torch.save(state, output)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def restore_rng(state):
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("Spot checkpoint requires CUDA RNG state")
        torch.cuda.set_rng_state_all(state["cuda_rng"])


def run_training(args, device, train_loader, test_loader, model_name, model_builder):
    """Use upstream train/test functions and save the state after each epoch."""
    path = checkpoint_path()
    config = run_config(args)
    state = read_checkpoint(path, config)
    model_index = MODEL_NAMES.index(model_name)

    if state is not None:
        saved_index = MODEL_NAMES.index(state["model_name"])
        if saved_index > model_index:
            result = state["completed"].get(model_name)
            if result is None:
                raise ValueError(f"Spot checkpoint is missing {model_name}")
            print(f"CHECKPOINT_SKIP model={model_name} epochs={args.epochs}", flush=True)
            return result["last_loss"], result["accuracy"]
        if saved_index < model_index and state["epoch"] != args.epochs:
            raise ValueError("Spot checkpoint changed models before the final epoch")
        if saved_index == model_index and state["epoch"] == args.epochs:
            restore_rng(state)
            print(f"CHECKPOINT_SKIP model={model_name} epochs={args.epochs}", flush=True)
            return state["last_loss"], state["accuracy"]
    elif model_index != 0:
        raise ValueError("Spot checkpoint is missing the baseline result")

    print(f"\n===== Running model: {model_name} =====")
    if state is not None and state["model_name"] != model_name:
        restore_rng(state)
    model = model_builder(upstream.NUM_CLASSES).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0)

    completed = {} if state is None else dict(state["completed"])
    if state is not None and state["model_name"] != model_name:
        completed[state["model_name"]] = {
            "last_loss": state["last_loss"],
            "accuracy": state["accuracy"],
        }

    start_epoch = 1
    last_loss = None
    history = []
    if state is not None and state["model_name"] == model_name:
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        for parameter_state in optimizer.state.values():
            for key, value in parameter_state.items():
                if isinstance(value, torch.Tensor):
                    parameter_state[key] = value.to(device)
        scheduler.load_state_dict(state["scheduler_state"])
        restore_rng(state)
        start_epoch = state["epoch"] + 1
        last_loss = state["last_loss"]
        history = list(state["accuracy"])
        print(f"CHECKPOINT_RESTORE model={model_name} epoch={state['epoch']}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        upstream.train(args, model, device, train_loader, optimizer, epoch)
        last_loss, accuracy = upstream.test(model, device, test_loader)
        history.append(accuracy)
        scheduler.step()
        payload = {
            "schema": SCHEMA,
            "config": config,
            "model_name": model_name,
            "epoch": epoch,
            "last_loss": last_loss,
            "accuracy": history,
            "completed": completed,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        write_checkpoint(path, payload)
        print(f"CHECKPOINT_SAVE model={model_name} epoch={epoch}", flush=True)

    if args.save_model:
        output_name = model_name.replace("/", "_").replace("-", "_") + ".pt"
        torch.save(model.state_dict(), output_name)
    return last_loss, history
