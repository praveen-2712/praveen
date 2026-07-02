"""
train_yolo.py — Neurologix Pro V3 (Optimized)
===============================================
Fine-tunes YOLO11m on the BraTS-derived bounding-box dataset.
Target: mAP@0.5 >= 0.75

Training schedule (research-backed):
  Phase 1 — 25 epochs, backbone frozen:  detection head converges on MRI features
  Phase 2 — 75 epochs, full fine-tuning: domain adaptation; early stopping (patience=10)
                                          will typically trigger around epoch 60-80
  Total max: 100 epochs
"""

import os
import shutil
import warnings
import logging
import argparse
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")

# ── Silence YOLO's internal per-batch stdout entirely ─────────────────────────
# ultralytics uses Python's logging module internally; setting it to WARNING
# stops the per-batch lines that fight with tqdm even when verbose=False.
logging.getLogger("ultralytics").setLevel(logging.WARNING)

from ultralytics import YOLO
from tqdm import tqdm


# ── Configuration ──────────────────────────────────────────────────────────────

YAML_PATH         = os.path.abspath("./data/yolo/dataset.yaml")
WEIGHTS_DIR       = os.path.abspath("./weights")
FINAL_WEIGHT_NAME = "detector_yolo.pt"
PROJECT_DIR       = os.path.abspath("./models/yolo_runs")
RUN_NAME          = "tumor_detector"

PHASE1_EPOCHS  = 25   # raised from 15 — head needs more iterations to converge on MRI features
PHASE1_FREEZE  = 10
PHASE2_EPOCHS  = 75   # raised from 35 — early stopping (patience=10) will cut short if needed
TARGET_MAP     = 0.75


# ── Callbacks ──────────────────────────────────────────────────────────────────

def make_callbacks(phase_label: str, total_phase_epochs: int):
    """
    One clean tqdm bar per epoch.  YOLO's own stdout is silenced via the
    logging config above, so nothing fights with the bar.
    """
    pbar_state = {"bar": None, "epoch_num": 0}

    def on_train_start(trainer):
        pbar_state["epoch_num"] = 1
        pbar_state["bar"] = tqdm(
            total=len(trainer.train_loader),
            desc=f"[{phase_label}] Epoch 1/{total_phase_epochs}",
            dynamic_ncols=True,
            leave=True,
        )

    def on_train_batch_end(trainer):
        bar = pbar_state["bar"]
        if bar is None:
            return
        bar.update(1)
        try:
            items = trainer.loss_items
            loss_str = " | ".join(f"{float(x):.3f}" for x in items)
            bar.set_postfix({"loss": loss_str}, refresh=False)
        except Exception:
            pass

    def on_train_epoch_end(trainer):
        bar = pbar_state["bar"]
        if bar is not None:
            bar.close()
            pbar_state["bar"] = None

        epoch_done = trainer.epoch
        next_epoch = epoch_done + 2
        if next_epoch <= total_phase_epochs:
            pbar_state["epoch_num"] = next_epoch
            pbar_state["bar"] = tqdm(
                total=len(trainer.train_loader),
                desc=f"[{phase_label}] Epoch {next_epoch}/{total_phase_epochs}",
                dynamic_ncols=True,
                leave=True,
            )

    def on_val_end(validator):
        bar = pbar_state["bar"]
        if bar is not None:
            bar.clear()
        try:
            map50 = validator.metrics.box.map50
            tqdm.write(
                f"  ↳ [{phase_label}] Epoch {pbar_state['epoch_num']-1}"
                f"/{total_phase_epochs}  mAP@0.5: {map50:.4f}"
                f"  (target >= {TARGET_MAP})"
            )
        except Exception:
            pass

    return {
        "on_train_start":     on_train_start,
        "on_train_batch_end": on_train_batch_end,
        "on_train_epoch_end": on_train_epoch_end,
        "on_val_end":         on_val_end,
    }


# ── Shared training kwargs ─────────────────────────────────────────────────────

def base_train_kwargs(epochs: int, freeze: int = 0) -> dict:
    return dict(
        data=YAML_PATH,
        epochs=epochs,
        imgsz=640,
        batch=16,
        freeze=freeze,
        patience=10,
        augment=True,
        mosaic=1.0,
        mixup=0.15,
        cos_lr=True,
        warmup_epochs=3,
        label_smoothing=0.1,
        overlap_mask=False,
        save=True,
        workers=2,
        device=0,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        verbose=False,
        plots=True,
    )


# ── Helper: attach & detach callbacks cleanly ──────────────────────────────────

def run_phase(model: YOLO, phase_label: str, epochs: int, freeze: int):
    """Register callbacks, train, then clear them so Phase 2 gets fresh bars."""
    cbs = make_callbacks(phase_label, epochs)
    for event, fn in cbs.items():
        model.add_callback(event, fn)

    model.train(**base_train_kwargs(epochs, freeze=freeze))

    # Clear callbacks before next phase to avoid double-firing
    for event in cbs:
        model.callbacks[event] = []


# ── Main ───────────────────────────────────────────────────────────────────────

def train(resume: bool = False):
    print(f"[YOLO] Initialising YOLO11m fine-tuning (two-phase). Resume={resume}\n")

    if not os.path.exists(YAML_PATH):
        print(f"[ERROR] {YAML_PATH} not found. Run scripts/generate_yolo_labels.py first.")
        return

    last_pt = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "last.pt")
    p1_best = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")
    if resume and os.path.exists(last_pt):
        print(f"[YOLO] Resuming from {last_pt}")
        model = YOLO(last_pt)
        # Ultralytics natively supports resume=True which loads epoch, optimizer, etc.
        model.train(resume=True)
    else:
        if resume:
            print(f"[YOLO] Resume requested but {last_pt} not found. Starting fresh.")
        model = YOLO("yolo11m.pt")

    # Phase 1: frozen backbone
    if not os.path.exists(p1_best):
        print(f"Phase 1 - Detection head only  "
              f"({PHASE1_EPOCHS} epochs, {PHASE1_FREEZE} backbone layers frozen — "
              f"early stopping active with patience=10)")
        run_phase(model, "P1", PHASE1_EPOCHS, freeze=PHASE1_FREEZE)
    else:
        print(f"[YOLO] Phase 1 best weights found at {p1_best}. Skipping Phase 1.")

    # Phase 2: full fine-tuning
    print(f"\nPhase 2 - Full fine-tuning  ({PHASE2_EPOCHS} epochs max, all layers trainable — "
          f"early stopping will typically trigger around epoch 60-80)")
    
    # Reload model from Phase 1 best weights to avoid Ultralytics KeyError:'model' on multiple train() calls
    p1_best = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")
    if os.path.exists(p1_best):
        model = YOLO(p1_best)
    
    run_phase(model, "P2", PHASE2_EPOCHS, freeze=0)

    # Post-training validation
    print("\n[YOLO] Running final validation...")
    val_results = model.val(data=YAML_PATH, verbose=False)
    try:
        final_map50 = val_results.box.map50
        final_map   = val_results.box.map
        print(f"[YOLO] mAP@0.5       : {final_map50:.4f}  (target >= {TARGET_MAP})")
        print(f"[YOLO] mAP@0.5:0.95  : {final_map:.4f}")
        status = "Target achieved." if final_map50 >= TARGET_MAP else \
                 "Below target. Consider more data or longer Phase 2."
        print(f"[YOLO] {status}")
    except Exception as e:
        print(f"[YOLO] Could not parse validation metrics: {e}")

    # Save best weights
    best_weights = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")
    if os.path.exists(best_weights):
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        target_path = os.path.join(WEIGHTS_DIR, FINAL_WEIGHT_NAME)
        if os.path.exists(target_path):
            ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = os.path.join(WEIGHTS_DIR, f"detector_yolo_{ts}.pt")
            shutil.copy2(target_path, backup)
            print(f"[YOLO] Previous weights backed up -> {backup}")
        shutil.copy2(best_weights, target_path)
        print(f"[YOLO] Best weights saved -> {target_path}")
    else:
        print(f"[YOLO] WARNING: best.pt not found at {best_weights}.")

    print("\n[YOLO] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO11m detector")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    train(resume=args.resume)
