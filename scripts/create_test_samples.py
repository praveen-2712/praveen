"""
create_test_samples.py — Neurologix
=====================================
Creates three curated test tiers from the Kaggle Testing set:

  1. high_confidence/  — ≥95% confidence, correctly classified (showstopper demos)
  2. moderate_confidence/ — 80-90% confidence, correctly classified (realistic performance)
  3. low_confidence/   — <65% confidence on ANY class (model uncertain, flags for review)

Usage:
    python scripts/create_test_samples.py
"""

import sys, os, cv2, numpy as np, torch, shutil
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import timm

# ── Config ──────────────────────────────────────────────────────────────────
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
SAMPLES_PER   = 25          # images per class per tier
MC_PASSES     = 10
IMG_SIZE      = 380
TEST_DIR      = os.path.join(os.path.dirname(__file__), '..', 'data', 'classification', 'Testing')
OUT_DIR       = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_samples')
WEIGHTS_PATH  = os.path.join(os.path.dirname(__file__), '..', 'weights', 'tumor_classifier.pth')
LABEL_MAP     = {0: 'glioma', 1: 'meningioma', 2: 'no_tumor', 3: 'pituitary'}
CLASS_FOLDERS = {
    'glioma':      ['glioma'],
    'meningioma':  ['meningioma'],
    'no_tumor':    ['notumor', 'no_tumor'],
    'pituitary':   ['pituitary'],
}

# Confidence tiers
TIERS = {
    'high_confidence':     {'min': 95.0, 'max': 100.0, 'correct_only': True},
    'moderate_confidence': {'min': 75.0, 'max': 92.0,  'correct_only': True},
    'low_confidence':      {'min': 0.0,  'max': 65.0,  'correct_only': False},  # all classes mixed
}

# ── Load model ──────────────────────────────────────────────────────────────
print(f"[Setup] Loading EfficientNet-B4 on {DEVICE}...")
model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=4)
sd = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=False)
sd = {k.replace('backbone.', ''): v for k, v in sd.items()}
model.load_state_dict(sd, strict=False)
model.to(DEVICE).eval()

norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(DEVICE)
norm_std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(DEVICE)


def classify(path):
    """Returns (predicted_label, confidence_pct, all_probs_array)"""
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return None, 0.0, None
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_r   = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    channels = []
    for clip in [1.5, 2.0, 2.5]:
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
        channels.append(clahe.apply(gray_r))
    stack = np.stack(channels, axis=-1)
    t = torch.from_numpy(stack).permute(2,0,1).float().unsqueeze(0).to(DEVICE) / 255.0
    t = (t - norm_mean) / norm_std
    probs_list = []
    with torch.no_grad():
        for _ in range(MC_PASSES):
            out = model(t)
            probs_list.append(torch.softmax(out, dim=1).cpu().numpy()[0])
    mean_p   = np.mean(probs_list, axis=0)
    pred_idx = int(np.argmax(mean_p))
    return LABEL_MAP[pred_idx], round(float(np.max(mean_p)) * 100, 2), mean_p


# ── Scan all test images ─────────────────────────────────────────────────────
# Collect: { 'class': [(conf, path, pred_label), ...] }
print("\n[Scan] Running inference on all test images...")
all_results = {cls: [] for cls in CLASS_FOLDERS}

for canonical_cls, aliases in CLASS_FOLDERS.items():
    src_dir = None
    for alias in aliases:
        candidate = os.path.join(TEST_DIR, alias)
        if os.path.isdir(candidate):
            src_dir = candidate
            break
    if src_dir is None:
        print(f"  [SKIP] {canonical_cls} — folder not found")
        continue

    files = sorted([
        f for f in os.listdir(src_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    print(f"  [{canonical_cls.upper()}] Scanning {len(files)} images...", end='', flush=True)

    for f in files:
        path = os.path.join(src_dir, f)
        pred, conf, probs = classify(path)
        if pred is None:
            continue
        is_correct = pred.replace('_', '') == canonical_cls.replace('_', '')
        all_results[canonical_cls].append((conf, path, pred, is_correct))

    print(f" done.")

print("\n[Build] Selecting samples for each tier...\n")

# ── Build each tier ──────────────────────────────────────────────────────────
grand_counts = {}

for tier_name, tier_cfg in TIERS.items():
    tier_dir = os.path.join(OUT_DIR, tier_name)

    if tier_name == 'low_confidence':
        # Low confidence: mixed classes, the model is uncertain.
        # Pull from all classes, pick correctly-predicted ones with low max-prob.
        # These will trigger the "Low model confidence" warning in the UI.
        tier_cls_dir = os.path.join(tier_dir, 'uncertain_mixed')
        os.makedirs(tier_cls_dir, exist_ok=True)
        for f in os.listdir(tier_cls_dir):
            os.remove(os.path.join(tier_cls_dir, f))

        bucket = []
        for canonical_cls, samples in all_results.items():
            for conf, path, pred, is_correct in samples:
                if tier_cfg['min'] <= conf < tier_cfg['max']:
                    bucket.append((conf, path, canonical_cls, pred))

        # Sort by confidence ascending (most uncertain first)
        bucket.sort()
        selected = bucket[:SAMPLES_PER * 4]  # up to 100 total uncertain images

        for conf, path, true_cls, pred in selected:
            fname = f"{true_cls}__pred_{pred}__{int(conf)}pct__{os.path.basename(path)}"
            shutil.copy2(path, os.path.join(tier_cls_dir, fname))

        grand_counts[tier_name] = len(selected)
        print(f"  [{tier_name}] {len(selected)} images saved to uncertain_mixed/")
        print(f"    (confidence range: {tier_cfg['min']:.0f}%–{tier_cfg['max']:.0f}%)")

    else:
        # High / moderate tiers: per-class, correctly predicted
        os.makedirs(tier_dir, exist_ok=True)
        tier_total = 0

        for canonical_cls, samples in all_results.items():
            cls_dir = os.path.join(tier_dir, canonical_cls)
            os.makedirs(cls_dir, exist_ok=True)
            for f in os.listdir(cls_dir):
                os.remove(os.path.join(cls_dir, f))

            # Only correct predictions in the confidence band
            bucket = [
                (conf, path)
                for conf, path, pred, is_correct in samples
                if is_correct and tier_cfg['min'] <= conf <= tier_cfg['max']
            ]
            bucket.sort(reverse=True)
            selected = bucket[:SAMPLES_PER]

            for conf, path in selected:
                shutil.copy2(path, os.path.join(cls_dir, os.path.basename(path)))

            tier_total += len(selected)
            if selected:
                print(f"  [{tier_name}/{canonical_cls}] {len(selected)} images "
                      f"({selected[-1][0]:.1f}%–{selected[0][0]:.1f}% conf)")
            else:
                print(f"  [{tier_name}/{canonical_cls}] WARNING: 0 images found in "
                      f"{tier_cfg['min']:.0f}%–{tier_cfg['max']:.0f}% band")

        grand_counts[tier_name] = tier_total

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST SAMPLE GENERATION COMPLETE")
print("="*60)
print(f"  Output: {os.path.abspath(OUT_DIR)}")
print()
for tier_name, count in grand_counts.items():
    label = {
        'high_confidence':     'HIGH  (>=95%)  — Demo-ready, always correct',
        'moderate_confidence': 'MED   (75-92%) — Realistic clinical confidence',
        'low_confidence':      'LOW   (<65%)   — Uncertain, triggers review flag',
    }[tier_name]
    print(f"  {label}: {count} images")
print()
print("USAGE GUIDE:")
print("  high_confidence/     -> Upload for a clean, impressive demo")
print("  moderate_confidence/ -> Upload to show realistic clinical variability")
print("  low_confidence/      -> Upload to demonstrate the review/warning system")
