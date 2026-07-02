"""
check_weights.py -- Verify tumor_classifier.pth matches current DualBranchClassifier.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn as nn
import timm

class DualBranchClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.branch_global = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=0, drop_rate=0.4, drop_path_rate=0.3)
        self.branch_local  = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=0, drop_rate=0.4, drop_path_rate=0.3)
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1280 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, img_full, img_crop):
        feat_global = self.branch_global(img_full)
        feat_local  = self.branch_local(img_crop)
        fused = torch.cat([feat_global, feat_local], dim=1)
        return self.head(fused)


WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "tumor_classifier.pth")

print("=" * 60)
print("Neurologix -- Classifier Weight Compatibility Check")
print("=" * 60)

# 1. Load checkpoint
print(f"\n[1] Loading: {WEIGHTS_PATH}")
sd = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
ckpt_keys = set(sd.keys())
print(f"    Checkpoint keys : {len(ckpt_keys)}")

# Show top-level key prefixes
prefixes = {}
for k in ckpt_keys:
    prefix = k.split('.')[0]
    prefixes[prefix] = prefixes.get(prefix, 0) + 1
print(f"    Key prefixes    : {dict(sorted(prefixes.items()))}")

# 2. Build model
print("\n[2] Building DualBranchClassifier(num_classes=4) ...")
model = DualBranchClassifier(num_classes=4)
model_keys = set(model.state_dict().keys())
print(f"    Model keys      : {len(model_keys)}")

model_prefixes = {}
for k in model_keys:
    prefix = k.split('.')[0]
    model_prefixes[prefix] = model_prefixes.get(prefix, 0) + 1
print(f"    Model prefixes  : {dict(sorted(model_prefixes.items()))}")

# 3. Compare
missing    = model_keys - ckpt_keys
unexpected = ckpt_keys - model_keys

print("\n[3] Key comparison:")
if not missing and not unexpected:
    print("    PERFECT MATCH -- all keys align.")
else:
    print(f"    MISMATCH DETECTED!")
    print(f"    Keys in model but missing from checkpoint : {len(missing)}")
    print(f"    Keys in checkpoint but not in model       : {len(unexpected)}")

    print("\n    --- First 15 MISSING keys (model needs, checkpoint lacks) ---")
    for k in sorted(missing)[:15]:
        print(f"        {k}")

    print("\n    --- First 15 UNEXPECTED keys (checkpoint has, model doesn't) ---")
    for k in sorted(unexpected)[:15]:
        print(f"        {k}")

# 4. Strict load
print("\n[4] Attempting strict load ...")
try:
    result = model.load_state_dict(sd, strict=True)
    print("    STRICT LOAD OK")
    print(f"    Missing keys    : {result.missing_keys[:5]}")
    print(f"    Unexpected keys : {result.unexpected_keys[:5]}")
except RuntimeError as e:
    lines = str(e).split('\n')
    print(f"    STRICT LOAD FAILED:")
    for line in lines[:10]:
        print(f"    {line}")

# 5. Shape mismatches on overlapping keys
print("\n[5] Shape audit on overlapping keys ...")
model_sd = model.state_dict()
shape_mismatches = []
for k in ckpt_keys & model_keys:
    if sd[k].shape != model_sd[k].shape:
        shape_mismatches.append((k, sd[k].shape, model_sd[k].shape))

if shape_mismatches:
    print(f"    Shape mismatches ({len(shape_mismatches)}):")
    for k, cs, ms in shape_mismatches[:10]:
        print(f"        {k}: checkpoint={cs}  model={ms}")
else:
    print(f"    All {len(ckpt_keys & model_keys)} overlapping tensors have matching shapes.")

# 6. Check if checkpoint is single-branch EfficientNet
print("\n[6] Diagnosing checkpoint architecture ...")
has_branch_global = any(k.startswith('branch_global') for k in ckpt_keys)
has_branch_local  = any(k.startswith('branch_local')  for k in ckpt_keys)
has_head          = any(k.startswith('head')           for k in ckpt_keys)
has_model         = any(k.startswith('model')          for k in ckpt_keys)

print(f"    Has 'branch_global' keys : {has_branch_global}")
print(f"    Has 'branch_local'  keys : {has_branch_local}")
print(f"    Has 'head'          keys : {has_head}")
print(f"    Has 'model'         keys : {has_model}")

if has_model and not has_branch_global:
    print("\n    DIAGNOSIS: Checkpoint is from a SINGLE-BRANCH model (old architecture).")
    print("    It does NOT match the current DualBranchClassifier.")
    print("    The classifier needs to be RETRAINED with the current architecture.")
elif has_branch_global and has_branch_local:
    print("\n    DIAGNOSIS: Checkpoint has both branches -- architecture matches.")
elif has_branch_global and not has_branch_local:
    print("\n    DIAGNOSIS: Only branch_global present -- partial mismatch.")

print("\n" + "=" * 60)
print("Check complete.")
print("=" * 60)
