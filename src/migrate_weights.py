import os
import shutil

def migrate_weights():
    """
    Automated Migration Script: Neurologix Pro V2 -> V3.
    Copies required weights and label configurations.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, "weights")
    os.makedirs(target_dir, exist_ok=True)

    v2_base = os.path.join(os.path.dirname(base_dir), "mpv2")
    
    mapping = {
        # Source (relative to mpv2) -> Destination (relative to weights/)
        "models/tumor_classifier.pth": "tumor_classifier.pth",
        "models/unet_segmentor.pth": "unet_segmentor.pth",
        "models/label_map.json": "label_map.json",
        "runs/detect/neurologix_yolo_bt_cpu4/weights/best.pt": "detector_yolo.pt"
    }

    print("--- Starting Weights Migration ---")
    for src_rel, dst_name in mapping.items():
        src_path = os.path.join(v2_base, src_rel)
        dst_path = os.path.join(target_dir, dst_name)
        
        if os.path.exists(src_path):
            try:
                shutil.copy2(src_path, dst_path)
                print(f"[SUCCESS] Migrated: {src_rel} -> {dst_name}")
            except Exception as e:
                print(f"[ERROR] Failed to migrate {src_rel}: {e}")
        else:
            print(f"[WARNING] Skipping missing file: {src_path}")
            
    # Check for Golden Reference (if it was somehow in V2)
    golden_v2 = os.path.join(v2_base, "assets", "golden_reference.jpg")
    if os.path.exists(golden_v2):
        assets_v3 = os.path.join(base_dir, "assets")
        os.makedirs(assets_v3, exist_ok=True)
        shutil.copy2(golden_v2, os.path.join(assets_v3, "golden_reference.jpg"))
        print("[SUCCESS] Migrated Golden Reference.")

    print("\nMigration Complete. System ready for V3 Inference.")

if __name__ == "__main__":
    migrate_weights()
