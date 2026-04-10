import numpy as np

def generate_report(label, confidence, regions, image_shape):
    """
    Generate a doctor-style radiological report based on the findings.
    """
    report = {
        "finding": label,
        "confidence": confidence,
        "n_lesions": len(regions),
        "regions": [],
        "interpretation": ""
    }

    if label == "no_tumor":
        report["interpretation"] = "No evidence of intracranial mass lesion detected."
        return report

    # Characteristics based on tumor type (Descriptive only)
    if label == "glioma":
        char_type = "Radiological patterns consistent with an intra-axial mass and infiltrative margins."
    elif label == "meningioma":
        char_type = "Radiological patterns consistent with an extra-axial, well-circumscribed lesion."
    elif label == "pituitary":
        char_type = "Radiological patterns consistent with a sellar or suprasellar mass."
    else:
        char_type = "Non-specific radiological patterns identified in the automated scan."

    report["interpretation"] = f"Automated analysis identified patterns suspicious for {label} ({confidence}% confidence)."

    # Analyze each region
    for r in regions:
        # Distance from image center (centroid vs 0.5, 0.5)
        cy, cx = r["centroid"]
        dist_from_center = np.sqrt(((cx / image_shape[1]) - 0.5)**2 + ((cy / image_shape[0]) - 0.5)**2)
        
        location = "Central" if dist_from_center < 0.25 else "Peripheral"

        report["regions"].append({
            "id": r["id"],
            "location": location,
            "area_px": int(r["area"]),
            "source": r.get("source", "Standard"),
            "sources": r.get("sources", [r.get("source", "Standard")]),
            "crop_b64": r.get("crop_b64", "")
        })

    return report
