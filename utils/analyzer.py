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
        "interpretation": "",
        "notes": ""
    }

    if label == "no_tumor":
        report["interpretation"] = "No evidence of intracranial mass lesion detected."
        report["notes"] = "Clinical correlation remains recommended if symptoms persist."
        return report

    # Characteristics based on tumor type
    if label == "glioma":
        char_type = "Intra-axial mass with infiltrative margins. Often involves deep white matter."
        risk = "Potential neurosurgical intervention required for histopathological confirmation."
    elif label == "meningioma":
        char_type = "Extra-axial, well-circumscribed lesion. Likely originating from the dura mater."
        risk = "Consider clinical follow-up or surgical resection depending on size and symptoms."
    elif label == "pituitary":
        char_type = "Sellar or suprasellar mass suggesting pituitary origin."
        risk = "Endocrine evaluation and visual field testing recommended."
    else:
        char_type = "Lesion detected with specific characteristics described below."
        risk = "Further diagnostic imaging may be warranted."

    report["interpretation"] = f"Findings are suspicious for {label} lesion ({confidence}% confidence)."
    report["notes"] = f"{char_type} {risk}"

    # Analyze each region
    for r in regions:
        # Distance from image center (centroid vs 0.5, 0.5)
        cy, cx = r["centroid"]
        dist_from_center = np.sqrt(((cx / image_shape[1]) - 0.5)**2 + ((cy / image_shape[0]) - 0.5)**2)
        
        location = "Central" if dist_from_center < 0.25 else "Peripheral"
        margin = "Regular" if r["area"] > 500 else "Small/Diffuse" # Simplified heuristic

        report["regions"].append({
            "id": r["id"],
            "location": location,
            "margin": margin,
            "area_px": int(r["area"]),
            "intensity": "Regular",
            "source": r.get("source", "Standard"),
            "sources": r.get("sources", [r.get("source", "Standard")]),
            "crop_b64": r.get("crop_b64", "")
        })

    return report
