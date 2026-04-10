from utils.analyzer import generate_report

def test_report_refinement():
    # Mock data
    label = "glioma"
    confidence = 92.5
    regions = [
        {"id": 0, "centroid": (100, 100), "area": 800, "source": "yolo"}
    ]
    image_shape = (224, 224, 3)

    print("Generating report...")
    report = generate_report(label, confidence, regions, image_shape)

    print("\nReport Interpretation:", report["interpretation"])
    print("Report Notes:", report["notes"])

    # Check for medical advice keywords
    forbidden_keywords = ["surgical", "resection", "evaluation", "imaging", "testing", "intervention"]
    found_advice = [k for k in forbidden_keywords if k in report["notes"].lower()]
    
    if len(found_advice) == 0:
        print("\nSUCCESS: No medical advice found in notes.")
    else:
        print(f"\nFAILURE: Medical advice found: {found_advice}")

    # Check for morphology fields in regions
    if "regions" in report and len(report["regions"]) > 0:
        r = report["regions"][0]
        if "margin" in r or "intensity" in r:
            print(f"FAILURE: Morphology fields found in region: {list(r.keys())}")
        else:
            print("SUCCESS: No morphology fields found in regions.")
    
    print("\nRegion data:", report["regions"][0])

if __name__ == "__main__":
    test_report_refinement()
