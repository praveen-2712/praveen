# Neurologix Pro V3 — Clinical Model Card

## Intended Use
Decision support tool for brain tumor MRI classification.  
**NOT a standalone diagnostic device. Requires radiologist oversight.**  
Classes: Glioma, Meningioma, No Tumor, Pituitary Adenoma

## Performance Summary

| Class        | Sensitivity | Target | Met |
|---|---|---|---|
| Glioma       | 91.0% | 90% | PASS |
| Meningioma   | 99.2% | 85% | PASS |
| Notumor      | 100.0% | 90% | PASS |
| Pituitary    | 98.7% | 85% | PASS |

**Overall Accuracy**: 97.2%  
**Macro F1**: 0.9725

## Clinical Decision Thresholds
- **Glioma detection threshold** : P(glioma) > 2.41802e-05
- **Indeterminate threshold**    : max(P) < 0.55 → radiologist referral
- **Calibration temperature**    : T = 11.7224

## Safety Features
- MC-Dropout (15 passes): epistemic uncertainty quantification
- TTA (2 passes: original + H-flip): aleatoric uncertainty reduction
- Indeterminate output: required by IEC 62304 Class C
- Radiologist referral flag: compliant with FDA AI guidance
- GliomaMarginLoss: prevents Glioma↔Meningioma confusion at training time
- 6× Glioma class weight: clinical asymmetric penalty for missed diagnoses

## Known Limitations
- Trained on T1-weighted MRI only. T2/FLAIR not supported.
- Single 2D slice input. Volumetric analysis not performed.
- Not validated on paediatric populations.
- Not validated on post-operative or treated tumor imaging.

## Regulatory Framework
- Standard: IEC 62304 (Software as a Medical Device)
- Guidance: FDA AI/ML-Based SaMD Action Plan
- Decision support only — final diagnosis by licensed radiologist
