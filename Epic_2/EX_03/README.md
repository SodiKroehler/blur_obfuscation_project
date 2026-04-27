# EX_03 — Two-Digit Obfuscation at 128×128

**Epic:** 2 | **Date:** 2026-04-26 | **Owner:** tanmay

## Hypothesis

## Results

## Notes
- Fixes the broken oracle from Epic 2 EX_01: classifier is now evaluated on tracked 28×28 crops, not on a downsampled full image
- Two non-overlapping MNIST digits per 128×128 composite
- Δ-VIS is averaged across both digits per image
