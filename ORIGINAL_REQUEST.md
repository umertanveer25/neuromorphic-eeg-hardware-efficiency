# Original User Request

## 2026-09-02T06:11:17+05:00

Write a complete, publication-ready Springer Nature journal manuscript (`manuscript.tex`) for the **Neuromorphic EEG Hardware Efficiency** project using the official Springer Nature LaTeX template (`sn-jnl.cls`, `sn-mathphys-num.bst`).

Working directory: `C:\Users\umert\.gemini\antigravity\scratch\Neuromorphic-EEG-Hardware-Efficiency`
Integrity mode: development

## Requirements

### R1. Springer Nature Manuscript LaTeX Structure (`manuscript.tex`)
- Title: *"Sub-Microjoule Event-Driven Spiking Neural Networks for Responsive Neurostimulation: Hardware-Software Co-Design for Patient-Independent EEG Seizure Detection"*
- Sole Author: **Umer Tanveer** (Department of Electrical and Computer Engineering / Biomedical Engineering).
- Include all standard Springer Nature sections: Abstract, Introduction, System Architecture & Neuromorphic Co-Design, Results & 8-Model Hardware Benchmarks, Bio-Thermal Safety & Longevity Analysis, Discussion, Methods, and Data Availability.
- Include all 5 high-resolution Nature-style figures (`Figure_1.png` to `Figure_5.png`) properly referenced with captions.
- Include all 5 LaTeX tables (Dataset Specs, 37-D Feature Breakdown, Full 8-Model Hardware Benchmark Table, SNN Microarchitecture Tape-Out Table, and 2023–2026 SOTA Literature Comparison Table).

### R2. BibTeX Bibliography Integration (`references.bib`)
- Fully integrate the 45 verified BibTeX references in `references.bib` using `\cite{}` commands throughout the manuscript text.
- Ensure citations include Jebaraj & Elango (2026), ISO 14708-3:2017, Pennes (1948), Davies et al. (2018 Loihi), and all 2023–2026 CHB-MIT competitors.

### R3. Overleaf-Ready Zip Package (`neuromorphic_eeg_manuscript_package.zip`)
- Bundle `manuscript.tex`, `sn-jnl.cls`, `references.bib`, and all 5 PNG figures into a clean, standalone Overleaf zip archive.

## Acceptance Criteria

### Compilation & Completeness
- [ ] `manuscript.tex` compiles cleanly with zero LaTeX syntax or undefined reference errors.
- [ ] All 5 figures and 5 tables are embedded and referenced sequentially (`Fig. 1` to `Fig. 5`, `Table 1` to `Table 5`).
- [ ] Word count exceeds 3,500 words with thorough mathematical, biological, and hardware explanations.
- [ ] Sole author is **Umer Tanveer**.
- [ ] Overleaf zip archive `neuromorphic_eeg_manuscript_package.zip` is created and verified.
