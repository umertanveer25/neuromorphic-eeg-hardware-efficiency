# ⚡ Sub-Microjoule Neuromorphic EEG Hardware Co-Design

[![GitHub Stars](https://img.shields.io/github/stars/umertanveer25/neuromorphic-eeg-hardware-efficiency?style=social)](https://github.com/umertanveer25/neuromorphic-eeg-hardware-efficiency)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![snnTorch](https://img.shields.io/badge/Neuromorphic-snnTorch-orange.svg)](https://snntorch.readthedocs.io/)
[![ISO 14708-3 Safety](https://img.shields.io/badge/ISO_14708--3-Compliant_(%3C10_mW)-brightgreen.svg)]()
[![Continuous Power Draw](https://img.shields.io/badge/Power_Draw-0.8514_%C2%B5W-purple.svg)]()

> Official Repository for **"Sub-Microjoule Event-Driven Spiking Neural Networks for Responsive Neurostimulation: Hardware-Software Co-Design for Patient-Independent EEG Seizure Detection"** by **Umer Tanveer** (2026).

---

## 📌 Abstract & Key Contributions

Patient-independent automated seizure detection on implantable Responsive Neurostimulation (RNS) devices faces a dual constraint: achieving high cross-subject generalization while operating strictly under **ISO 14708-3 medical thermal tissue safety caps (< 10 mW)**.

In this work, we present a high-efficiency **hardware-software co-design** framework combining a **37-Dimensional Multi-Domain Feature Extraction Engine** (DWT + Hjorth Parameters + Spectral Entropy) with an **event-driven Leaky Integrate-and-Fire (LIF) Spiking Neural Network** targeting 28nm CMOS / Intel Loihi 2 neuromorphic silicon topology.

### 🏆 Key Benchmarks:
* 🎯 **State-of-the-Art LOPO Accuracy:** Achieves **78.71% LOPO accuracy** with LightGBM and **75.19% LOPO accuracy** with SNN, crushing the base paper SNN benchmark (**64.34%**, **+10.85% jump**).
* ⚡ **Sub-Microjoule Energy Draw:** Requires only **0.8514 µJ** per 1-second EEG window inference.
* 🔋 **Extended Implant Battery Longevity:** Operates at **0.8514 µW continuous power draw**, comfortably extending surgical battery replacement cycles to **15.4 years** (vs. 1.2 years for conventional deep neural networks).
* 🌡️ **Zero Cortical Tissue Heating:** Operates over **11,000x below the ISO 14708-3 thermal necrosis limit (10 mW)** ($\Delta T < 0.001^\circ	ext{C}$).

---

## 🖼️ System Architecture & Visual Results

### Figure 1: Closed-Loop Neuromorphic RNS Interface
![System Architecture](figures/Figure_1.png)

### Figure 2: LIF Spiking Dynamics & 81.96% Event Spike Sparsity
![Spiking Dynamics](figures/Figure_2.png)

### Figure 3: Multi-Model Pareto Frontier & ISO 14708-3 Safety Cap
![Pareto Frontier](figures/Figure_3.png)

### Figure 4: Full 8-Model Cross-Patient LOPO Benchmarks & Transfer Heatmap
![LOPO Benchmarks](figures/Figure_4.png)

### Figure 5: Cortical Bio-Thermal Dissipation & Battery Lifespan Projections
![Bio-Thermal Safety](figures/Figure_5.png)

---

## 📊 Full 8-Model Hardware Benchmark Table

| Model Group | Model Architecture | Input Features | LOPO Acc (%) | Params | Ops / Sample | Energy Draw ($\mu	ext{J}$) | Continuous Power ($\mu	ext{W}$) | ISO 14708-3 Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base Features** | **ANN Baseline** | 3-D (SII) | 56.58% | 1,538 | 1,280 MACs | 0.0059 $\mu	ext{J}$ | 0.0059 $\mu	ext{W}$ | ✅ PASSED |
| **Base Features** | **CNN Baseline** | 3-D (SII) | 54.12% | 98 | 176 MACs | 0.0008 $\mu	ext{J}$ | 0.0008 $\mu	ext{W}$ | ✅ PASSED |
| **Base Features** | **LightGBM Baseline** | 3-D (SII) | 60.25% | 6,300 | 600 Ops | 0.0001 $\mu	ext{J}$ | 0.0001 $\mu	ext{W}$ | ✅ PASSED |
| **Base Features** | **SNN Baseline** | 3-D (SII) | 64.34% | 1,538 | 1,094,885 SOPs | 0.9854 $\mu	ext{J}$ | 0.9854 $\mu	ext{W}$ | ✅ PASSED |
| | | | | | | | | |
| **OUR WORK** | **LightGBM (SOTA)** | **37-D Multi-Domain** | **78.71%** | 6,300 | 600 Ops | **0.0001 $\mu	ext{J}$** | **0.0001 $\mu	ext{W}$** | ✅ PASSED (< 10 mW) |
| **OUR WORK** | **Spiking SNN (LIF)** | **37-D Multi-Domain** | **75.19%** | 10,242 | 946,027 SOPs | **0.8514 $\mu	ext{J}$** | **0.8514 $\mu	ext{W}$** | ✅ PASSED (< 10 mW) |
| **OUR WORK** | **ANN (MLP)** | **37-D Multi-Domain** | **72.88%** | 10,242 | 9,984 MACs | **0.0459 $\mu	ext{J}$** | **0.0459 $\mu	ext{W}$** | ✅ PASSED (< 10 mW) |
| **OUR WORK** | **1D-CNN** | **37-D Multi-Domain** | **71.40%** | 642 | 2,352 MACs | **0.0108 $\mu	ext{J}$** | **0.0108 $\mu	ext{W}$** | ✅ PASSED (< 10 mW) |

---

## 🛠️ Installation & Usage

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/umertanveer25/neuromorphic-eeg-hardware-efficiency.git
cd neuromorphic-eeg-hardware-efficiency
pip install -r requirements.txt
```

### 2. Run Neuromorphic SNN Hardware Profiler
```bash
python src/snn_hardware_sim.py
```

### 3. Run Full Multi-Model Energy & Thermal Safety Benchmark
```bash
python src/multi_model_hardware_profiler.py
```

### 4. Run LOPO Cross-Validation Evaluation
```bash
python src/evaluate_hardware_lopo.py
```

---

## 📜 Citation

If you use this repository, codebase, or neuromorphic hardware profiler in your research, please cite:

```bibtex
@article{tanveer2026neuromorphic,
  author    = {Tanveer, Umer},
  title     = {Sub-Microjoule Event-Driven Spiking Neural Networks for Responsive Neurostimulation: Hardware-Software Co-Design for Patient-Independent {EEG} Seizure Detection},
  journal   = {IEEE Transactions on Biomedical Circuits and Systems (TBICAS)},
  year      = {2026},
  publisher = {IEEE}
}
```

---

## 📄 License
This project is released under the [MIT License](LICENSE).
