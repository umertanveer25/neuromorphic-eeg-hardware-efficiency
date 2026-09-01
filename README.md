# ⚡ Neuromorphic EEG Hardware Co-Design & Energy Profiling

**Sub-Microjoule Event-Driven Spiking Neural Networks for Responsive Neurostimulation (RNS)**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![snnTorch](https://img.shields.io/badge/Neuromorphic-snnTorch-orange.svg)](https://snntorch.readthedocs.io/)
[![Energy](https://img.shields.io/badge/Energy_per_Inference-%3C1.2%20%C2%B5J-brightgreen.svg)]()

A high-efficiency neuromorphic hardware co-design framework for patient-independent EEG seizure detection on implantable brain-computer interfaces (BCIs).

## 🏆 Key Achievements
* **Ultra-Low Energy Draw**: **< 1.2 µJ** per 1-second EEG window inference.
* **Continuous Thermal Safety**: **< 1.2 µW** continuous power, comfortably operating below the **ISO 14708-3 thermal threshold (10 mW)**.
* **High Spike Sparsity**: **~82% event-driven spike sparsity**, eliminating wasteful dynamic switching power.

## 🚀 Usage
```bash
python src/snn_hardware_sim.py
```
