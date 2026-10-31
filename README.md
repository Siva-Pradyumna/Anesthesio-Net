# AnesthesioNet-AI

**A Predictive, Latency-Resilient, Full-Lifecycle Multi-Drug Closed-Loop AI Anesthesia Co-Pilot with Dual-Engine (Digital Twin + Real-World Dataset) Execution, Conformal Safety Guardrails, High-Order Control Barrier Functions, and Real-Time Explainability**

---

## Overview
**AnesthesioNet-AI** is an autonomous closed-loop medical automation platform engineered to serve as an intelligent co-pilot for anesthesiologists, enabling a **"1-to-Many" Supervision Model** across multiple operating suites.

### Core Modules
* **Mode A (Digital Twin Simulator):** 3-compartment coupled PK/PD simulator (Eleveld & Schnider models) with Minto/Bouillon pharmacodynamic synergy response surfaces and coupled hemodynamics.
* **Mode B (Historical Replay Engine):** High-throughput data streaming from **VitalDB** (6,388 cases) with Doubly Robust Off-Policy counterfactual evaluation.
* **Signal Integrity Layer:** Multi-rate telemetry processing with dynamic velocity gates and spectral SQI/EMG electrocautery noise rejection.
* **Network Latency Estimator:** Delay-compensated Extended Kalman Filter (EKF) with action-delay rollback.
* **Predictive Forecasting:** Multi-horizon Temporal Fusion Transformer (TFT) with Split Conformal Prediction (95% coverage risk bounds).
* **Safety & Control Layer:** Multi-Input Multi-Output Soft Actor-Critic (MIMO-SAC) RL filtered through High-Order Control Barrier Functions (HOCBF-QP) complying with **IEC 60601-1-10 (PCLC)**.
* **Hardware Fail-Safe:** Local edge watchdog daemon triggering patient-specific basal TCI hold upon network heartbeat disruption.

---

## Quickstart & Verification

### 1. Prerequisites
* Python 3.11+
* `uv` or standard virtual environment

### 2. Setup
```bash
# Create and activate virtual environment
uv venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
uv pip install numpy scipy
```

### 3. Run the Digital Twin Physiology Simulation
```bash
python demo_simulation.py
```

### 4. Run Verification Tests
```bash
python tests/test_physiology.py
```
