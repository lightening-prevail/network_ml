# Split Intelligence Under Policy-Shaped Loss

A research prototype of distributed split-DNN inference in which a
congestion-aware gateway deliberately drops **intermediate feature packets**
according to their importance scores, and the server classifies from the
incomplete feature tensor. The central question is whether *policy-aware
training* beats *random-dropout training* when loss is structured.

This is an experimental system, not an image-classification demo. A real CNN
produces the features, real packets are built and dropped, real reconstruction
and remaining-layer inference happen, and every number in the dashboard comes
from a stored experiment row.

---

## What is implemented

| Component | Status |
| --- | --- |
| MobileNetV2-CIFAR / ResNet-18-CIFAR with structural split points | working, tested |
| Channel-group and spatial-tile packetization, 60-byte binary header | working, tested |
| Activation-L2 importance, quantized to uint8 | working, tested |
| Gateway dropping: random, burst, importance (rank + threshold), hybrid | working, tested |
| Reconstruction: zero / per-channel mean / nearest neighbour | working, tested |
| Training strategies A (standard), B (random dropout), C (policy-aware ordered) | working, tested |
| Shared loss semantics between training masks and gateway | working, tested |
| Experiment store (SQLite), JSON configs, checkpoints, CSV/JSON export | working, tested |
| Full evaluation matrix (policy × loss × bandwidth, split depth, 1 vs 2 nodes) | working |
| FastAPI + WebSocket backend, live inference, live gateway emulation | working |
| React/TypeScript/Tailwind dashboard, 7 pages, Recharts + React Flow | working, builds |
| UDP edge node / gateway / inference server for physical deployment | working, loopback-tested |
| P4/BMv2 importance-threshold demo | source provided, **not hardware-validated** |
| Raspberry Pi camera capture | Picamera2 code path, **not run here (no Pi)** |

## What is *not* claimed

- Network latency is **modeled** from measured bytes; only compute latency is
  measured (CPU host, not a Pi).
- The bandwidth→loss coupling is a documented fluid-queue model, not a real
  saturated link.
- The P4 program matches a **fixed importance threshold**; it does not rank
  packets like the software gateway, and it has not run on hardware.
- Short training schedules produce absolute accuracies well below published
  CIFAR-10 results. Comparisons are paired, but they are prototype-scale.

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) and the Limitations section of
the dashboard's Research Results page for the complete list.

---

## Measured results in this workspace

These are actual stored results (seed-averaged over 2 seeds, 2000 held-out
CIFAR-10 images, top-1). Setup: MobileNetV2-CIFAR, split at `block3`
(24×16×16 features, 6 channel-group packets), 6000 training images, 6 epochs,
SGD + cosine, batch 128, CPU. Full rows and figures:
`experiments/results/report.json`, `experiments/figures/`, and the dashboard.

| training strategy | 0 % loss | 30 % loss | 50 % loss | drop at 50 % |
| --- | --- | --- | --- | --- |
| A standard | 48.5 % | 40.9 % | 32.9 % | −15.6 pp |
| B random dropout | 45.5 % | 44.9 % | 44.2 % | −1.2 pp |
| C policy-aware | 44.5 % | 44.6 % | 42.0 % | −2.4 pp |

Bandwidth sweep (fluid bandwidth→loss coupling at 10 inferences/s):

| link | effective loss | standard | random | policy |
| --- | --- | --- | --- | --- |
| 0.5 Mbps | 74.9 % | 24.4 % | **41.5 %** | 34.1 % |
| 1 Mbps | 49.9 % | 32.9 % | **44.2 %** | 42.0 % |
| ≥ 2 Mbps | 0 % | **48.5 %** | 45.5 % | 44.5 % |

Split depth with the same weights (0 % / 30 % loss): `stem` 46.1/37.6,
`block1` 46.1/40.8, `block3` 46.1/43.5, `block6` 46.1/44.6, `block13`
46.1/45.0 %. One vs two edge nodes at 2 Mbps, 30 % loss: identical accuracy
(0.4345) and 69.6 ms vs 136.5 ms end-to-end (measured compute + modeled shared
link).

**What this says — including the negative part.** Both learned-loss strategies
are dramatically more robust than standard training, and policy-aware training
almost eliminates clean-accuracy loss. However, **policy-aware training did not
outperform random-dropout training** under importance-aware dropping at these
budgets: it was 1–2 pp behind at 0 %, 30 % and 50 % loss, and further behind at
a 0.5 Mbps bottleneck. So this prototype does **not** provide evidence for the
strong hypothesis that policy-shaped training beats random dropout; it provides
evidence that *any* loss-aware training is essential, plus a clear clean-cost
ordering.

A likely confound is documented rather than hidden: rank-based importance
dropping removes the **lowest-norm** payloads and then zero-fills them, which is
a comparatively mild perturbation. At 30 % loss the *standard* model scores
40.9 % under importance dropping but 38.1 % under random dropping — equal packet
budgets, different information loss. Training strategy C mixes importance,
burst and hybrid masks, which may also dilute the importance-specific signal.
Longer schedules, an importance-only training mix, and a less degenerate
reconstruction are the obvious next experiments.

Gradient-importance ablation on the policy checkpoint (800 images, 30 % loss):
activation-L2 ranking 47.1 %, gradient-L2 ranking 45.1 %, no loss 46.8 %. The
edge cannot compute gradient scores alone, so this remains an offline ablation.

---

## Architecture

```
   Raspberry Pi 4 + Camera (or CIFAR-10 test frames)
   ┌──────────────────────────────────────┐
   │ Edge Node A                          │
   │ capture → preprocess → front layers  │
   │ → packetize → importance → UDP       │
   └───────────────┬──────────────────────┘
                   │  feature packets (60 B header + float32 payload)
   ┌───────────────▼──────────────────────┐        ┌──────────────────────┐
   │ Edge Node B (independent stream)     │───────►│  Network Gateway     │
   └──────────────────────────────────────┘        │  random / burst /    │
                                                   │  importance / hybrid │
                                                   │  no CNN code         │
                                                   └──────────┬───────────┘
                                                              │ survivors
                                                   ┌──────────▼───────────┐
                                                   │ Inference Server     │
                                                   │ reassemble, detect   │
                                                   │ missing, reconstruct │
                                                   │ → remaining layers   │
                                                   │ → prediction + times │
                                                   └──────────────────────┘

   FastAPI backend ⇄ SQLite metrics     WebSocket ⇄ React dashboard
```

Two execution modes share the same policy and model code:

1. **In-process emulation** (`splitintel/`): the dashboard and offline
   experiments run the edge front, gateway dropping and server back inside one
   Python process. Loss is simulated exactly at packet granularity, with the
   same mask functions as training.
2. **Real UDP path** (`edge/`): edge node, gateway and server are separate
   processes talking over UDP, for Raspberry Pi deployment or a loopback demo.

---

## Quickstart

Python 3.10–3.12 recommended (the project was built and tested on 3.11).

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

Train and evaluate the three strategies on CIFAR-10 (CPU is fine):

```bash
python scripts/run_experiments.py --config configs/base.json \
    --strategies standard random policy --seeds 0 --epochs 8
```

This writes checkpoints to `experiments/checkpoints/`, metrics to
`experiments/split_intelligence.db`, per-experiment JSON to
`experiments/results/`, and figures to `experiments/figures/`.

Run the dashboard:

```bash
.venv\Scripts\python -m splitintel.cli serve --port 8000
# open http://127.0.0.1:8000  (build the UI first if frontend/dist is absent)
```

Frontend development (hot reload, proxied to :8000):

```bash
cd frontend
npm install
npm run dev        # http://127.0.0.1:5173
npm run build      # production bundle served by FastAPI at /
```

Test the real UDP path over loopback (after training one checkpoint):

```bash
python scripts/check_udp_path.py --loss-rate 0.3
```

Headless report generation:

```bash
python -m splitintel.cli report --out experiments/results/report.json
```

---

## Repository layout

```
splitintel/            research core + dashboard backend
  models.py            MobileNetV2/ResNet18 CIFAR variants, split points
  packetize.py         packet layout, binary format, importance
  policies.py          dropping policies (shared by gateway and training)
  reconstruct.py       missing-feature reconstruction strategies
  train.py             strategies A/B/C, checkpointing, gradient ablation hook
  evaluate.py          loss sweeps, measured vs modeled latency
  runner.py            experiment orchestration (full evaluation matrix)
  store.py             SQLite store + CSV/JSON export
  inference_service.py live inference, model registry, traffic engine
  plots.py, report.py  figures and research report from stored rows
  api.py, cli.py       FastAPI app and command line
edge/                  physical deployment: edge_node, gateway, inference_server, camera
p4/                    BMv2 importance-threshold demo
frontend/              React + TypeScript + Tailwind dashboard (7 pages)
configs/               experiment configurations (JSON)
scripts/               reproducible drivers, UDP check, gradient-importance ablation
docs/                  methodology, packet format, Raspberry Pi deployment
tests/                 packet, model, policy, reconstruction, pipeline tests
```

---

## Experiment configuration

Every experiment records model architecture, dataset, split layer, training
strategy, training-time drop policy, testing-time drop policy, packetization,
loss rate, bandwidth, edge-node count, accuracy, latency and communication
overhead. `configs/base.json` is a complete example; YAML is accepted too.

Changing the split point does not require retraining: the state dict is
split-agnostic, so one checkpoint is evaluated at every split depth (recorded
in each metric row). Splits are structural cuts, not separately trained
edge/server pairs — a documented limitation.

## Reproducibility

- Seeds are set for Python, NumPy and Torch; masks use dedicated
  `numpy.random.Generator` instances derived from the experiment seed.
- The test split and its order are identical across strategies and policies.
- Every loss sweep includes a 0 % reference row, and the same policy
  implementation produces training masks and deployment drops.
- Checkpoints embed the full experiment configuration and training history;
  each metric row carries its own condition (policy, mode, rate, bandwidth,
  reconstruction, split, node count).

## License

Research prototype, provided as-is for academic experimentation.
