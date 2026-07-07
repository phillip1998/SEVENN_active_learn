# Project Context

## Project Goal

This project is about building and iterating a DFT-labeled training set for a machine-learned interatomic potential (MLP/GNN potential) intended for morphology molecular dynamics of YCO-series OLED emitter molecules.

The practical target is not only a low force RMSE model, but a potential that can run stable condensed-phase morphology MD and reproduce morphology-relevant structural statistics.

## Initial Planning Context

The initial discussion centered on how to build a DFT trainset for OLED molecular structures when training an MLP for morphology simulation.

Key points from that discussion:

- Single-molecule normal mode sampling or isolated conformer sampling is not sufficient for morphology MD.
- The trainset should cover structures that appear in condensed phase, especially dimer and cluster configurations.
- Important configurations include pi-stacking, acceptor-acceptor contacts, side-chain interdigitation, high-torsion conformers, local density fluctuations, and packing defects.
- Dimer sampling should be intentional, not purely random. Biasing toward pi-stack and other physically relevant contacts is part of the trainset design.
- MACE and NequIP are the main candidate model families. The choice should be based on data efficiency, condensed-phase stability, long-range/nonbonded interaction handling, uncertainty estimation workflow, and active learning practicality.
- Active learning is expected to be a core part of the loop after the initial model is trained.

## Expected Data Loop

The intended workflow is:

```text
Initial DFT dataset
-> Train MACE/NequIP model
-> Run MLP-MD
-> Detect uncertain or unphysical structures
-> Select diverse representative structures
-> Run DFT single-point energy/force calculations
-> Add results to dataset
-> Retrain model
-> Repeat
```

The initial dataset may include:

- Monomer conformers
- Dimer contacts
- Trimer or small cluster structures
- Distorted or high-energy but relevant structures
- Condensed-phase snapshots from morphology trajectories, when available

Each DFT-labeled structure should include:

- Total energy
- Atomic forces
- Stress, if periodic cells and pressure-related training are relevant

## Active Learning Criteria

Candidate structures for additional DFT labeling should be selected using a combination of:

- Ensemble or committee disagreement
- Force uncertainty thresholds
- Detection of unphysical MD events
- Diversity filtering such as clustering or farthest point sampling

Examples of problematic events:

- Abnormal bond elongation
- Ring distortion
- Atomic overlap or unrealistic close contacts
- Temperature runaway
- Energy drift
- Abnormal density change
- Collapse of pi-stack distance

## Morphology Validation Targets

The model should eventually be judged by morphology observables, not only energy/force validation errors.

Important observables include:

- Density
- Radial distribution functions
- Pi-pi stacking distance distribution
- Torsion angle distribution
- Nearest-neighbor orientation distribution
- Cluster motif distribution

Active learning can be considered mature when:

- MLP-MD runs stably at target temperature and density.
- High-uncertainty frames become rare.
- Validation force error is acceptable.
- Morphology observables no longer change meaningfully with additional active learning rounds.
- Newly added DFT structures provide little further improvement.

## User Request Context

The user asked to preserve the above planning context so future work in this workspace can refer back to it.

When interpreting future requests, assume the user is working toward an MLP-MD loop for OLED morphology, likely using existing or future trajectory files such as `YCOL160.xtc`, and wants practical project guidance rather than generic molecular simulation theory.

