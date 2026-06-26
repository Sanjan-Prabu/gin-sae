# SAE on a Pretrained Molecular GNN (GIN) — Validation Experiment

A first step toward interpreting graph-neural-network drug-repurposing models with
**sparse autoencoders (SAEs)**: we validate the SAE approach on a *pretrained, frozen*
molecular GNN before moving to graph models of interest.

## What this is

- **Model:** a pretrained **GIN (Graph Isomorphism Network)** — `gin_supervised_contextpred`
  (Hu et al. 2020, pretrained on ~2M molecules). It takes a small molecule, analyzes its
  atoms, and produces a per-atom embedding from the molecule's structure. We picked it
  because it is in the graph-neural-network family and uses the **same message-passing
  mechanism** as the graph models we ultimately want to interpret. It stays **frozen** —
  we only train the SAE.
- **Method (InterPLM recipe, molecular edition):**
  1. Pass molecules through the frozen GIN → harvest a 300-dim embedding per atom.
  2. Train a **TopK** sparse autoencoder on the embeddings (TopK enforces the sparsity
     that separates out interpretable features).
  3. Validate features against chemistry: using **RDKit** as an independent answer key,
     check whether each feature's top-activating atoms share a functional group, far above
     its background rate (purity + enrichment).
- **Cross-validation:** the SAE is trained on atoms from one set of molecules and the
  features are validated on a **separate held-out set of compounds** the SAE never saw.

## Results (held-out compounds)

Trained on **6,000** molecules (111,747 atoms); validated on **1,831 separate held-out**
molecules (33,509 atoms).

- **Reconstruction:** held-out **FVU 0.11** (~89% of embedding variance explained).
- **Interpretability (held-out):** **500** features are >=0.70 pure for a chemical concept,
  **158** are >=0.80, **20** are >=0.90.
- **Clean functional-group detectors** (purity | enrichment over background):
  ether 0.90 | 9x, halogen 0.72 | 22x, hydroxyl 0.72 | 11x, carboxylic acid 0.50 | 25x,
  ketone 0.50 | 20x; rarer groups show very high enrichment (nitrile 120x, sulfone 134x).

**Takeaway:** an SAE recovers real, chemically meaningful features from a message-passing
GNN — on compounds it never trained on. This validates the technique before applying it to
graph-based drug-repurposing models.

## Figures (`figures/`)

- `fig1_top_molecules.png` — top molecules for representative features, with the atoms that
  activate each feature highlighted (e.g., the ether feature lights up on ether oxygens).
- `fig2_feature_concept_heatmap.png` — feature x functional-group purity. The highest-purity
  features are common structural concepts (ring / aromatic); specific functional-group
  detectors (e.g. the ether feature) appear as distinct bright cells.
- `fig3_interpretability_summary.png` — how many features are interpretable (purity tiers),
  and the best detector per functional group by enrichment.

## Run it

Open `gin_sae_experiment.ipynb` and run top to bottom. Requirements:
`torch`, `dgl`, `dgllife`, `rdkit`, `pandas`, `numpy`, `matplotlib`. A GPU helps but isn't
required (the SAE is small). The notebook downloads the pretrained GIN and the Tox21
molecule set automatically, and regenerates the figures into `figs/`.

## Notes

- Harvested embeddings (~100 MB) are **not committed** — the notebook regenerates them.
- `results/gin_sae_cv_results.json` holds the held-out metrics and per-concept detectors.
