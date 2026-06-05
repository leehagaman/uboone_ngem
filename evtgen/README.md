# EvtGen π⁰ Dalitz decay simulation

Standalone EvtGen simulation of π⁰ → e⁺e⁻γ Dalitz decays **at rest**, for
comparison against the Geant4 truth Dalitz sample from the BNB overlay files
(suspected of using a less accurate 3-body matrix element).

## Files

- `pi0_dalitz.dec` — user decay file forcing `pi0 -> e+ e- gamma PI0_DALITZ;`
  (EvtGen's Kroll–Wada QED model), overriding the default branching fractions so
  every generated π⁰ is a Dalitz decay.
- `pi0_dalitz_gen.cpp` — driver: creates a π⁰ at rest, decays it N times, and
  writes the γ / e⁺ / e⁻ four-momenta (MeV; `[E,px,py,pz]`) to a CSV. Because the
  π⁰ is at rest, the output is already in the π⁰ rest frame.
- `build_and_run.sh [n_events] [out_csv]` — compiles and runs against the conda
  EvtGen env.

## EvtGen install

EvtGen 2.2.3 from conda-forge, installed into a standalone env (base conda is
read-only):

```bash
conda create -y -p /nevis/riverside/data/leehagaman/ngem/evtgen_env -c conda-forge evtgen
```

## Run

```bash
./build_and_run.sh 1000000 /nevis/riverside/data/leehagaman/ngem/data_files/evtgen_pi0_dalitz.csv
```

## Compare with Geant4

- `../ipynb_notebooks/pi0_dalitz_decays.ipynb` — extracts the Geant4 truth Dalitz
  sample → `pi0_dalitz_momenta.npz`.
- `../ipynb_notebooks/pi0_dalitz_geant4_vs_evtgen.ipynb` — overlays the two
  (rest-frame energies, opening angle, energy asymmetry, dilepton angle, `m_ee`).


## Rough behavior should match what's described here: https://doi.org/10.1103/PhysRevC.95.025202
