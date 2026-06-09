# Standalone Geant4 π⁰ Dalitz decay generator

High-statistics π⁰ → e⁺e⁻γ Dalitz decays from Geant4's own decay model
(`G4DalitzDecayChannel`) — the same model that decays GENIE π⁰s in the larg4
detector simulation. Used as the **pure-model source** for the Geant4→EvtGen
reweighting (replacing the ~5k selection-biased decays extracted from the overlay).

Decay-only: we call `G4DalitzDecayChannel::DecayIt()` directly with the π⁰ at rest,
so the products come out already in the π⁰ rest frame — no geometry, physics list,
or tracking, so millions of decays are cheap.

## Files

- `g4_dalitz_gen.cpp` — driver: `G4DalitzDecayChannel("pi0", 1.0, "e-", "e+")`,
  N decays at rest, dumps γ / e⁺ / e⁻ four-momenta (MeV) to CSV.
- `build_and_run.sh [n_events] [out_csv] [seed]` — compiles via `geant4-config`
  and runs (sources the conda env's `activate.d` scripts to set `G4ENSDFSTATEDATA`
  etc., required to initialize the particle table).

## Geant4 install

Geant4 11.3.2 from conda-forge, in a standalone env (base conda is read-only):

```bash
conda create -y -p /nevis/riverside/data/leehagaman/ngem/geant4_env -c conda-forge geant4
```

## Run

```bash
./build_and_run.sh 1000000 /nevis/riverside/data/leehagaman/ngem/data_files/geant4_pi0_dalitz.csv
```

## Validation

Standalone Geant4 reproduces the production decay model (its `m_ee` spectrum matches
the overlay truth Dalitz decays almost exactly), and reveals that
`G4DalitzDecayChannel` samples the lepton angle **isotropically** — no `1+cos²θ*`
term, vs EvtGen / QED which peaks at the poles. The overlay's apparent central
peaking in cos θ* is a soft-lepton truth-storage selection effect. See
`../plots/pi0_dalitz/geant4_standalone_validation.png`.

## Use in the reweighting

`../ipynb_notebooks/pi0_dalitz_reweighting.ipynb` uses this 1M sample as the source
and the 1M EvtGen sample as the target to build a 2D `w(m_ee, cos θ*)` weight
(N_eff ≈ 90%), then applies it to the boosted overlay decays for the lab-frame view.
