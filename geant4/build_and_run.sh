#!/bin/bash
# Build the standalone Geant4 pi0 Dalitz generator against the conda geant4 env
# and run it. Decay-only: no G4 data files needed.
#
# Usage: ./build_and_run.sh [n_events] [out_csv] [seed]
set -e

ENV_PREFIX="/nevis/riverside/data/leehagaman/ngem/geant4_env"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

N_EVENTS="${1:-1000000}"
OUT_CSV="${2:-/nevis/riverside/data/leehagaman/ngem/other_files/geant4_pi0_dalitz.csv}"
SEED="${3:-12345}"

G4CONFIG="$ENV_PREFIX/bin/geant4-config"
if [ ! -x "$G4CONFIG" ]; then
    echo "ERROR: geant4-config not found at $G4CONFIG" >&2
    exit 1
fi
echo "Geant4 version: $($G4CONFIG --version)"

echo "compiling g4_dalitz_gen ..."
g++ -std=c++17 -O2 "$HERE/g4_dalitz_gen.cpp" \
    $($G4CONFIG --cflags) $($G4CONFIG --libs) \
    -Wl,-rpath,"$ENV_PREFIX/lib" \
    -o "$HERE/g4_dalitz_gen"

# set the Geant4 data-file env vars (G4ENSDFSTATEDATA etc.) needed at runtime
export CONDA_PREFIX="$ENV_PREFIX"
for f in "$ENV_PREFIX"/etc/conda/activate.d/*.sh; do
    [ -f "$f" ] && source "$f"
done

echo "running ($N_EVENTS events, seed $SEED) ..."
LD_LIBRARY_PATH="$ENV_PREFIX/lib:$LD_LIBRARY_PATH" \
    "$HERE/g4_dalitz_gen" "$N_EVENTS" "$OUT_CSV" "$SEED"

echo "done -> $OUT_CSV"
