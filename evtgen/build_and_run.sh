#!/bin/bash
# Build the EvtGen pi0 Dalitz generator against the conda evtgen env and run it.
#
# Usage: ./build_and_run.sh [n_events] [out_csv]
set -e

ENV_PREFIX="/nevis/riverside/data/leehagaman/ngem/evtgen_env"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

N_EVENTS="${1:-1000000}"
OUT_CSV="${2:-/nevis/riverside/data/leehagaman/ngem/other_files/evtgen_pi0_dalitz.csv}"

# locate EvtGen data files (DECAY.DEC, evt.pdl) inside the env
DECAY_DEC="$(find "$ENV_PREFIX/share" -name DECAY.DEC 2>/dev/null | head -1)"
EVT_PDL="$(find "$ENV_PREFIX/share" -name evt.pdl 2>/dev/null | head -1)"
echo "DECAY.DEC : $DECAY_DEC"
echo "evt.pdl   : $EVT_PDL"

if [ -z "$DECAY_DEC" ] || [ -z "$EVT_PDL" ]; then
    echo "ERROR: could not find DECAY.DEC / evt.pdl under $ENV_PREFIX/share" >&2
    exit 1
fi

# compile
echo "compiling pi0_dalitz_gen ..."
g++ -std=c++17 -O2 -DEVTGEN_CPP11 \
    "$HERE/pi0_dalitz_gen.cpp" \
    -I"$ENV_PREFIX/include" \
    -L"$ENV_PREFIX/lib" \
    -lEvtGen -lEvtGenExternal \
    -Wl,-rpath,"$ENV_PREFIX/lib" \
    -o "$HERE/pi0_dalitz_gen"

# run (LD_LIBRARY_PATH so dependent libs in the env are found)
echo "running ($N_EVENTS events) ..."
LD_LIBRARY_PATH="$ENV_PREFIX/lib:$LD_LIBRARY_PATH" \
    "$HERE/pi0_dalitz_gen" "$DECAY_DEC" "$EVT_PDL" \
    "$HERE/pi0_dalitz.dec" "$N_EVENTS" "$OUT_CSV"

echo "done -> $OUT_CSV"
