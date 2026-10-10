#!/usr/bin/env bash
# Resumable overnight driver for the GPU test runs:
#
#   bash scripts/test/run_overnight.sh csgm     [num_images]   # scripts/test/run_csgm_test.py,     400 runs, ~7.5 h
#   bash scripts/test/run_overnight.sh iadrift  [num_images]   # scripts/test/run_iadrift_test.py,  200 runs, ~3.5 h
#   bash scripts/test/run_overnight.sh ensemble [num_images]   # scripts/test/run_ensemble_test.py, 300 runs, ~5.5 h
#
# Runs the test script in chunks of MAX_RUNS_PER_CHUNK runs until a chunk prints "ALL RUNS COMPLETE". Short-lived
# processes are deliberate: one 32-run IADrift chunk segfaulted after 16 runs (protocol Section 8). Each finished run
# is saved atomically, so a crashed chunk is retried and resumes from the checkpoint. A chunk counts as failed only if
# it saved no new run (checked with the runner's --dry-run); after MAX_FAILURES failed chunks in a row the loop stops.
# Run it inside tmux.
#
# Log: <repo>/logs/<name>_test.log. Python: $PYTHON, else <repo>/.venv/bin/python, else python3.
set -u

NAME="${1:-}"
NUM_IMAGES="${2:-100}"
case "$NAME" in
    csgm|iadrift|ensemble) ;;
    *) echo "usage: bash scripts/test/run_overnight.sh {csgm|iadrift|ensemble} [num_images]" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -z "${PYTHON:-}" ]; then
    if [ -x "$REPO/.venv/bin/python" ]; then PYTHON="$REPO/.venv/bin/python"; else PYTHON=python3; fi
fi
RUNNER="$REPO/scripts/test/run_${NAME}_test.py"
LOG="$REPO/logs/${NAME}_test.log"
MAX_RUNS_PER_CHUNK=10
MAX_FAILURES=3

# CSGM runs reproduce the saved runs only with the default allocator (the runners refuse 'platform');
# run_iadrift_test.py sets XLA_PYTHON_CLIENT_ALLOCATOR=platform itself.
unset XLA_PYTHON_CLIENT_ALLOCATOR
mkdir -p "$REPO/logs"
cd "$REPO" || exit 1

saved_runs() {
    JAX_PLATFORMS=cpu "$PYTHON" "$RUNNER" --dry-run --num-images "$NUM_IMAGES" 2>/dev/null \
        | sed -n 's/.*already saved: \([0-9]*\).*/\1/p' | head -n 1
}

failures=0
chunk=0
echo "=== $(date '+%F %T') $NAME test started/resumed (num_images=$NUM_IMAGES, saved $(saved_runs)) ===" | tee -a "$LOG"

while true; do
    chunk=$((chunk + 1))
    before=$(saved_runs)
    echo "--- $(date '+%F %T') chunk $chunk starting (saved ${before:-?})" | tee -a "$LOG"
    chunk_log=$(mktemp)
    "$PYTHON" -u "$RUNNER" --num-images "$NUM_IMAGES" --max-runs "$MAX_RUNS_PER_CHUNK" > "$chunk_log" 2>&1
    status=$?
    cat "$chunk_log" >> "$LOG"
    after=$(saved_runs)
    echo "--- $(date '+%F %T') chunk $chunk finished with exit code $status (saved ${after:-?})" | tee -a "$LOG"

    if grep -q "ALL RUNS COMPLETE" "$chunk_log"; then
        rm -f "$chunk_log"
        echo "=== $(date '+%F %T') ALL RUNS COMPLETE ===" | tee -a "$LOG"
        break
    fi
    rm -f "$chunk_log"
    if [ -n "$before" ] && [ -n "$after" ] && [ "$after" -gt "$before" ]; then
        failures=0
        [ "$status" -ne 0 ] && echo "!!! chunk crashed after progress (exit $status); continuing" | tee -a "$LOG"
    else
        failures=$((failures + 1))
        echo "!!! chunk made no progress ($failures/$MAX_FAILURES consecutive; exit $status); retrying in 60 s" \
            | tee -a "$LOG"
        if [ "$failures" -ge "$MAX_FAILURES" ]; then
            echo "=== $(date '+%F %T') STOPPED after $MAX_FAILURES consecutive chunks without progress; see $LOG ===" \
                | tee -a "$LOG"
            exit 1
        fi
        sleep 60
    fi
done
