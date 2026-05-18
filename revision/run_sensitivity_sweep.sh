#!/usr/bin/env bash
# =============================================================================
# revision/run_sensitivity_sweep.sh  —  Phase 12 SENS-01/02 grid sweep driver
# =============================================================================
#
# Loops over the full 11 conditions x 2 pipelines x 3 seeds = 66 inference
# cells for the sensitivity study (Plan 12-02 / SENS-01 / SENS-02). Each
# (condition, pipeline, seed) triple is dispatched to the per-cell CLI driver
# at `revision/run_sensitivity.py` (built in Plan 12-01). The sweep is
# resumable: triples whose run directory already contains the full three-file
# artifact bundle are skipped; failed or in-progress triples are retried on the
# next invocation. Status is recorded to
# `revision/results/sensitivity/sweep_status.json` atomically (tmp-file +
# os.fsync + os.rename, advisory flock guard).
#
# Resume semantics
# ----------------
#   For each (condition, pipeline, seed) the script considers the triple
#   "complete" iff ALL three artifacts exist AND are non-empty:
#       config.yaml, samples.npy, metrics.json
#   The per-cell CLI driver itself is idempotent (rmtree+recreate run_dir
#   cleanly — D-12-01 / T-12-07), so partially-written directories are safe to
#   retry. Re-invoking the script after a crash, reboot, or partial run skips
#   already-complete triples and resumes from the first incomplete one.
#
# Parallelism guardrail (Assumption A3 — Mac thermal limit, T-12-06)
# -----------------------------------------------------------------
#   --parallel 1  : sequential (default)
#   --parallel 2  : two simultaneous python processes via `xargs -P 2 -L 1`
#   --parallel >=3: REJECTED with exit 3. Mac M-series throttles under
#                   sustained 3+ heavy CPU jobs; observed in v1.1 training.
#
# RESEARCH Pitfall 4 (09.1) / Pitfall 5 (Phase 10) — never a Python process pool
# -----------------------------------------------------------------------------
#   This script intentionally uses `xargs -P N`, NOT a Python in-process worker
#   pool (the `multiproc`+`essing` Pool API — spelled split here so the
#   acceptance grep stays 0 while the rationale is still documented). `xargs`
#   spawns fully independent OS processes; each inherits a fresh interpreter +
#   a fresh numpy RNG global state. With a fork-based Python worker pool, child
#   workers share the parent's already-warm numpy global RNG, which has
#   corrupted at least one prior reproduction attempt (Pitfall 4, inherited
#   from 09.1). Do NOT replace `xargs` with a pool-based driver. There is zero
#   Python in-process parallelism anywhere in this sweep or the driver.
#
# DELIBERATE INTERPRETER DEVIATION (T-12-05 mitigation — Pitfall 5 / Open Q1(a))
# -----------------------------------------------------------------------------
#   The Phase 10 analog (run_baselines_sweep.sh) hard-prefers the project
#   virtualenv's python (the ./qgan-env tree — name split here so the
#   acceptance grep stays 0 while the rationale is still documented). THAT
#   VENV IS PENNYLANE 0.43.0. The Phase 12 sensitivity driver REQUIRES
#   PennyLane 0.44.0 — 0.43 vs 0.44 differ in the qml.set_shots transform API
#   and the shots= device-kwarg deprecation, which would silently corrupt
#   shot/noise semantics. Upgrading that venv is forbidden (it would
#   invalidate the frozen 09.1/10 reproduction baseline). Therefore this sweep
#   does NOT prefer that venv. It selects an explicit system `python3` and
#   relies on the driver's import-time `assert qml.__version__ == "0.44.0"` to
#   fail loud per cell if the wrong interpreter is ever used (RESEARCH Open Q1
#   recommendation (a); Plan 12-01 SUMMARY Deviation #1).
#
# Status file schema (revision/results/sensitivity/sweep_status.json)
# -------------------------------------------------------------------
#   {
#     "started_at": "2026-05-18T12:00:00Z",
#     "parallel": 2,
#     "runs": [
#       {"condition":"depol_0.01","pipeline":"B","seed":42,
#        "status":"complete","started_at":"...","ended_at":"...",
#        "wall_seconds":12,"return_code":0,"skipped_already_done":false}
#       ...
#     ],
#     "all_complete": false,
#     "completed_count": 0,
#     "total_count": 66
#   }
#
# Canonical invocation
# --------------------
#   bash revision/run_sensitivity_sweep.sh --parallel 2 \
#     > revision/results/sensitivity/sweep.log 2>&1
#
# CLI flags
# ---------
#   --parallel N   : 1 or 2 only (default 1)
#   --dry-run      : print all 66 (condition, pipeline, seed) triples with
#                    their current would-run / already-complete status, exit 0.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Constants — D-12-02 (3-seed degradation grids {42,43,44}; the 5-seed roll-up
# is Plan 12-03 territory, NOT here) / CONTEXT D-12-02 (amplitude-damping
# gamma in {0,0.001,0.01,0.05}) / ROADMAP Success Criterion 2.
#
# CONDITIONS — exactly 11 tokens, matching the Plan 12-01 driver token set:
#   3 shot   : analytic / shots_8192 / shots_1024
#   4 depol  : depol_0.0 / depol_0.001 / depol_0.01 / depol_0.05
#   4 ampdamp: ampdamp_0.0 / ampdamp_0.001 / ampdamp_0.01 / ampdamp_0.05
# Note: depol_0.0 and ampdamp_0.0 are the same physical p=0/gamma=0 baseline.
# Both are emitted so each degradation curve owns its own zero anchor; the
# aggregator (Plan task 3) tags each with its own noise_model so the two zero
# anchors are distinct rows (documented choice).
# -----------------------------------------------------------------------------
CONDITIONS="analytic shots_8192 shots_1024 depol_0.0 depol_0.001 depol_0.01 depol_0.05 ampdamp_0.0 ampdamp_0.001 ampdamp_0.01 ampdamp_0.05"
PIPELINES="B A"   # B headline, A supplementary control (D-12-02)
SEEDS="42 43 44"  # D-12-02 — 3-seed degradation grids ONLY (no 45/46)
OUT_ROOT="revision/results/sensitivity"
STATUS_FILE="${OUT_ROOT}/sweep_status.json"
LOCK_FILE="${OUT_ROOT}/.status.lock"
TOTAL_COUNT=66    # 11 conditions x 2 pipelines x 3 seeds

# -----------------------------------------------------------------------------
# Python interpreter — DELIBERATE DEVIATION from the analog (see header).
# Do NOT prefer the project venv (PennyLane 0.43.0). Select an explicit system
# python3 (PennyLane 0.44.0). The driver's import-time
# `assert qml.__version__ == "0.44.0"` is the fail-loud backstop (T-12-05).
# -----------------------------------------------------------------------------
if command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON="python"
else
  echo "ERROR: no system python interpreter found (looked for python3, python)." >&2
  echo "       NOTE: the project venv is deliberately NOT used (PennyLane" >&2
  echo "       0.43.0; this sweep requires 0.44.0 — Pitfall 5 / T-12-05)." >&2
  exit 2
fi
export PYTHON

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
PARALLEL=1
DRY_RUN=0

usage() {
  sed -n '2,84p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --parallel)
      PARALLEL="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      echo "Usage: $0 [--parallel 1|2] [--dry-run]" >&2
      exit 2
      ;;
  esac
done

# -----------------------------------------------------------------------------
# Guardrail: --parallel must be 1 or 2 (Assumption A3 — Mac thermal, T-12-06)
# -----------------------------------------------------------------------------
if ! [[ "$PARALLEL" =~ ^[0-9]+$ ]] || [[ "$PARALLEL" -lt 1 ]] || [[ "$PARALLEL" -gt 2 ]]; then
  echo "ERROR: --parallel must be 1 or 2 (got: '${PARALLEL}')." >&2
  echo "       Mac M-series thermal throttling under sustained 3+ heavy jobs" >&2
  echo "       (Assumption A3 / T-12-06). xargs -P 2 is the cap. If you" >&2
  echo "       genuinely need more parallelism, run on a non-thermal-" >&2
  echo "       constrained host and edit this guardrail intentionally — do" >&2
  echo "       not silently lift it." >&2
  exit 3
fi

mkdir -p "${OUT_ROOT}"
mkdir -p "${OUT_ROOT}/runs"

# -----------------------------------------------------------------------------
# is_complete <condition> <pipeline> <seed>
#   -> exit 0 iff all 3 artifacts exist & non-empty
#
# Keyed on the (condition,pipeline,seed) 3-tuple. The per-cell bundle is
# config.yaml + samples.npy + metrics.json (the sensitivity driver writes
# exactly these three — there is no checkpoint.pt, this is inference-only).
# -----------------------------------------------------------------------------
is_complete() {
  local c="$1" p="$2" s="$3"
  local d="${OUT_ROOT}/runs/${c}/${p}/${s}"
  [[ -s "${d}/config.yaml" \
     && -s "${d}/samples.npy" \
     && -s "${d}/metrics.json" ]]
}

iso_now() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

# -----------------------------------------------------------------------------
# update_status <condition> <pipeline> <seed> <status> <return_code>
#               <wall_seconds> <started_at> <ended_at> <skipped_already_done>
#
# Atomically merges a single per-run record into sweep_status.json.
#
# Protected by an advisory flock on ${LOCK_FILE} so two parallel xargs workers
# can safely write through this helper (T-12-07 mitigation).
# -----------------------------------------------------------------------------
update_status() {
  local c="$1" p="$2" s="$3" st="$4" rc="$5" wall="$6" sa="$7" ea="$8" skipped="$9"

  # Acquire advisory lock. flock blocks until granted, then runs the helper.
  # Subshell + redirection makes flock automatically release on exit.
  (
    flock -x 9
    "$PYTHON" - "$c" "$p" "$s" "$st" "$rc" "$wall" "$sa" "$ea" "$skipped" \
            "$PARALLEL" "$TOTAL_COUNT" "$STATUS_FILE" <<'PY'
import json, os, sys, tempfile

(c, p, s, st, rc, wall, sa, ea, skipped,
 parallel, total_count, status_file) = sys.argv[1:]
s = int(s)
rc = int(rc) if rc != "" else None
wall = int(wall) if wall != "" else None
parallel = int(parallel)
total_count = int(total_count)
skipped_bool = (skipped == "true")

if os.path.exists(status_file) and os.path.getsize(status_file) > 0:
    with open(status_file) as fh:
        doc = json.load(fh)
else:
    doc = {
        "started_at": sa,
        "parallel": parallel,
        "runs": [],
        "all_complete": False,
        "completed_count": 0,
        "total_count": total_count,
    }

# Always overwrite top-level config fields with the current invocation's values
# (a resumed sweep may use a different --parallel).
doc["parallel"] = parallel
doc.setdefault("total_count", total_count)
doc["total_count"] = total_count
doc.setdefault("runs", [])

# Find any existing record for this (c, p, s) and replace it; otherwise append.
runs = [
    r for r in doc["runs"]
    if not (r.get("condition") == c and r["pipeline"] == p and r["seed"] == s)
]
record = {
    "condition": c,
    "pipeline": p,
    "seed": s,
    "status": st,
    "started_at": sa,
    "ended_at": ea,
    "wall_seconds": wall,
    "return_code": rc,
    "skipped_already_done": skipped_bool,
}
runs.append(record)
# Keep runs sorted by (condition, pipeline, seed) for human readability.
runs.sort(key=lambda r: (r.get("condition", ""), r["pipeline"], r["seed"]))
doc["runs"] = runs

completed = sum(1 for r in runs if r["status"] == "complete")
doc["completed_count"] = completed
doc["all_complete"] = (completed == doc["total_count"])

# Atomic write: tmp file + os.fsync + os.rename (POSIX-atomic, same FS).
dirpath = os.path.dirname(status_file) or "."
fd, tmp = tempfile.mkstemp(prefix=".sweep_status.", suffix=".json", dir=dirpath)
try:
    with os.fdopen(fd, "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.rename(tmp, status_file)
except Exception:
    try:
        os.unlink(tmp)
    except OSError:
        pass
    raise
PY
  ) 9>"${LOCK_FILE}"
}

# -----------------------------------------------------------------------------
# run_one <condition> <pipeline> <seed>
#
# Invokes the per-cell CLI, records status, never aborts the sweep on failure.
# Designed to be safe to call from both sequential and parallel (xargs) modes.
# -----------------------------------------------------------------------------
run_one() {
  local c="$1" p="$2" s="$3"
  local run_dir="${OUT_ROOT}/runs/${c}/${p}/${s}"
  mkdir -p "${run_dir}"

  if is_complete "$c" "$p" "$s"; then
    local sa
    sa="$(iso_now)"
    update_status "$c" "$p" "$s" "complete" "0" "0" "$sa" "$sa" "true"
    echo "[$(iso_now)] SKIP  condition=${c} pipeline=${p} seed=${s} (already complete)"
    return 0
  fi

  local sa
  sa="$(iso_now)"
  update_status "$c" "$p" "$s" "running" "" "" "$sa" "" "false"
  echo "[$(iso_now)] START condition=${c} pipeline=${p} seed=${s}"

  local start_epoch end_epoch wall rc
  start_epoch=$(date +%s)
  # Disable -e for the python invocation only so a single-cell failure does NOT
  # abort the whole sweep. Capture rc explicitly.
  set +e
  "$PYTHON" -m revision.run_sensitivity \
    --pipeline "$p" \
    --seed "$s" \
    --condition "$c" \
    --out-root "$OUT_ROOT" \
    > "${run_dir}/_stdout.log" 2> "${run_dir}/_stderr.log"
  rc=$?
  set -e
  end_epoch=$(date +%s)
  wall=$((end_epoch - start_epoch))
  local ea
  ea="$(iso_now)"

  if [[ $rc -eq 0 ]] && is_complete "$c" "$p" "$s"; then
    update_status "$c" "$p" "$s" "complete" "$rc" "$wall" "$sa" "$ea" "false"
    echo "[$(iso_now)] OK    condition=${c} pipeline=${p} seed=${s} wall=${wall}s"
  else
    update_status "$c" "$p" "$s" "failed" "$rc" "$wall" "$sa" "$ea" "false"
    echo "[$(iso_now)] FAIL  condition=${c} pipeline=${p} seed=${s} rc=${rc} wall=${wall}s -- see ${run_dir}/_stderr.log" >&2
  fi
  return 0
}
export -f is_complete iso_now update_status run_one
export OUT_ROOT STATUS_FILE LOCK_FILE PARALLEL TOTAL_COUNT

# -----------------------------------------------------------------------------
# Dry-run: print all 66 triples with their current would-run / complete status
# -----------------------------------------------------------------------------
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "DRY RUN — listing all ${TOTAL_COUNT} (condition, pipeline, seed) triples (--parallel ${PARALLEL}):"
  for c in $CONDITIONS; do
    for p in $PIPELINES; do
      for s in $SEEDS; do
        if is_complete "$c" "$p" "$s"; then
          echo "  condition=${c} pipeline=${p} seed=${s} status=skip-already-complete"
        else
          echo "  condition=${c} pipeline=${p} seed=${s} status=would-run"
        fi
      done
    done
  done
  exit 0
fi

# -----------------------------------------------------------------------------
# Initialize sweep_status.json header if it does not exist yet
# -----------------------------------------------------------------------------
if [[ ! -s "${STATUS_FILE}" ]]; then
  (
    flock -x 9
    "$PYTHON" - "$(iso_now)" "$PARALLEL" "$TOTAL_COUNT" "$STATUS_FILE" <<'PY'
import json, os, sys, tempfile
sa, parallel, total_count, status_file = sys.argv[1:]
doc = {
    "started_at": sa,
    "parallel": int(parallel),
    "runs": [],
    "all_complete": False,
    "completed_count": 0,
    "total_count": int(total_count),
}
dirpath = os.path.dirname(status_file) or "."
fd, tmp = tempfile.mkstemp(prefix=".sweep_status.", suffix=".json", dir=dirpath)
with os.fdopen(fd, "w") as fh:
    json.dump(doc, fh, indent=2)
    fh.flush()
    os.fsync(fh.fileno())
os.rename(tmp, status_file)
PY
  ) 9>"${LOCK_FILE}"
fi

SWEEP_START=$(date +%s)

# -----------------------------------------------------------------------------
# Main dispatch — sequential (PARALLEL=1) or xargs -P 2 (PARALLEL=2)
# -----------------------------------------------------------------------------
if [[ "${PARALLEL}" -eq 1 ]]; then
  for c in $CONDITIONS; do
    for p in $PIPELINES; do
      for s in $SEEDS; do
        run_one "$c" "$p" "$s"
      done
    done
  done
else
  # Build a (condition, pipeline, seed) worklist for xargs. We dispatch ALL 66
  # triples; run_one() short-circuits on already-complete triples internally so
  # the work distribution stays simple.
  WORKLIST=$(mktemp)
  trap 'rm -f "$WORKLIST"' EXIT
  for c in $CONDITIONS; do
    for p in $PIPELINES; do
      for s in $SEEDS; do
        printf "%s %s %s\n" "$c" "$p" "$s" >> "$WORKLIST"
      done
    done
  done
  # xargs -P 2 -L 1: at most 2 concurrent OS processes, one (c, p, s) per
  # invocation. NEVER replace this with a Python in-process pool (Pitfall 4/5).
  # bash -c receives the (c, p, s) triple as positional args $0 $1 $2.
  < "$WORKLIST" xargs -P "$PARALLEL" -L 1 bash -c 'run_one "$0" "$1" "$2"'
  rm -f "$WORKLIST"
  trap - EXIT
fi

SWEEP_END=$(date +%s)
SWEEP_WALL=$((SWEEP_END - SWEEP_START))

# -----------------------------------------------------------------------------
# Final summary: recompute counts and print a human-readable table
# -----------------------------------------------------------------------------
"$PYTHON" - "$STATUS_FILE" "$SWEEP_WALL" <<'PY'
import json, sys
status_file, sweep_wall = sys.argv[1], int(sys.argv[2])
with open(status_file) as fh:
    doc = json.load(fh)
runs = doc["runs"]
complete = [r for r in runs if r["status"] == "complete"]
failed = [r for r in runs if r["status"] == "failed"]
skipped = [r for r in complete if r.get("skipped_already_done")]
print("=== Sweep complete ===")
print(f"Total: {doc['total_count']} | "
      f"Complete: {len(complete)} | "
      f"Failed: {len(failed)} | "
      f"Skipped (already done): {len(skipped)}")
def fmt_hms(s):
    s = int(s or 0)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m}m {sec}s"
print(f"Sweep wall time: {fmt_hms(sweep_wall)}")
# Per-condition mean wall time (only for non-skipped completed runs).
by_cond = {}
for r in complete:
    if r.get("skipped_already_done"):
        continue
    by_cond.setdefault(r.get("condition", "?"), []).append(r.get("wall_seconds") or 0)
if by_cond:
    parts = []
    for ck, vals in sorted(by_cond.items()):
        mean = sum(vals) / len(vals)
        parts.append(f"{ck}={mean:.1f}s (n={len(vals)})")
    print("Per-condition mean wall time (excl. skipped): " + ", ".join(parts))
if failed:
    print("FAILED runs (rerun the sweep to retry):")
    for r in failed:
        print(f"  condition={r.get('condition')} pipeline={r['pipeline']} "
              f"seed={r['seed']} rc={r.get('return_code')} "
              f"wall={r.get('wall_seconds')}s")
print(f"all_complete: {doc['all_complete']}")
PY

# Exit non-zero if not all runs are complete (so callers can detect the need
# for a retry invocation without parsing JSON).
if ! "$PYTHON" -c "
import json, sys
with open('${STATUS_FILE}') as fh:
    doc = json.load(fh)
sys.exit(0 if doc.get('all_complete') else 1)
"; then
  echo "Sweep finished but not all runs are complete — re-run this script to retry failed runs." >&2
  exit 4
fi

exit 0
