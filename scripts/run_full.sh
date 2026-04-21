#!/usr/bin/env bash
# Full-scale pipeline entry point for 5× H200. Forks 5 parallel Phase A
# workers (one per GPU) + 1 master barrier tmux. The master runs global
# select_data and auto-launches Phase B (finetune + eval).
#
# Usage:   bash scripts/run_full.sh
# Monitor: tail -f logs/phaseA_g{0..4}_*.log     (Phase A workers)
#          tail -f logs/phaseA_master_*.log      (barrier + select_data + launch)
#          tail -f logs/ft_g{0..4}_*.log         (Phase B, after Phase A)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"

bash scripts/launch_phaseA_parallel.sh "${STAMP}"

echo ""
echo "All tmuxes launched. Useful commands:"
echo "  tmux ls"
echo "  tail -f logs/phaseA_g*_${STAMP}.log"
echo "  tail -f logs/phaseA_master_${STAMP}.log"
echo "  tail -f logs/ft_g*_${STAMP}.log   (once Phase A finishes)"
