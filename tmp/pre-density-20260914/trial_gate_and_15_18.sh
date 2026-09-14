#!/usr/bin/env bash
set -uo pipefail
if bash tmp/pre-density-20260914/root_sampling_gate.sh; then
 printf 'ROOT_SAMPLING_GATE_PASSED commit=0a9c849\n'
else
 printf 'ROOT_SAMPLING_GATE_FAILED commit=0a9c849 training_preexperiment_remains_independent\n'
fi
exec bash tmp/pre-density-20260914/trial15_18.sh
