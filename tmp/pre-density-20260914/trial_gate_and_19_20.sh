#!/usr/bin/env bash
set -uo pipefail
if bash tmp/pre-density-20260914/root_sampling_gate.sh; then
 printf 'ROOT_SAMPLING_GATE_PASSED commit=0a9c849 shell_lf_normalized\n'
else
 printf 'ROOT_SAMPLING_GATE_FAILED commit=0a9c849 shell_lf_normalized\n'
fi
exec bash tmp/pre-density-20260914/trial19_20.sh
