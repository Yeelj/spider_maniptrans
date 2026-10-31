#!/usr/bin/env bash
# Launch SPIDER playback with the Python, libraries, and compiler verified here.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RETARGET_ENV="${RETARGET_ENV:-$HOME/miniconda3/envs/physgraph}"
RETARGET_CC="${RETARGET_CC:-/usr/bin/gcc-11}"
RETARGET_CXX="${RETARGET_CXX:-/usr/bin/g++-11}"

for executable in "$RETARGET_ENV/bin/python" "$RETARGET_CC" "$RETARGET_CXX"; do
    if [[ ! -x "$executable" ]]; then
        printf 'Required executable not found: %s\nSet RETARGET_ENV, RETARGET_CC, or RETARGET_CXX explicitly.\n' "$executable" >&2
        exit 1
    fi
done

export PATH="$RETARGET_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$RETARGET_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Conda activation sets CXX to a compiler whose sysroot lacks crypt.h.
# Select the system toolchain explicitly rather than inheriting that setting.
export CC="$RETARGET_CC"
export CXX="$RETARGET_CXX"
export MAX_JOBS="${MAX_JOBS:-2}"

cd "$script_dir/.."
exec "$RETARGET_ENV/bin/python" "$script_dir/visualize_taco_retarget.py" \
    --reference_source spider "$@"
