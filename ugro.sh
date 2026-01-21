#!/bin/bash
# UGRO wrapper script
UGRO_DIR="/"${HOME}/Development/Tools/ugro"
cd "$UGRO_DIR"
exec pixi run python -m ugro.cli "$@"
