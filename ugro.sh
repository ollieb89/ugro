#!/bin/bash
# UGRO wrapper script
UGRO_DIR="/home/ollie/Development/Tools/ugro"
cd "$UGRO_DIR"
exec pixi run python -m ugro.cli "$@"
