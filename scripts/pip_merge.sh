#!/usr/bin/env bash
# pip_merge.sh — Picture-in-picture video merge
#
# Usage:
#   bash scripts/pip_merge.sh <screenshare.mp4> <webcam.mp4> [output.mp4]
#
# Result: screenshare fills the frame; webcam sits in the bottom-right corner
#         at 22% of the screenshare width. Audio comes from webcam only.

set -euo pipefail

SCREENSHARE="${1:-}"
WEBCAM="${2:-}"
OUTPUT="${3:-output_pip.mp4}"

if [[ -z "$SCREENSHARE" || -z "$WEBCAM" ]]; then
    echo "Usage: bash scripts/pip_merge.sh <screenshare.mp4> <webcam.mp4> [output.mp4]"
    exit 1
fi

# PiP size: 22% of screenshare width, height auto-scaled, 20px margin from corner
PIP_SCALE="iw*0.22"

echo "Merging..."
echo "  Screenshare : $SCREENSHARE"
echo "  Webcam (PiP): $WEBCAM"
echo "  Output      : $OUTPUT"

ffmpeg -i "$SCREENSHARE" -i "$WEBCAM" \
    -filter_complex "
        [1:v]scale=${PIP_SCALE}:-1[pip];
        [0:v][pip]overlay=W-w-20:H-h-20
    " \
    -map 1:a \
    -c:v libx264 -preset fast -crf 18 \
    -c:a aac -b:a 192k \
    -movflags +faststart \
    "$OUTPUT"

echo ""
echo "Done: $OUTPUT"
