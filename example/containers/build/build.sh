#!/usr/bin/env bash
set -euo pipefail

# Use the INDEX passed in via environment.
INDEX=${INDEX:-"undefined"}

SRC_DIR="/src"      # Mounted folder with generated .pyx code.
BUILD_DIR="/build"  # Internal working directory.

# Set output folder for this build: e.g. /src/index/5/build
OUTPUT_DIR="$SRC_DIR/index/$INDEX/build"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/build.log"
META_FILE="$OUTPUT_DIR/build_metadata.json"

echo "Starting build at $(date) with index $INDEX" | tee -a "$LOG_FILE"

# Copy the .pyx file from SRC_DIR into BUILD_DIR.
cp "$SRC_DIR/fizzbuzz.pyx" "$BUILD_DIR/"

cd "$BUILD_DIR"
BUILD_OUTPUT=$(python setup.py build_ext --inplace 2>&1)
BUILD_EXIT_CODE=$?

echo "$BUILD_OUTPUT" | tee -a "$LOG_FILE"

if [ $BUILD_EXIT_CODE -eq 0 ]; then
    BUILD_STATUS="success"
else
    BUILD_STATUS="failure"
fi

# Find the compiled shared object (assumes Linux naming convention).
SO_FILE=$(find "$BUILD_DIR" -maxdepth 1 -type f -name "fizzbuzz*.so" | head -n 1)

if [ -z "$SO_FILE" ]; then
    COMPILED_FILE=""
else
    COMPILED_FILE=$(basename "$SO_FILE")
    # Copy the compiled artifact into the output folder.
    cp "$SO_FILE" "$OUTPUT_DIR/"
fi

# Write build metadata in JSON format.
cat <<EOF > "$META_FILE"
{
  "timestamp": "$(date --iso-8601=seconds)",
  "index": "$INDEX",
  "build_status": "$BUILD_STATUS",
  "compiled_file": "$COMPILED_FILE",
  "output_directory": "$OUTPUT_DIR",
  "log_file": "$LOG_FILE"
}
EOF

echo "Build metadata written to $META_FILE" | tee -a "$LOG_FILE"
exit $BUILD_EXIT_CODE
