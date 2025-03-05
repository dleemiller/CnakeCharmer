#!/usr/bin/env bash
set -euo pipefail

INDEX=${INDEX:-"undefined"}

DATA_DIR="/data"  # Mounted folder where build artifacts reside.
OUTPUT_DIR="$DATA_DIR/index/$INDEX/run"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/run.log"
META_FILE="$OUTPUT_DIR/run_metadata.json"

echo "Starting run at $(date) with index $INDEX" | tee -a "$LOG_FILE"
cd "$DATA_DIR"

# Run the module and capture its output.
# This example calls fizzbuzz.fizzbuzz(15) and prints the results.
RUN_OUTPUT=$(python -c 'import fizzbuzz; print("\n".join(fizzbuzz.fizzbuzz(15)))' 2>&1)
RUN_EXIT_CODE=$?

echo "$RUN_OUTPUT" | tee -a "$LOG_FILE"

if [ $RUN_EXIT_CODE -eq 0 ]; then
    RUN_STATUS="success"
else
    RUN_STATUS="failure"
fi

# Write run metadata.
cat <<EOF > "$META_FILE"
{
  "timestamp": "$(date --iso-8601=seconds)",
  "index": "$INDEX",
  "run_status": "$RUN_STATUS",
  "output_directory": "$OUTPUT_DIR",
  "log_file": "$LOG_FILE"
}
EOF

echo "Run metadata written to $META_FILE" | tee -a "$LOG_FILE"
exit $RUN_EXIT_CODE
