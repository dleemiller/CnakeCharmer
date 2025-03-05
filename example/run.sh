#!/usr/bin/env bash
set -e

INDEX_DIR="./code/cython/index"
NEXT_FILE="$INDEX_DIR/next.txt"

# Ensure our index folder exists.
mkdir -p "$INDEX_DIR"

# If next.txt doesn't exist, initialize it to 5.
if [ ! -f "$NEXT_FILE" ]; then
    echo 5 > "$NEXT_FILE"
fi

INDEX=$(cat "$NEXT_FILE")
echo "Using build/run index: $INDEX"

# Increment index by 5 for next run.
NEXT_INDEX=$((INDEX + 5))
echo "$NEXT_INDEX" > "$NEXT_FILE"

echo "Starting build process..."
docker-compose run --rm -e INDEX="$INDEX" build

echo "Build completed. Check './code/cython/index/$INDEX/build' for logs and metadata."

echo "Starting run process..."
docker-compose run --rm -e INDEX="$INDEX" run

echo "Run completed. Check './code/cython/index/$INDEX/run' for logs and metadata."
