import os
import time
import asyncio
import aiohttp
import gzip
import pyarrow.parquet as pq
import duckdb
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Configuration
PARQUET_PATH = "the-stack-v2-dedup-cython.parquet"
DB_PATH = "stack_cython.duckdb"
OUTPUT_DIR = "downloaded_content"
BASE_URL = "https://softwareheritage.s3.amazonaws.com/content/"
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
BATCH_SIZE = 500  # Number of rows to process before committing to DB
MAX_CONCURRENT_REQUESTS = 20  # Maximum concurrent downloads
SKIP_LARGE_FILES = True  # Skip files larger than MAX_FILE_SIZE
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Initialize DuckDB connection
conn = duckdb.connect(DB_PATH)

# Check if the table already exists
table_exists = (
    conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='stack_cython'"
    ).fetchone()
    is not None
)

if table_exists:
    # If table exists, check if content column is TEXT
    try:
        # Try to alter the column type if needed
        conn.execute("ALTER TABLE stack_cython ALTER COLUMN content TYPE TEXT")
    except:
        print(
            "Note: Could not alter content column. If you encounter type errors, you may need to recreate the database."
        )

# Check if our table exists, if not create it from the parquet file
conn.execute(
    "CREATE TABLE IF NOT EXISTS stack_cython AS SELECT *, NULL::TEXT AS content FROM '"
    + PARQUET_PATH
    + "'"
)


# Function to safely decompress gzipped content
def decompress_if_needed(content):
    # Check for gzip magic bytes (starts with 0x1f 0x8b)
    if content and len(content) > 2 and content[0] == 0x1F and content[1] == 0x8B:
        try:
            return gzip.decompress(content)
        except Exception as e:
            print(f"Error decompressing content: {e}")
    return content  # Return original if not gzipped or decompression fails


# Async function to download content
async def download_content(session, blob_id, src_encoding, semaphore):
    url = BASE_URL + blob_id

    async with semaphore:  # Limit concurrent requests
        for attempt in range(MAX_RETRIES):
            try:
                async with session.get(url, timeout=30) as response:
                    if response.status != 200:
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(RETRY_DELAY)
                            continue
                        return None, blob_id

                    # Check content size before downloading fully
                    content_length = response.headers.get("Content-Length")
                    if (
                        content_length
                        and SKIP_LARGE_FILES
                        and int(content_length) > MAX_FILE_SIZE
                    ):
                        return (
                            f"SKIPPED_LARGE_FILE:{int(content_length) // 1024}KB",
                            blob_id,
                        )

                    # Download the content
                    content = await response.read()

                    # Decompress if it's gzipped
                    content = decompress_if_needed(content)

                    # Try to decode the content with the provided encoding
                    try:
                        text_content = content.decode(src_encoding)
                        return text_content, blob_id
                    except UnicodeDecodeError:
                        # Fallback to utf-8
                        try:
                            text_content = content.decode("utf-8")
                            return text_content, blob_id
                        except UnicodeDecodeError:
                            # If we can't decode it after decompression, we'll skip it
                            return "SKIPPED_BINARY_CONTENT", blob_id

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    return f"ERROR:{str(e)}", blob_id

        return None, blob_id


# Get count of rows that need processing
result = conn.execute(
    "SELECT COUNT(*) FROM stack_cython WHERE content IS NULL"
).fetchone()
total_rows = result[0]
print(f"Found {total_rows} rows that need content downloaded")


# Function to update database with downloaded content
def update_db(results):
    success_count = 0
    error_count = 0
    skipped_count = 0

    for content, blob_id in results:
        if content is None:
            error_count += 1
        elif content.startswith("SKIPPED_"):
            skipped_count += 1
            conn.execute(
                """
                UPDATE stack_cython
                SET content = ?
                WHERE blob_id = ?
            """,
                [content, blob_id],
            )
        elif content.startswith("ERROR:"):
            error_count += 1
            conn.execute(
                """
                UPDATE stack_cython
                SET content = ?
                WHERE blob_id = ?
            """,
                [content, blob_id],
            )
        else:
            success_count += 1
            conn.execute(
                """
                UPDATE stack_cython
                SET content = ?
                WHERE blob_id = ?
            """,
                [content, blob_id],
            )

    conn.commit()
    return success_count, error_count, skipped_count


# Main async processing function
async def process_batch(batch_data):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for blob_id, src_encoding in batch_data:
            task = download_content(session, blob_id, src_encoding, semaphore)
            tasks.append(task)

        # Process downloads with progress bar
        results = []
        for future in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Downloading"
        ):
            result = await future
            results.append(result)

        return results


# Async main process
async def main():
    total_processed = 0
    total_success = 0
    total_errors = 0
    total_skipped = 0

    while True:
        # Get blob_ids that need to be processed
        result = conn.execute(
            """
            SELECT blob_id, src_encoding
            FROM stack_cython 
            WHERE content IS NULL
            LIMIT ?
        """,
            [BATCH_SIZE],
        ).fetchall()

        if not result:
            break  # No more rows to process

        print(f"Processing batch of {len(result)} items...")
        results = await process_batch(result)

        # Update database with results
        success, errors, skipped = update_db(results)
        total_success += success
        total_errors += errors
        total_skipped += skipped

        total_processed += len(result)
        print(f"Processed {total_processed}/{total_rows} files")
        print(f"  Success: {success}, Errors: {errors}, Skipped: {skipped}")

    print(f"Download complete! Processed {total_processed} files.")
    print(
        f"Total success: {total_success}, Total errors: {total_errors}, Total skipped: {total_skipped}"
    )


# Run the async process
if __name__ == "__main__":
    print("Starting download process...")
    asyncio.run(main())

    # Query to check completion status
    completion_stats = conn.execute(
        """
        SELECT 
            COUNT(*) AS total_rows,
            SUM(CASE WHEN content IS NOT NULL THEN 1 ELSE 0 END) AS completed_rows,
            SUM(CASE WHEN content LIKE 'SKIPPED_%' THEN 1 ELSE 0 END) AS skipped_rows,
            SUM(CASE WHEN content LIKE 'ERROR:%' THEN 1 ELSE 0 END) AS error_rows,
            (SUM(CASE WHEN content IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) AS completion_percentage
        FROM stack_cython
    """
    ).fetchone()

    print(f"Total rows: {completion_stats[0]}")
    print(f"Completed rows: {completion_stats[1]}")
    print(f"  - Skipped binary/large files: {completion_stats[2]}")
    print(f"  - Errors: {completion_stats[3]}")
    print(f"Completion percentage: {completion_stats[4]:.2f}%")

    # Close the connection
    conn.close()
