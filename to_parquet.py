import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

input_jsonl = "data.jsonl"
output_parquet = "data.parquet"

CHUNK_SIZE = 50_000

writer = None
chunk = []
total_rows = 0
skipped = 0

print("Converting in chunks...")
with open(input_jsonl, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            chunk.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  Skipping line {line_num}: {e}")
            skipped += 1
            continue

        if len(chunk) >= CHUNK_SIZE:
            df = pd.DataFrame(chunk)
            table = pa.Table.from_pandas(df)
            if writer is None:
                writer = pq.ParquetWriter(output_parquet, table.schema)
            writer.write_table(table)
            total_rows += len(chunk)
            print(f"  Written {total_rows} rows so far...")
            chunk = []

# Write any remaining rows
if chunk:
    df = pd.DataFrame(chunk)
    table = pa.Table.from_pandas(df)
    if writer is None:
        writer = pq.ParquetWriter(output_parquet, table.schema)
    writer.write_table(table)
    total_rows += len(chunk)

if writer:
    writer.close()

print(f"Done! {total_rows} rows written, {skipped} bad lines skipped.")