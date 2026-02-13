"""Test is_in conversion alone vs in AND."""
import os
os.environ["POLARS_VERBOSE"] = "1"
os.environ["POLARS_VERBOSE_SENSITIVE"] = "1"

import polars as pl
import pyarrow.dataset as ds

dataset = ds.dataset(
    pl.DataFrame({
        "department": ["Engineering", "Sales", "HR", "Engineering", "Sales"],
        "salary": [80000, 65000, 95000, 70000, 110000],
    }).to_arrow()
)

# Case 1: is_in alone
print("=" * 60)
print("CASE 1: is_in alone")
print("=" * 60)
lf = pl.scan_pyarrow_dataset(dataset).filter(
    pl.col("department").is_in(["Engineering", "HR"])
)
result = lf.collect()
print(f"Result: {len(result)} rows")
print()

# Case 2: simple equality (for comparison)
print("=" * 60)
print("CASE 2: equality (for comparison)")
print("=" * 60)
lf = pl.scan_pyarrow_dataset(dataset).filter(
    pl.col("department") == "Engineering"
)
result = lf.collect()
print(f"Result: {len(result)} rows")
