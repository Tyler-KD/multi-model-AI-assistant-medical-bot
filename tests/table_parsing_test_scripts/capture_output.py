import sys
import os
import json
import time
from pathlib import Path
import subprocess

# Run the test script and capture output
output_file = "table_test_output.txt"
print(f"Running test and capturing output to {output_file}")

# Use subprocess to run the test
process = subprocess.run(
    ["python", "test_scripts/test_main_app.py"],
    capture_output=True,
    text=True
)

# Write results to file
with open(output_file, "w") as f:
    f.write("=== STDOUT ===\n")
    f.write(process.stdout)
    f.write("\n\n=== STDERR ===\n")
    f.write(process.stderr)

print(f"Test completed. Check {output_file} for results.") 