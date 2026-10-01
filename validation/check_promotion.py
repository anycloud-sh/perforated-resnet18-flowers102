"""Require the committed Lambda result to describe the promoted digest."""

import hashlib
import json
import math
import os
import re
from pathlib import Path


image_digest = os.environ["DIGEST"]
assert re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest)
result = json.loads(Path("validation/lambda-a10.json").read_text())
log = Path("validation/lambda-a10.log").read_bytes()
assert result["image_digest"] == image_digest
assert result["artifact_revision"] == "24c756da650de6f5105a378a10b6ed9e6f4003de"
assert result["source_revision"] == "9d317e629428d73d92b88dc501f0dce3b1017c86"
assert result["model_revision"] == "8a59acfc5285e72257e8d322da51bf486c3ce060"
assert result["model_sha256"] == "114d3ed72896110ff229bd5a7c4ae8e7d048eb185ba1cdeb75cd76557d8674bc"
assert result["provider"] == "Lambda"
assert result["gpu"] == "NVIDIA A10"
assert result["repository_public"] is True
assert result["package_linked"] is True
assert result["anonymous_pull"] is True
assert result["anonymous_manifest_http_status"] == 200
assert result["anonymous_manifest_digest"] == image_digest
assert result["cuda_synchronized"] is True
assert result["cuda_allocated_bytes"] > 0
assert math.isfinite(result["cuda_matmul_checksum"])
assert result["epoch_count"] == 50
assert result["job_state"] == "completed"
assert hashlib.sha256(log).hexdigest() == result["log_sha256"]
assert b"CUDA_PROOF device='NVIDIA A10'" in log
assert b"synchronized=true" in log
assert b"===== Final Comparison =====" in log
assert b"[0/1020 (0%)]" in log
assert b"/6149" in log

rows = []
for line in log.decode("utf-8").splitlines():
    match = re.fullmatch(
        r"\s*(\d+)\s+(\d+\.\d{2})\s+(?:N/A|[+-]\d+\.\d{2})"
        r"\s+(\d+\.\d{2})\s+(?:N/A|[+-]\d+\.\d{2})\s*",
        line,
    )
    if match:
        rows.append((int(match[1]), float(match[2]), float(match[3])))
assert [row[0] for row in rows] == list(range(1, 51))
assert [row[1] for row in rows] == result["baseline_accuracy"]
assert [row[2] for row in rows] == result["perforated_accuracy"]
assert all(math.isfinite(score) and 0 <= score <= 100 for row in rows for score in row[1:])
print("Promotion evidence matches the validated digest and 50-epoch GPU run.")
