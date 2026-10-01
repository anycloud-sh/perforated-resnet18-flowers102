"""Require the committed Lambda result to describe the promoted digest."""

import json
import os
import re
from pathlib import Path


image_digest = os.environ["DIGEST"]
assert re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest)
result = json.loads(Path("validation/lambda-a10.json").read_text())
assert result["image_digest"] == image_digest
assert result["source_revision"] == "9d317e629428d73d92b88dc501f0dce3b1017c86"
assert result["model_revision"] == "8a59acfc5285e72257e8d322da51bf486c3ce060"
assert result["model_sha256"] == "114d3ed72896110ff229bd5a7c4ae8e7d048eb185ba1cdeb75cd76557d8674bc"
assert result["provider"] == "Lambda"
assert result["cuda_synchronized"] is True
assert result["epoch_count"] == 50
assert len(result["baseline_accuracy"]) == 50
assert len(result["perforated_accuracy"]) == 50
assert result["job_state"] == "completed"
print("Promotion evidence matches the validated digest and 50-epoch GPU run.")
