"""Require full Lambda and AWS Spot evidence for the v0.2.0 digest."""

import hashlib
import json
import math
import os
import re
from pathlib import Path


digest = os.environ["DIGEST"]
assert re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
artifact_revision = None

for prefix, provider in (("lambda-spot-image", "Lambda"), ("aws-spot", "AWS")):
    result = json.loads(Path(f"validation/{prefix}.json").read_text())
    log = Path(f"validation/{prefix}.log").read_bytes()
    assert result["image_digest"] == digest
    assert result["provider"] == provider
    assert result["job_state"] == "completed"
    assert result["epoch_count"] == 50
    assert result["source_revision"] == "9d317e629428d73d92b88dc501f0dce3b1017c86"
    assert result["model_revision"] == "8a59acfc5285e72257e8d322da51bf486c3ce060"
    assert result["model_sha256"] == "114d3ed72896110ff229bd5a7c4ae8e7d048eb185ba1cdeb75cd76557d8674bc"
    assert re.fullmatch(r"[0-9a-f]{40}", result["artifact_revision"])
    assert result["candidate_tag"] == f"candidate-{result['artifact_revision']}"
    if artifact_revision is None:
        artifact_revision = result["artifact_revision"]
    assert result["artifact_revision"] == artifact_revision
    assert result["log_sha256"] == hashlib.sha256(log).hexdigest()
    assert result["anonymous_pull"] is True
    assert result["repository_public"] is True
    assert result["package_linked"] is True
    assert result["anonymous_manifest_digest"] == digest
    assert b"CUDA_PROOF device=" in log and b"synchronized=true" in log
    assert b"===== Final Comparison =====" in log
    assert b"[0/1020 (0%)]" in log and b"/6149" in log

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
    assert all(math.isfinite(value) and 0 <= value <= 100 for row in rows for value in row[1:])

    if provider == "AWS":
        assert result["region"] == "us-east-2"
        assert result["vm_type"] == "g5.xlarge"
        assert result["spot"] is True
        assert result["credential_name"] == "anycloudshaws"
        assert b"CHECKPOINT_SAVE model=torchvision/resnet-18 epoch=50" in log
        assert b"CHECKPOINT_SAVE model=perforated-ai/resnet-18-perforated-cascor epoch=50" in log

print("Both complete GPU runs match the validated Spot-capable digest.")
