# Upstream reproduction on Lambda

Source: PerforatedAI/PerforatedAI commit
`9d317e629428d73d92b88dc501f0dce3b1017c86`, unmodified
`examples/base_examples/resnet_pretrained/flowers_comparison.py` (SHA-256
`8941933f971aaaeed45d140312c2cc6949962fb8eb706fd76e8c4f79a3f4f32e`).
Runtime: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` on Lambda NVIDIA A10,
`torchvision==0.20.1`, `transformers==4.46.3`, `scipy==1.14.1`.

| Job ID | Package | Result |
| --- | --- | --- |
| `pytorch-1790879547153-a1aq` | `perforatedai==3.2.9` | Stopped before the script: the base image has no `curl`; source download changed to Python's standard library. |
| `pytorch-1790879798003-5sdq` | `perforatedai==3.2.9` | Baseline trained, but checkpoint loading failed with missing `*.main_module.*` and unexpected plain module state-dict keys. |
| `pytorch-1790880189094-u9mh` | `perforatedai==3.2.0` | One upstream epoch completed: baseline 2.20%, Perforated 3.25% on 6,149 test images. This is a compatibility check, not the published comparison. |
| `pytorch-1790880522571-7u3r` | `perforatedai==3.2.0` | Full 50-epoch upstream run completed on Lambda A10. Final test accuracy: standard ResNet-18 82.50%; Perforated checkpoint 86.18% (+3.68 percentage points). |

The public checkpoint was uploaded on April 19, 2026. Version 3.2.0 was
released April 17; its `from_hf_pretrained` loader converts the network before
loading this checkpoint. Version 3.2.9's loader changed that path, causing the
state-dict mismatch above. The artifact pins 3.2.0 and downloads the exact
checkpoint revision instead of following `main`.

The full upstream run's unedited console output is in `upstream-50.log`. Its
data downloads use Torchvision's official Flowers-102 source; weights and
dataset files are not redistributed here.

The Job printed `STARTED_AT=2026-10-01T18:52:19Z` and
`FINISHED_AT=2026-10-01T19:46:50Z` (54 minutes 31 seconds, including package,
dataset, and weight downloads). AnyCloud recorded `completed`, exit code 0,
and $1.26 of Lambda compute. The 50-row table is measured output for this run,
not a claim that the same result holds across seeds or datasets.
