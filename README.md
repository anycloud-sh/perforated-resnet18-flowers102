# Perforated AI ResNet-18 on Flowers-102

Run a reproducible Flowers-102 transfer-learning comparison of Perforated AI's
`resnet-18-perforated-cascor` checkpoint and standard ResNet-18 on a cloud GPU.

This AnyCloud-powered Job follows Perforated AI's
[official comparison](https://github.com/PerforatedAI/PerforatedAI/tree/9d317e629428d73d92b88dc501f0dce3b1017c86/examples/base_examples/resnet_pretrained).
The creator owns the model and training method. This repository contains its
original comparison script plus a small runner that pins and verifies the
released checkpoint and proves CUDA execution. It does not reproduce the
proprietary checkpoint training process.

## Run

```bash
anycloud job ghcr.io/anycloud-sh/perforated-resnet18-flowers102@sha256:732f4747857c0fe8744c6058159ddc642e14cbb974504506a9aa54932bcfea7f \
  --credentials awstest --region us-east-2 --vm-type g5.xlarge \
  --spot --gpus all --disk-size 100
```

Use your saved AWS credential name if it differs from `awstest`. The account
needs G-family Spot quota in `us-east-2`; available quota does not guarantee
Spot capacity. The image's default command trains both models for 50 epochs
using the upstream data splits, augmentations, optimizer, learning rate,
scheduler, and seed.

## Input, output, and evidence

- **Input:** Torchvision downloads Flowers-102 (1,020 training images and 6,149
  test images). The runner downloads the pinned Perforated AI checkpoint and
  standard Torchvision ResNet-18 weights at runtime. Neither is in the image.
- **Output:** Job logs print loss and accuracy after each epoch, then a table of
  both models' accuracy and epoch-to-epoch changes. These logs are the result;
  no dashboard endpoint or durable output file is required. Spot checkpoints
  are temporary recovery state, not a retained final model.
- **Downloads:** the Job downloaded the 345 MB Flowers archive, approximately
  45 MB of standard ResNet-18 weights, and a 50 MB Perforated checkpoint.

Both complete validation runs of `v0.2.0` produced these final test accuracies:

| Model                   | Epoch-50 accuracy |
| ----------------------- | ----------------: |
| Standard ResNet-18      |            82.50% |
| Perforated AI ResNet-18 |            86.18% |

The observed difference was **+3.68 percentage points** for the Perforated
checkpoint. This seeded comparison on the upstream test split does not
guarantee the same ordering in another run. The complete, finite 50-row tables
are in the [AWS Spot record](validation/aws-spot.json) and
[raw training log](validation/aws-spot.log), and the
[Lambda record](validation/lambda-spot-image.json) and
[raw log](validation/lambda-spot-image.log).

| Validation                         | GPU         | Recorded VM time      | Recorded compute cost |
| ---------------------------------- | ----------- | --------------------- | --------------------: |
| AWS Spot, `g5.xlarge`, `us-east-2` | NVIDIA A10G | 66 minutes 26 seconds |       $0.57 estimated |
| Lambda                             | NVIDIA A10  | 56 minutes 55 seconds |                 $1.22 |

The AWS container was observed running for about one hour. Job wall time was
72 minutes 30 seconds, including two capacity failures before the successful
launch in `us-east-2b`. The [Job status log](validation/aws-spot-status.log)
records exit code 0 and a successful final checkpoint sync. Cleanup completed
without an error, and the automatic checkpoint bucket was removed.

## Spot recovery

AnyCloud automatically mounts its Spot recovery bucket at `/mnt/checkpoint`.
The runner detects that mount and saves model, optimizer, scheduler, accuracy
history, and random-number state after every completed epoch. On restart it
restores the latest compatible checkpoint and continues at the correct epoch,
including a restart between models. Corrupt or incompatible state fails the
Job. Periodic bucket sync can require repeating recent epochs.

The [hosted recovery tests](https://github.com/anycloud-sh/perforated-resnet18-flowers102/actions/runs/37040879663)
restore both real model formats after interruption and cover the transition
between models. The AWS validation run saved 50 checkpoints per model and
confirmed remote sync for both models; it had no Spot interruption. The bucket
is for recovery within one Job and is removed after the Job finishes. Final
accuracy remains in Job logs and the committed validation evidence.

## Build and release

[Build candidate image](.github/workflows/build.yml) runs on source pushes to
`main` or by manual dispatch. It builds `linux/amd64` from the pinned
[`Dockerfile`](Dockerfile) and publishes
`ghcr.io/anycloud-sh/perforated-resnet18-flowers102:candidate-<commit>` using
the repository's `GITHUB_TOKEN` after image tests pass. The
[validated hosted build](https://github.com/anycloud-sh/perforated-resnet18-flowers102/actions/runs/37040879663)
published `candidate-4d5128838674fc2f1af6f9667e20fd50da7c7bf9` from
[source commit `4d51288`](https://github.com/anycloud-sh/perforated-resnet18-flowers102/commit/4d5128838674fc2f1af6f9667e20fd50da7c7bf9).

[Promote validated digest](.github/workflows/promote.yml) accepts the candidate
digest after `validation/check_spot_promotion.py` verifies complete Lambda and
AWS Spot runs. The [v0.2.0 promotion run](https://github.com/anycloud-sh/perforated-resnet18-flowers102/actions/runs/37062534888)
tags the existing tested manifest without rebuilding it. The
[anonymous registry checks](validation/spot-release.json) confirm that
`v0.2.0` resolves to the digest in the Run command above and `v0.1.0` is retained.

The original `v0.1.0` Lambda release is retained at
`sha256:b674943c8ea20e190fd66f77d054eccbb25131f20ad47a16e1d1c0bb6d9a38d6`,
with its [validation record](validation/lambda-a10.json),
[raw log](validation/lambda-a10.log), and
[public pull repeat](validation/public-pull-repeat.md).
The older [`check_promotion.py`](validation/check_promotion.py) records its
historical release gate and [promotion run](https://github.com/anycloud-sh/perforated-resnet18-flowers102/actions/runs/36921919359).

## Run on Lambda

The same validated image also runs on a Lambda A10:

```bash
anycloud job ghcr.io/anycloud-sh/perforated-resnet18-flowers102@sha256:732f4747857c0fe8744c6058159ddc642e14cbb974504506a9aa54932bcfea7f \
  --credentials lambda --gpu-type a10 --gpus all --disk-size 100
```

Use your saved Lambda credential name if it differs from `lambda`. Without a
checkpoint mount, the runner uses the unchanged upstream training path.

## Change the run

Pass the upstream script's training options after `--`. For example, this
shorter run changes the epoch count and batch size, then prints the same table:

```bash
anycloud job ghcr.io/anycloud-sh/perforated-resnet18-flowers102@sha256:732f4747857c0fe8744c6058159ddc642e14cbb974504506a9aa54932bcfea7f \
  --credentials awstest --region us-east-2 --vm-type g5.xlarge \
  --spot --gpus all --disk-size 100 \
  -- python runner.py --epochs 10 --batch-size 32
```

For another dataset or model, use the complete source here as an integration
example and adapt the official comparison script. The public checkpoint's final
classifier has random weights, so replace it for transfer learning; it is not
an ImageNet inference model.

## Provenance and scope

- Upstream source: [`9d317e629428d73d92b88dc501f0dce3b1017c86`](https://github.com/PerforatedAI/PerforatedAI/tree/9d317e629428d73d92b88dc501f0dce3b1017c86),
  copied verbatim at [`upstream/flowers_comparison.py`](upstream/flowers_comparison.py).
- Model: [`8a59acfc5285e72257e8d322da51bf486c3ce060`](https://huggingface.co/perforated-ai/resnet-18-perforated-cascor/tree/8a59acfc5285e72257e8d322da51bf486c3ce060),
  SHA-256 `114d3ed72896110ff229bd5a7c4ae8e7d048eb185ba1cdeb75cd76557d8674bc`.
- Runtime: PyTorch 2.5.1 CUDA 12.4 image, `perforatedai==3.2.0`, and pinned
  direct Python dependencies in [`requirements.txt`](requirements.txt).
- Hardware: one AWS Spot NVIDIA A10G on `g5.xlarge`, plus Lambda NVIDIA A10
  validation; the default command verifies a synchronized CUDA matrix
  multiplication before training.
- License: the upstream source and model card state Apache 2.0. See
  [`LICENSE`](LICENSE), [`LICENSE.upstream`](LICENSE.upstream), and [`NOTICE`](NOTICE).

The newer `perforatedai==3.2.9` cannot load this checkpoint with the official
example's current loader. The pinned `3.2.0` release completes the example;
the failed attempt and its error are recorded in the [reproduction log](validation/upstream-reproduction.md).
