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
anycloud job ghcr.io/anycloud-sh/perforated-resnet18-flowers102@sha256:b674943c8ea20e190fd66f77d054eccbb25131f20ad47a16e1d1c0bb6d9a38d6 \
  --credentials lambda --gpu-type a10 --gpus all --disk-size 100
```

Use your saved Lambda credential name if it differs from `lambda`. The image's
default command trains both models for 50 epochs using the upstream data splits,
augmentations, optimizer, learning rate, scheduler, and seed.

## Input, output, and evidence

- **Input:** Torchvision downloads Flowers-102 (1,020 training images and 6,149
  test images). The runner downloads the pinned Perforated AI checkpoint and
  standard Torchvision ResNet-18 weights at runtime. Neither is in the image.
- **Output:** Job logs print loss and accuracy after each epoch, then a table of
  both models' accuracy and epoch-to-epoch changes. These logs are the result;
  the Job writes no durable output file and requires no output bucket.
- **Observed result:** after 50 epochs, standard ResNet-18 reached **82.50%**
  test accuracy and the Perforated checkpoint reached **86.18%** (+3.68
  percentage points). This is one seeded comparison on the upstream test split,
  not a general performance guarantee. See the [validation record](validation/lambda-a10.json)
  and complete [Job log](validation/lambda-a10.log) for every epoch. A
  [second Lambda run](validation/public-pull-repeat.md) from the public digest
  completed the same 50-epoch table.
- **Runtime and downloads:** the container ran for 52 minutes 55 seconds on
  Lambda A10; VM time including setup was about 58 minutes ($1.25). The Job
  downloaded the 345 MB Flowers archive, approximately 45 MB of standard
  ResNet-18 weights, and a 50 MB Perforated checkpoint. The default run has no
  checkpointing because it is a finite comparison without resumable model
  state in this artifact.

The candidate digest is the exact image used for the Lambda evidence. The
release tag `v0.1.0` points to that same manifest. GitHub Actions builds the
candidate from `main` and promotes a digest only after checking the committed
50-epoch validation record.

## Build and release

[Build candidate image](.github/workflows/build.yml) runs on source pushes to
`main` or by manual dispatch. It builds `linux/amd64` from the pinned
[`Dockerfile`](Dockerfile) and publishes
`ghcr.io/anycloud-sh/perforated-resnet18-flowers102:candidate-<commit>` using
the repository's `GITHUB_TOKEN` after image tests pass. The hosted build for the
original Lambda release published
`candidate-24c756da650de6f5105a378a10b6ed9e6f4003de`.

[Promote validated digest](.github/workflows/promote.yml) accepts the candidate
digest after `validation/check_spot_promotion.py` verifies complete Lambda and
AWS Spot runs. It adds `v0.2.0` to the tested manifest only after both pass.
The older [`check_promotion.py`](validation/check_promotion.py) records the
`v0.1.0` Lambda release gate; the [original promotion run](https://github.com/anycloud-sh/perforated-resnet18-flowers102/actions/runs/36921919359)
passed; anonymous registry requests for the digest and `v0.1.0` both resolve
to `sha256:b674943c8ea20e190fd66f77d054eccbb25131f20ad47a16e1d1c0bb6d9a38d6`.

## Change the run

Pass the upstream script's training options after `--`. For example, this
shorter run changes the epoch count and batch size, then prints the same table:

```bash
anycloud job ghcr.io/anycloud-sh/perforated-resnet18-flowers102@sha256:b674943c8ea20e190fd66f77d054eccbb25131f20ad47a16e1d1c0bb6d9a38d6 \
  --credentials lambda --gpu-type a10 --gpus all --disk-size 100 \
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
- Hardware: one Lambda NVIDIA A10; the default command requires CUDA and
  verifies a synchronized matrix multiplication before training.
- License: the upstream source and model card state Apache 2.0. See
  [`LICENSE`](LICENSE), [`LICENSE.upstream`](LICENSE.upstream), and [`NOTICE`](NOTICE).

The newer `perforatedai==3.2.9` cannot load this checkpoint with the official
example's current loader. The pinned `3.2.0` release completes the example;
the failed attempt and its error are recorded in the [reproduction log](validation/upstream-reproduction.md).
