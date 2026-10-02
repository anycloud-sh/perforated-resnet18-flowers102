# Perforated AI ResNet-18 on Flowers-102

Run Perforated AI's Flowers-102 comparison on AWS Spot. Fine-tune
`resnet-18-perforated-cascor` alongside standard ResNet-18 and see how
their test accuracy changes over 50 epochs.

This AnyCloud-powered Job uses Perforated AI's
[official example](https://github.com/PerforatedAI/PerforatedAI/tree/9d317e629428d73d92b88dc501f0dce3b1017c86/examples/base_examples/resnet_pretrained).

## Run

```bash
anycloud job ghcr.io/anycloud-sh/perforated-resnet18-flowers102@sha256:732f4747857c0fe8744c6058159ddc642e14cbb974504506a9aa54932bcfea7f \
  --credentials awstest --region us-east-2 --vm-type g5.xlarge \
  --spot --gpus all --disk-size 100
```

Replace `awstest` with your saved AWS credential name. The account needs
G-family Spot quota in `us-east-2`.

## What you get

The Job downloads the dataset and weights (~440 MB), trains on 1,020 images,
and evaluates on 6,149 test images. Loss and accuracy print after each epoch,
followed by a 50-row comparison in Job logs. No dashboard needed.

Our validated run finished with:

| Model                   | Epoch-50 test accuracy |
| ----------------------- | ---------------------: |
| Standard ResNet-18      |                 82.50% |
| Perforated AI ResNet-18 |                 86.18% |

That's **+3.68 percentage points** for the Perforated model in this seeded run.

On one NVIDIA A10G (`g5.xlarge`), recorded VM time was **66 minutes** and
estimated compute cost was **$0.57**. The same image also passed a full run on
Lambda A10.

[Every epoch](validation/aws-spot.log) · [AWS run](validation/aws-spot.json) ·
[Lambda run](validation/lambda-spot-image.json)

## Spot recovery

The Job checkpoints after each completed epoch to `/mnt/checkpoint`.
After an interruption, it resumes from the latest synced checkpoint;
recent epochs may repeat.
The automatic recovery bucket is deleted when the Job finishes. Accuracy
stays in Job logs; a trained model is not retained by default.

## Try a shorter run

For 10 epochs and a smaller batch, append this to the command above:

`-- python runner.py --epochs 10 --batch-size 32`

## Source and build

- Source, model revisions, and attribution: [NOTICE](NOTICE).
- Runtime: [Dockerfile](Dockerfile) and [pinned dependencies](requirements.txt).
  The runner replaces the checkpoint's random classifier for transfer learning.
- [Hosted CI](.github/workflows/build.yml) builds the image and tests recovery
  with both real models before publishing.
  [Promotion](.github/workflows/promote.yml) tags the validated digest as `v0.2.0`;
  [release checks](validation/spot-release.json) also confirm `v0.1.0` is retained.
  Its [original promotion gate](validation/check_promotion.py) is kept as release history.
- Upstream reproduction and compatibility notes: [validation](validation/upstream-reproduction.md).
- Apache 2.0: [LICENSE](LICENSE) and [upstream license](LICENSE.upstream).
