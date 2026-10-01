# Public-pull Lambda repeat

On October 1, 2026, the GHCR package was public and anonymously accessible.
An unauthenticated GHCR token request and manifest `HEAD` returned HTTP 200
for the digest and release tag `v0.1.0`; both resolved to
`sha256:b674943c8ea20e190fd66f77d054eccbb25131f20ad47a16e1d1c0bb6d9a38d6`.
The public package page showed the linked source repository.

Job `perforate-1790886617870-mlg7` submitted that digest to Lambda A10 with
the README's default command. The Job record had no attached GitHub token and
recorded a Docker image pull before starting the container at 20:34:08 UTC.
The log showed a synchronized CUDA matrix multiplication on `NVIDIA A10`
(`allocated_bytes=8716288`, `matmul_checksum=100.711426`) and verification of
model revision `8a59acfc5285e72257e8d322da51bf486c3ce060` with SHA-256
`114d3ed72896110ff229bd5a7c4ae8e7d048eb185ba1cdeb75cd76557d8674bc`.

The Job completed with exit code 0 at 21:27:03 UTC after all 50 epochs of
both models over the official 1,020 training and 6,149 test images. Its final
50-row comparison matched the
[committed accuracy arrays](lambda-a10.json) row for row, ending at 82.50%
for standard ResNet-18 and 86.18% for the Perforated checkpoint. VM time was
about 57 minutes, with an estimated $1.23 of Lambda compute.

The repeat's final table and completed Job status were inspected live. The
container was removed before the complete second log could be exported;
`anycloud logs` then reported no live container. The
[first full Lambda log](lambda-a10.log) remains the retained epoch-by-epoch
evidence for the same immutable digest, alongside the
[unmodified upstream run](upstream-50.log).
