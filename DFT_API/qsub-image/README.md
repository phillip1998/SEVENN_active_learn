# SGE qsub helper image

This builds the helper image used by the API server as `DFT_QSUB_IMAGE=nfs-shared-storage-image`.

It is only responsible for submitting jobs:

```bash
qsub /work/run.sh
```

The Gaussian calculation itself is executed later by Grid Engine on the cluster node.

## Build from the cluster submit host

```bash
cd DFT_API/qsub-image
bash build_from_host.sh nfs-shared-storage-image
```

This copies the exact `qsub` binary from the submit host, plus its shared libraries and common Grid Engine config paths, into the image. This avoids GDI protocol mismatches such as:

```text
client uses newer GDI version ... while qmaster uses older version 8.1.8
```

The copied shared libraries are placed under `/usr/local/sge-libs` inside the image to avoid conflicts with base-image paths such as `/lib64`. Core glibc libraries such as `libc.so.6` are intentionally not copied, because overriding the base image glibc can break `/bin/sh` during build or runtime.

## Smoke test

Run this on the cluster login/submit host:

```bash
docker run --rm --network host \
  --user "$(id -u):$(id -g)" \
  -v /var/lib/gridengine:/var/lib/gridengine:ro \
  nfs-shared-storage-image \
  "which qsub && qsub --help"
```

If you see `unable to read qmaster name`, qsub cannot read:

```text
/var/lib/gridengine/default/common/act_qmaster
```

Mount `/var/lib/gridengine` from the host, or provide the qmaster hostname directly:

```bash
docker run --rm --network host \
  --user "$(id -u):$(id -g)" \
  -e SGE_QMASTER=your-qmaster-hostname \
  -e SGE_QMASTER_PORT=6444 \
  -e QSUB_HELPER_DEBUG=1 \
  nfs-shared-storage-image \
  "cat /var/lib/gridengine/default/common/act_qmaster && which qsub && qsub --help"
```

If the debug lines do not appear, Docker is still running an old image. Rebuild with `--no-cache` or remove the old image first.

If qsub says it cannot get `SGE_QMASTER_PORT` or service `sge_qmaster`, set `SGE_QMASTER_PORT`. The common SGE default is `6444`; `SGE_EXECD_PORT` commonly defaults to `6445`.

If qsub says `common/bootstrap` is missing, the build did not copy the real SGE root. Find it on the submit host:

```bash
find / -path '*/common/bootstrap' 2>/dev/null
```

Then rebuild with that root:

```bash
SGE_ROOT=/opt/sge bash build_from_host.sh nfs-shared-storage-image
```

Do not run the build script with plain `sudo` unless you preserve the qsub environment. `sudo` often resets `PATH`, which makes `qsub` disappear. If you must use sudo, run:

```bash
sudo env "PATH=$PATH" "SGE_ROOT=$SGE_ROOT" "SGE_CELL=${SGE_CELL:-default}" \
  bash build_from_host.sh nfs-shared-storage-image
```

The script copies only the qsub binaries, shared libraries, and Grid Engine `common` config where possible. Permission-denied files such as SGE setuid helper utilities are skipped unless they are directly needed.

If Docker prints `exec /usr/local/bin/qsub-entrypoint: no such file or directory`, rebuild after pulling this Dockerfile change. That error is usually caused by Windows CRLF line endings in `entrypoint.sh`; the Dockerfile now normalizes them during build.

If your cluster needs extra Grid Engine config mounted from the host, pass it through the API container with `DFT_QSUB_EXTRA_VOLUMES`.

Common examples:

```bash
DFT_QSUB_EXTRA_VOLUMES='{
  "/opt/sge": {"bind": "/opt/sge", "mode": "ro"},
  "/etc/gridengine": {"bind": "/etc/gridengine", "mode": "ro"},
  "/var/lib/gridengine": {"bind": "/var/lib/gridengine", "mode": "ro"}
}'
```

## Cluster user

Many clusters reject submissions from `root`. If that happens, run helper containers with your cluster UID/GID:

```bash
docker run --rm --network host \
  --user "$(id -u):$(id -g)" \
  -e SGE_QMASTER=UFSLAB \
  -e SGE_QMASTER_PORT=6444 \
  -e SGE_EXECD_PORT=6445 \
  -v "$PWD":/work \
  nfs-shared-storage-image \
  "which qsub && qsub /work/run.sh"
```

If qsub cannot resolve your username, also mount host passwd/group:

```bash
docker run --rm --network host \
  --user "$(id -u):$(id -g)" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -e SGE_QMASTER=UFSLAB \
  -e SGE_QMASTER_PORT=6444 \
  -e SGE_EXECD_PORT=6445 \
  -v "$PWD":/work \
  nfs-shared-storage-image \
  "which qsub && qsub /work/run.sh"
```

For the API server, set:

```bash
DFT_QSUB_USER="$(id -u):$(id -g)"
```

Alternatively, rebuild the helper image with the same UID/GID as your cluster submit user:

```dockerfile
ARG SUBMIT_UID=1000
ARG SUBMIT_GID=1000
RUN groupadd -g ${SUBMIT_GID} submitter \
    && useradd -m -u ${SUBMIT_UID} -g ${SUBMIT_GID} submitter
USER submitter
```

Then build:

```bash
docker build \
  --build-arg SUBMIT_UID="$(id -u)" \
  --build-arg SUBMIT_GID="$(id -g)" \
  -t nfs-shared-storage-image .
```
