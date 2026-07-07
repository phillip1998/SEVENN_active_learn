# DFT Calculation API

FastAPI server for receiving Gaussian `.gjf` files, submitting an SGE/Grid Engine `qsub` calculation from a helper Docker container, and returning finished `.log` files.

## Docker run on a cluster

```bash
docker build -t my-api-image .
```

Build the qsub helper image:

```bash
cd qsub-image
docker build -t nfs-shared-storage-image .
cd ..
```

Run the API container with Docker socket access and the cluster-visible workspace. `DFT_HOST_JOBS_DIR` must be the Docker host path for the same storage mounted at `DFT_JOBS_DIR` inside the API container. `DFT_SCHEDULER_JOBS_DIR` must be the path compute nodes can `chdir` into.

```bash
docker run -d -p 8000:8000 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /home/ufslab/API_workspace:/home/ufslab/API_workspace \
  -e DFT_JOBS_DIR=/home/ufslab/API_workspace \
  -e DFT_HOST_JOBS_DIR=/home/ufslab/API_workspace \
  -e DFT_SCHEDULER_JOBS_DIR=/home/ufslab/API_workspace \
  -e DFT_GAUSSIAN_BIN=/opt/gaussian/g16/g16 \
  -e DFT_QSUB_IMAGE=nfs-shared-storage-image \
  -e DFT_QSUB_USER="$(id -u):$(id -g)" \
  my-api-image
```

For each API job, the server writes a job-specific `run.sh` into the job directory and starts a short-lived helper container equivalent to:

```bash
docker run --rm --network host \
  -v /home/ufslab/API_workspace/{job_id}:/home/ufslab/API_workspace/{job_id} \
  --user "$(id -u):$(id -g)" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -e SGE_QMASTER=UFSLAB \
  -e SGE_QMASTER_PORT=6444 \
  -e SGE_EXECD_PORT=6445 \
  nfs-shared-storage-image \
  qsub /home/ufslab/API_workspace/{job_id}/run.sh
```

The helper container only submits to the scheduler. The actual Gaussian calculation runs later through Grid Engine, and the API reports `completed` once every expected `.log` file exists in the job directory.

Generated Gaussian `run.sh` files use this fixed header. The node part of `#$ -q all.q@...` comes from the API request.

```bash
#$ -pe mpi_36 36
#$ -N Gau_jyp
#$ -S /bin/bash
#$ -q all.q@node03
#$ -V
#$ -wd /home/ufslab/API_workspace/{job_id}

echo "Got $NSLOTS slots."
cat $TMPDIR/machines

export gr=/opt/gaussian
export g16root=/opt/gaussian
export GAUSS_EXEDIR=/opt/gaussian/g16
export PATH=/opt/gaussian/g16:$PATH
export GAUSS_SCRDIR=/scratch
export JOB=test0706
```

The actual calculation commands are supplied in the `POST /jobs` request, for example `g16 temp0605.gjf >>temp0605.log`. The API writes `/opt/gaussian/g16/g16 ...` into `run.sh` so the job does not depend on the submitted environment's `PATH`.

## Configuration

```bash
DFT_EXECUTOR=docker_qsub
DFT_JOBS_DIR=/home/ufslab/API_workspace
DFT_HOST_JOBS_DIR=/home/ufslab/API_workspace
DFT_SCHEDULER_JOBS_DIR=/home/ufslab/API_workspace
DFT_GAUSSIAN_BIN=/opt/gaussian/g16/g16
DFT_JOB_DIR_MODE=777
DFT_QSUB_IMAGE=nfs-shared-storage-image
DFT_QSUB_NETWORK_MODE=host
DFT_QSUB_WORKDIR=/work
DFT_QSUB_COMMAND_TEMPLATE="qsub {submit_script}"
DFT_QSUB_EXTRA_VOLUMES='{"/etc/passwd":{"bind":"/etc/passwd","mode":"ro"},"/etc/group":{"bind":"/etc/group","mode":"ro"}}'
DFT_QSUB_ENVIRONMENT='{"SGE_QMASTER":"UFSLAB","SGE_QMASTER_PORT":"6444","SGE_EXECD_PORT":"6445"}'
DFT_QSUB_USER="$(id -u):$(id -g)"
```

`nfs-shared-storage-image` should be an SGE submit image built on the cluster submit host with `qsub-image/build_from_host.sh`. It needs `qsub` available in `PATH`, network access to the scheduler, and permission to submit jobs as the desired cluster user. If your cluster requires additional live SGE config mounts, add them with `DFT_QSUB_EXTRA_VOLUMES`, for example:

```bash
DFT_QSUB_EXTRA_VOLUMES='{
  "/opt/sge": {"bind": "/opt/sge", "mode": "ro"},
  "/etc/gridengine": {"bind": "/etc/gridengine", "mode": "ro"}
}'
```

The API defaults now match the successful manual command: helper containers run with `DFT_QSUB_USER`, mount host `/etc/passwd` and `/etc/group`, and pass `SGE_QMASTER=UFSLAB`, `SGE_QMASTER_PORT=6444`, and `SGE_EXECD_PORT=6445`.

The helper container mounts each job at the same absolute path seen by compute nodes, for example `/home/ufslab/API_workspace/{job_id}`. This avoids `Eqw` failures caused by submitting a script from container-only paths such as `/work`. If `qstat -j` says `can't chdir`, set `DFT_SCHEDULER_JOBS_DIR` to the exact path visible on compute nodes.

If qsub needs a different username source for the submitted UID, adjust the passwd/group mounts:

```bash
DFT_QSUB_EXTRA_VOLUMES='{
  "/etc/passwd": {"bind": "/etc/passwd", "mode": "ro"},
  "/etc/group": {"bind": "/etc/group", "mode": "ro"}
}'
```

If `/var/lib/gridengine` is not available on the Docker host, provide the qmaster hostname directly:

```bash
DFT_QSUB_EXTRA_VOLUMES='{}'
DFT_QSUB_ENVIRONMENT='{"SGE_QMASTER": "your-qmaster-hostname", "SGE_QMASTER_PORT": "6444", "SGE_EXECD_PORT": "6445"}'
```

For local development without qsub:

```bash
DFT_EXECUTOR=local uvicorn app:app --host 0.0.0.0 --port 8000
```

## API

### Submit calculations

```bash
curl -X POST http://localhost:8000/jobs \
  -F "node=node03" \
  -F "job_name=Gau_jyp" \
  -F "gaussian_job=test0706" \
  -F "commands_text=g16 example1.gjf >>example1.log
g16 example2.gjf >>example2.log" \
  -F "files=@example1.gjf" \
  -F "files=@example2.gjf"
```

You can also repeat the `commands` form field:

```bash
curl -X POST http://localhost:8000/jobs \
  -F "node=node05" \
  -F "commands=g16 example1.gjf >>example1.log" \
  -F "commands=g16 example2.gjf >>example2.log" \
  -F "files=@example1.gjf" \
  -F "files=@example2.gjf"
```

If `commands` and `commands_text` are omitted, the API generates one command per uploaded file:

```bash
g16 example1.gjf >>example1.log
g16 example2.gjf >>example2.log
```

For safety, commands must match the Gaussian form `g16 input.gjf >>output.log`.

Response:

```json
{
  "job_id": "uuid",
  "status": "queued",
  "node": "node03",
  "gaussian_job": "test0706",
  "commands": ["g16 example1.gjf >>example1.log"],
  "input_files": [],
  "log_files": []
}
```

### Check status

```bash
curl http://localhost:8000/jobs/{job_id}
```

or:

```bash
curl http://localhost:8000/jobs/{job_id}/status
```

Statuses are `queued`, `submitting`, `submitted`, `running`, `completed`, or `failed`.

### Download all result logs

```bash
curl -o logs.zip http://localhost:8000/jobs/{job_id}/results
```

This endpoint returns a zip containing the completed `.log` files. It returns HTTP `409` until all expected `.log` files are available.

### Download one log

```bash
curl -O http://localhost:8000/jobs/{job_id}/logs/example1.log
```
