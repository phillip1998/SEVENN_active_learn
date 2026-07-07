FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py run.sh ./

ENV DFT_JOBS_DIR=/home/ufslab/API_workspace
ENV DFT_HOST_JOBS_DIR=/home/ufslab/API_workspace
ENV DFT_SCHEDULER_JOBS_DIR=/home/ufslab/API_workspace
ENV DFT_GAUSSIAN_BIN=/opt/gaussian/g16/g16
ENV DFT_EXECUTOR=docker_qsub
ENV DFT_QSUB_IMAGE=nfs-shared-storage-image
ENV DFT_QSUB_WORKDIR=/work
ENV DFT_QSUB_EXTRA_VOLUMES="{\"/etc/passwd\":{\"bind\":\"/etc/passwd\",\"mode\":\"ro\"},\"/etc/group\":{\"bind\":\"/etc/group\",\"mode\":\"ro\"}}"
ENV DFT_QSUB_ENVIRONMENT="{\"SGE_QMASTER\":\"UFSLAB\",\"SGE_QMASTER_PORT\":\"6444\",\"SGE_EXECD_PORT\":\"6445\"}"

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
