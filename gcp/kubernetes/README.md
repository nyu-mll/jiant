# Kubernetes Configuration

This directory contains configuration files and scripts for running a Kubernetes cluster on Google Kubernetes Engine (GKE) that can run our training and evaluation code. This is a convenient way to get an auto-scaling batch scheduler.

For more on Kubernetes, see [this comic](https://cloud.google.com/kubernetes-engine/kubernetes-comic/) and the [GKE quick-start guide](https://cloud.google.com/kubernetes-engine/docs/quickstart).

All the instructions below assume Ubuntu Linux running on a Google Compute Engine (GCE) instance, but should also work on any Ubuntu machine with the Google Cloud SDK installed.

# Set-Up Instructions

First, install Docker community edition:
```sh
wget https://download.docker.com/linux/ubuntu/dists/xenial/pool/stable/amd64/docker-ce_18.06.1~ce~3-0~ubuntu_amd64.deb
sudo dpkg -i docker-ce_18.06.1~ce~3-0~ubuntu_amd64.deb
```
Be sure to match the version exactly because this is required for `nvidia-docker`

Second, install `nvidia-docker` to [enable GPU support](https://github.com/NVIDIA/nvidia-docker):
```sh
# Add the package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update

# Install nvidia-docker2 and reload the Docker daemon configuration
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd

# Test nvidia-smi with the latest official CUDA image
docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
```

Finally, install `kubectl` so you can connect to a Kubernetes cluster:
```sh
sudo snap install kubectl --classic
```

## Docker Image

Build the Docker image containing all the library dependencies and environment variables for `jiant`:

```sh
cd /path/to/jiant
# <project_id> is your Google Cloud project (e.g. jsalt-sentence-rep)
sudo docker build -t gcr.io/<project_id>/jiant-sandbox:v1 .
```

Authenticate to Google Cloud so we can push containers to Google Container Registry (GCR), where they are accessible to Kubernetes nodes:
```sh
gcloud auth configure-docker
```

Now push the image that we built locally:
```sh
sudo gcloud docker -- push gcr.io/<project_id>/jiant-sandbox:v1
```
You can keep track of uploaded images by going to Cloud Console -> Container Engine.

## Kubernetes Cluster

Go to [Google Cloud Console](cloud.google.com/console) in your browser, go to **Kubernetes Engine**, and create a cluster. We recommend giving each node at least 13 GB of RAM, or 26 GB for GPU workers. If you want to run on GPUs, we recommend creating a pool with either P100s or K80s, and enabling autoscaling.

Authenticate to the cluster:
```sh
gcloud container clusters get-credentials <cluster_name> --zone <cluster_zone>
```

Now you can run a simple test job:
```sh
kubectl run hello-world --image gcr.io/<project_id>/jiant-sandbox:v1 -- \
    echo "Hello World!"
```

You should see a workload named `hello-world` in the Kubernetes Engine -> Workloads page, and if you click through to container logs you should see the output of your command.

### GPUs on Kubernetes

To enable GPU support, you need to load a `DaemonSet` that installs GPU drivers on each node. Run:
```sh
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/stable/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

This creates a persistent workload, which you can see as `nvidia-driver-installer` under Kubernetes Engine -> Workloads if you enable showing system objects. Wait a few minutes for it to finish before scheduling anything.

### NFS on Kubernetes

To allow Kubernetes jobs to access our NFS server, we need to create two resources: a `PersistentVolume` that defines the storage volume, and a `PersistentVolumeClaim` that allows jobs to connect to it.

Use the YAML files in this directory to configure these. You'll likely need to edit the files and change the server IP to match your NFS server.
```sh
kubectl create -f gcp/kubernetes/nfs_pv.yaml
kubectl create -f gcp/kubernetes/nfs_pvc.yaml
```

If this is successful, you should see the volume claim in cloud console under Kubernetes Engine -> Storage.


## Running Jobs

We can schedule basic jobs on the cluster using `kubectl run` as above, but in order to use GPUs and access NFS we need to use a full-fledged YAML config.

The `run_batch.sh` script handles creating an appropriate config on the fly; see the documentation in that file for more details. Basic usage is:

```sh
export JIANT_PATH="/nfs/jsalt/home/$USER/jiant"
./run_batch.sh <job_name> \
   "python $JIANT_PATH/main.py --config_file $JIANT_PATH/config/demo.conf"
```

You should see your job as `<job_name>` in Kubernetes Engine -> Workloads, and can monitor status, resource usage, and logs from that page.
