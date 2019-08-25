# Kubernetes Configuration

This directory contains configuration files and scripts for running a Kubernetes cluster on Google Kubernetes Engine (GKE) that can run our training and evaluation code. This is a convenient way to get an auto-scaling batch scheduler.

For more on Kubernetes, see [this comic](https://cloud.google.com/kubernetes-engine/kubernetes-comic/) and the [GKE quick-start guide](https://cloud.google.com/kubernetes-engine/docs/quickstart).

All the instructions below assume Ubuntu Linux running on a Google Compute Engine (GCE) instance, but should also work on any Ubuntu machine with the Google Cloud SDK installed.

# Set-Up Instructions

## jsonnet
Several of the Kubernetes scripts use [jsonnet](https://jsonnet.org/) for configuration. To install the jsonnet command-line utility, you need to build it from source and add it to your system path:
```
git clone https://github.com/google/jsonnet.git jsonnet
cd jsonnet
make
sudo cp jsonnet /usr/bin
```

## Configuring Cluster Resources

[`templates/jiant_env.libsonnet`](templates/jiant_env.jsonnet) contains several variables (such as the name of the jiant image and the location of the NFS server) that are specific to the cluster environment. Edit that file to reflect your environment before running jobs or further set-up.

## On Your Workstation

_If you're using the "Deep Learning Image: PyTorch 1.1.0" VM image on Google Cloud Platform, you should already have Docker, nvidia-docker, and kubectl installed, and you can skip the rest of this section._

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
sudo docker build -t gcr.io/<project_id>/jiant:v1 .
```

Authenticate to Google Cloud so we can push containers to Google Container Registry (GCR), where they are accessible to Kubernetes nodes:
```sh
gcloud auth configure-docker
```

Now push the image that we built locally:
```sh
sudo gcloud docker -- push gcr.io/<project_id>/jiant:v1
```
You can keep track of uploaded images by going to Cloud Console -> Container Engine.

_**Note:** you should update [`templates/jiant_env.libsonnet`](templates/jiant_env.libsonnet) with the name of this container (`gcr.io/...`), so that the job templates will use it correctly._

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

### One-time cluster set-up

After you first create the Kubernetes cluster, run:
```
./init_cluster.sh
```

This does two things:

- It adds a DaemonSet that installs GPU drivers on each node.
- It sets up a PersistentVolume that defines the NFS volume in a way Kubernetes
  can understand, and sets up a PersistentVolumeClaim that allows jobs to access it.

## Running Jobs

We can schedule basic jobs on the cluster using `kubectl run` as above, but in order to use GPUs and access NFS we need to use a full-fledged YAML config.

The `run_batch.sh` script handles creating an appropriate config on the fly; see the documentation in that file for more details. Basic usage is:

```sh
export JIANT_PATH="/nfs/jiant/home/$USER/jiant"
./run_batch.sh <job_name> \
   "python $JIANT_PATH/main.py --config_file $JIANT_PATH/jiant/config/demo.conf --overrides 'run_name = kubernetes-demo, target_tasks = \"wnli,commitbank\"'"
```

You should see your job as `<job_name>` in Kubernetes Engine -> Workloads, and can monitor status, resource usage, and logs from that page.
