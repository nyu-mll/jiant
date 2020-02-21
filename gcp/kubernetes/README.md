# `jiant` on GKE

This directory contains configuration files and scripts for running a Kubernetes cluster on Google Kubernetes Engine (GKE) that can run our training and evaluation code. This is a convenient way to get an auto-scaling batch scheduler.

You may be referring to this guide for several reasons:

* You should probably check the prerequisites regardless: [Prerequisites](#Prerequisites)
* If you are looking to start *from scratch*, including building a Docker image: [Creating a Docker Image](#creating-a-docker-image)
* If you are looking to start a GKE cluster + NFS from an existing image: [Creating a GKE Cluster + NFS](#creating-a-gke-cluster--nfs)
* If you are looking to access an existing GKE cluster for the first time: [Using an existing GKE cluster](#using-an-existing-gke-cluster)

For more on Kubernetes, see [this comic](https://cloud.google.com/kubernetes-engine/kubernetes-comic/) and the [GKE quick-start guide](https://cloud.google.com/kubernetes-engine/docs/quickstart).

======

# Prerequisites

### Workstation

All setup instructions for the work station assume you are using a Google Compute Engine (GCE) instance (e.g. "Deep Learning with Linux" for the OS), but should also work on any Ubuntu machine with the Google Cloud SDK installed. We suggest you search for one with Docker, nvidia-docker, kubectl, and Pytorch installed, such as "Deep Learning Image: PyTorch 1.2.0" image. 

We recommend that you create a lightweight GCP instance in the same region as your desired cluster. Select "Allow full access to all Cloud APIs" if possible.

### jsonnet

Once you are on your workstation, several of the Kubernetes scripts use [jsonnet](https://jsonnet.org/) for configuration. To install the jsonnet command-line utility, you need to build it from source and add it to your system path:
```
git clone https://github.com/google/jsonnet.git jsonnet
cd jsonnet
make
sudo cp jsonnet /usr/bin
```

======

# Creating a Docker Image

### Library setup

_If you're using the "Deep Learning Image: PyTorch 1.2.0" VM image on Google Cloud Platform, you should already have Docker, nvidia-docker, and kubectl installed, and you can skip the rest of this section._

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

### Docker Image

Build the Docker image containing all the library dependencies and environment variables for `jiant`:

```sh
cd /path/to/jiant
# <project_id> is your Google Cloud project (e.g. jsalt-sentence-rep)
sudo docker build -t gcr.io/<project_id>/jiant:v1 .
```

You may have to use `sudo` for the following. Authenticate to Google Cloud so we can push containers to Google Container Registry (GCR), where they are accessible to Kubernetes nodes:
```sh
gcloud auth configure-docker
```

Now push the image that we built locally:
```sh
gcloud docker -- push gcr.io/<project_id>/jiant:v1
```
You can keep track of uploaded images by going to Cloud Console -> Container Engine.

_**Note:** you should update [`templates/jiant_env.libsonnet`](templates/jiant_env.libsonnet) with the name of this container (`gcr.io/...`), so that the job templates will use it correctly._

======

# Creating a GKE Cluster + NFS

Before starting this section, you should have a URL (`gcr.io/XXXX`) to a Docker image.

### GKE Cluster

Go to [Google Cloud Console](https://cloud.google.com/console) in your browser, go to **Kubernetes Engine**, and create a cluster. We recommend giving each node at least 13 GB of RAM, or 26 GB for GPU workers. If you want to run on GPUs, we recommend creating a pool with either P100s or K80s, and enabling autoscaling.

Authenticate to the cluster:
```sh
gcloud container clusters get-credentials <cluster_name> --zone <cluster_zone>
```

(If you run into an "Insufficient Scope" error, run `gcloud init` and authenticate with the relevant account.)

Now you can run a simple test job:
```sh
kubectl run hello-world --generator=job/v1 --image gcr.io/XXXX --restart=Never -- echo "Hello World!"
```

You should see a workload named `hello-world` in the Kubernetes Engine -> Workloads page, and if you click through to container logs you should see the output of your command.

If you want to scale back your cluster, follow these [instructions](https://stackoverflow.com/questions/46838221/how-to-stop-a-kubernetes-cluster). 

### NFS Setup

Go to [Google Cloud Console](cloud.google.com/console) in your browser, go to **Filestore**, and create a Filestore instance in the same region as your cluster. Take note of the `IP Address`, `Fileshare name`, `Instance ID` and `capacity`.

Modify [`config/auto.nfs`](config/auto.nfs):

```
jiant -rw,hard <IP Address>:/<Fileshare name>
```

Go to `/path/to/jiant/gcp` and run

```bash
./mount_nfs.sh
```

You should see some basic folders in `/nfs/jiant`. Take this opportunity to do some setup of the NFS, such as:

1. Pull in the relevant task data and save them to `/nfs/jiant/data_dir`
2. Create `/nfs/jiant/home/${USER}` folder, and git clone `jiant`. This should be your working copy of `jiant` hereafter (including for the subsequent step). You may want to repeat the above modification of [`config/auto.nfs`](config/auto.nfs) for that copy of `jiant` too.
3. Create `/nfs/jiant/exp/${USER}` folder

### Kubernetes Configuration Update

Now, go to ``/nfs/jiant/${USER}/gcp/kubernetes``. Modify [`templates/jiant_env.libsonnet`](templates/jiant_env.jsonnet):

* Set `nfs_server_ip` to the Filestore IP Address.
* Set `nfs_server_path` to the Filestore Fileshare name.
* Set `nfs_volume_name` to the Filestore Instance ID.
* Set `nfs_volume_size` to the Filestore capacity.
* Set `gcr_image` to the Docker image URL.

### Final cluster set-up

Finally, run:

```bash
./init_cluster.sh
```

This does two things:

- It adds a DaemonSet that installs GPU drivers on each node.
- It sets up a PersistentVolume that defines the NFS volume in a way Kubernetes
  can understand, and sets up a PersistentVolumeClaim that allows jobs to access it.

======

### Running Jobs

To run jobs, ensure that you have NFS mounted at `/nfs/jiant`, and go to `/nfs/jiant/${USER}/gcp/kubernetes`

We can schedule basic jobs on the cluster using `kubectl run` as in the "Hello World" example above, but in order to use GPUs and access NFS we need to use a full-fledged YAML config.

The `run_batch.sh` script handles creating an appropriate config on the fly; see the documentation in that file for more details. Basic usage is:

```sh
export JIANT_PATH="/nfs/jiant/home/$USER/jiant"
./run_batch.sh <job_name> \
   "python $JIANT_PATH/main.py --config_file $JIANT_PATH/jiant/config/demo.conf --overrides 'run_name = kubernetes-demo, target_tasks = \"wnli,commitbank\"'"
```

You should see your job as `<job_name>` in Kubernetes Engine -> Workloads, and can monitor status, resource usage, and logs from that page.

There are also additional options, such as for sending jobs to different GPU types, and deleting jobs (pods in Kubernetes terminology).

======

### Using an existing GKE cluster

If someone has already set up a GKE cluster for `jiant`, and you have a different set of things to do:

1. Create a lightweight GCP instance and SSH into it. git clone `jiant`.
2. Get the Filestore IP Address and Fileshare name from the cluster owner, and mount the NFS as in [NFS Setup](#nfs-setup). This includes modifying [`config/auto.nfs`](config/auto.nfs) as directed.
3. Set up up your home folder in `/nfs/jiant/home/${USER}`, and clone jiant to `/nfs/jiant/home/${USER}/jiant`. This will be your working copy of `jiant` hereafter.
4. Create `/nfs/jiant/exp/${USER}`
5. Authenticate to the cluster:
    ```sh
    gcloud container clusters get-credentials <cluster_name> --zone <cluster_zone>
    ```
    If you run into an "Insufficient Scope" error, run `gcloud init` and authenticate with the relevant account.
6. Run jobs as in [Running Jobs](#running-jobs).
