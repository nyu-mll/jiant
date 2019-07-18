# Shared variables describing the Kubernetes environment.
# You'll likely need to change these if setting up jiant on a new cluster.
{
  # NFS server information.
  nfs_server_ip: "10.87.154.18",
  nfs_server_path: "/data",
  nfs_volume_name: "nfs-jiant",
  nfs_volume_size: "2.5T",

  # Mount point for the NFS volume, as the container will see it.
  nfs_mount_path: "/nfs/jiant",

  # Name of pre-built Docker image, accessible from Kubernetes.
  gcr_image: "gcr.io/google.com/jiant-stilts/jiant:v2",
  
  # Path to BERT model cache; should be writable by Kubernetes workers.
  bert_cache_path: "/nfs/jiant/share/bert_cache",
}
