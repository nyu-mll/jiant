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

  # Default experiment directory; must be writable by Kubernetes workers.
  nfs_exp_dir: "/nfs/jiant/exp",

  # Name of pre-built Docker image, accessible from Kubernetes.
  gcr_image: "gcr.io/google.com/jiant-stilts/jiant-conda:v2",

  # Default location for glue_data
  jiant_data_dir: "/nfs/jiant/share/glue_data",
  # Path to ELMO cache.
  elmo_src_dir: "/nfs/jiant/share/elmo",
  # Path to BERT etc. model cache; should be writable by Kubernetes workers.
  pytorch_transformers_cache_path: "/nfs/jiant/share/pytorch_transformers_cache",
  # Path to default word embeddings file
  word_embs_file: "/nfs/jiant/share/wiki-news-300d-1M.vec",
}
