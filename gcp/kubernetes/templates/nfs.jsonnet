local jiant_env = import 'jiant_env.libsonnet';

std.manifestYamlStream([
  # Define persistent volume
  {
    apiVersion: "v1",
    kind: "PersistentVolume",
    metadata: { name: jiant_env.nfs_volume_name, },
    spec: {
      capacity: { storage: jiant_env.nfs_volume_size, },
      accessModes: [ "ReadWriteMany" ],
      nfs: {
        path: jiant_env.nfs_server_path,
        server: jiant_env.nfs_server_ip,
      }, 
    },
  },
  # Define persistent volume claim
  {
    apiVersion: "v1",
    kind: "PersistentVolumeClaim",
    metadata: { name: jiant_env.nfs_volume_name + "-claim", },
    spec: {
      accessModes: [ "ReadWriteMany" ],
      storageClassName: "",
      resources: { requests: { storage: jiant_env.nfs_volume_size } }
    },
  },
])
