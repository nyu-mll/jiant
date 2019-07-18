local jiant_env = import 'jiant_env.libsonnet';

# Run a jiant job (or anything, really) with the correct path and NFS set-up.
function(job_name, command, project_dir, uid, fsgroup,
         notify_email="", gpu_type="p100") std.manifestYamlDoc({
  apiVersion: "v1",
  kind: "Pod",
  metadata: { name: job_name },
  spec: {
    restartPolicy: "Never",
    securityContext: { 
      runAsUser: std.parseInt(uid), 
      fsGroup: std.parseInt(fsgroup)
    },
    containers: [{
      name: 'jiant',
      image: jiant_env.gcr_image,
      command: ["bash"],
      args: ["-l", "-c", command],
      # Use one GPU.
      resources: { limits: { "nvidia.com/gpu": 1 } },
      # Mount the NFS volume inside the container.
      volumeMounts: [
        { 
          mountPath: jiant_env.nfs_mount_path, 
          name: jiant_env.nfs_volume_name 
        },
      ],
      env: [
        { name: "JIANT_PROJECT_PREFIX", value: project_dir },
        { name: "NOTIFY_EMAIL", value: notify_email },
        { 
          name: "PYTORCH_PRETRAINED_BERT_CACHE", 
          value: jiant_env.bert_cache_path 
        },
      ]
    }],
    # Make sure we request GPU nodes of the correct type.
    nodeSelector: {
      "cloud.google.com/gke-accelerator": "nvidia-tesla-" + gpu_type,
    },
    # Make sure Kubernetes allows us to schedule on GPU nodes.
    tolerations: [{ 
      key: "nvidia.com/gpu",
      operator: "Equal",
      value: "present",
      effect: "NoSchedule",
    }],
    # Connect the pod to the NFS claim on Kubernetes.
    volumes: [{
      name: jiant_env.nfs_volume_name,
      persistentVolumeClaim: {
        claimName: jiant_env.nfs_volume_name + "-claim",
        readOnly: false,
      },
    }],
  },
})
