# Helpers for accessing GCP services

import logging as log  # python standard logging
import os


def get_instance_id():
    # https://stackoverflow.com/questions/31688646/get-the-name-or-id-of-the-current-google-compute-instance  # noqa
    import requests

    metadata_server = "http://metadata/computeMetadata/v1/instance/"
    metadata_flavor = {"Metadata-Flavor": "Google"}
    gce_id = requests.get(metadata_server + "id", headers=metadata_flavor).text
    return gce_id
    #  gce_name = requests.get(metadata_server + 'hostname', headers = metadata_flavor).text
    #  gce_machine_type = requests.get(metadata_server + 'machine-type', headers = metadata_flavor).text  # noqa


def get_remote_log_url(log_name, project_name="jsalt-sentence-rep"):
    instance_id = get_instance_id()
    url = (
        "https://console.cloud.google.com/logs/viewer?"
        "authuser=2&project={project_name:s}"
        "&resource=gce_instance%2Finstance_id%2F{instance_id:s}"
        "&logName=projects%2F{project_name:s}%2Flogs%2F{log_name:s}"
    )
    return url.format(instance_id=instance_id, log_name=log_name, project_name=project_name)


def configure_remote_logging(log_name):
    # Avoid deadlock situation with subprocess. See:
    # https://github.com/GoogleCloudPlatform/google-cloud-python/issues/4992
    # and https://github.com/grpc/grpc/issues/14056#issuecomment-370962039
    os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

    # Set up cloud logging
    from google.cloud import logging as cloud_logging
    from google.cloud.logging.handlers import CloudLoggingHandler
    from google.cloud.logging.resource import Resource

    logging_client = cloud_logging.Client()
    instance_id = get_instance_id()
    log_resource = Resource("gce_instance", {"instance_id": instance_id})
    log.info("Configuring remote logging to %s with log name '%s'", str(log_resource), log_name)
    cloud_handler = CloudLoggingHandler(logging_client, name=log_name, resource=log_resource)
    log.getLogger().addHandler(cloud_handler)
