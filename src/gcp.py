# Helpers for accessing GCP services

import logging as log  # python standard logging


def get_instance_id():
    # https://stackoverflow.com/questions/31688646/get-the-name-or-id-of-the-current-google-compute-instance
    import requests
    metadata_server = "http://metadata/computeMetadata/v1/instance/"
    metadata_flavor = {'Metadata-Flavor': 'Google'}
    gce_id = requests.get(metadata_server + 'id', headers=metadata_flavor).text
    return gce_id
    #  gce_name = requests.get(metadata_server + 'hostname', headers = metadata_flavor).text
    #  gce_machine_type = requests.get(metadata_server + 'machine-type', headers = metadata_flavor).text


def configure_remote_logging(log_name):
    # Set up cloud logging
    from google.cloud import logging as cloud_logging
    from google.cloud.logging.handlers import CloudLoggingHandler
    from google.cloud.logging.resource import Resource
    logging_client = cloud_logging.Client()
    instance_id = get_instance_id()
    log_resource = Resource("gce_instance", {"instance_id": instance_id})
    log.info("Configuring remote logging to %s with log name '%s'",
             str(log_resource), log_name)
    cloud_handler = CloudLoggingHandler(logging_client, name=log_name,
                                        resource=log_resource)
    log.getLogger().addHandler(cloud_handler)
