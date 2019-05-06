import h5py


def copy_h5py_file(src_filepath, dst_filename):
    """Writes the content of a hdf5 file (specified by src_filepath)
       to another hdf5 file (specified by dst_filename).

    Args:
        src_filepath: a string specifying path of the source hdf5 file.
        dst_filename: a string naming the dst_filename.  If the file already exists,
            then content in the original file will be written over.

    Returns:
        A h5py File object which is a copy of the h5py File object corresponding
        to src_filepath.
    """
    src_file = h5py.File(src_filepath, "r")
    dst_file = h5py.File(dst_filename, "w")
    keys = list(src_file["/"].keys())
    for key in keys:
        src_file["/"].copy(src_file["/" + key], dst_file["/"], name=key)
    src_file.close()
    return dst_file
