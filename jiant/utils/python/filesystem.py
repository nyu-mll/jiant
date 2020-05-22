import os


def find_files(base_path, func):
    return sorted(
        [
            os.path.join(dp, filename)
            for dp, dn, filenames in os.walk(base_path)
            for filename in filenames
            if func(filename)
        ]
    )


def find_files_with_ext(base_path, ext):
    return find_files(base_path=base_path, func=lambda filename: filename.endswith(f".{ext}"))
