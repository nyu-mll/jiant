import os
import time
import torch
import traceback

from contextlib import contextmanager

import jiant.utils.python.io as py_io
import jiant.utils.python.filesystem as filesystem


class BaseZLogger:
    def log_context(self):
        raise NotImplementedError()

    def write_entry(self, key, entry):
        raise NotImplementedError()

    def write_obj(self, key, obj, entry):
        raise NotImplementedError()

    def flush(self):
        raise NotImplementedError()


class ZLogger(BaseZLogger):
    def __init__(self, fol_path, log_errors=True, overwrite=False):
        self.fol_path = fol_path
        self.log_errors = log_errors
        self.overwrite = overwrite

        self.write_mode = "w" if overwrite else "a"
        os.makedirs(fol_path)
        self.handles = {}

    @contextmanager
    def log_context(self):
        try:
            yield self
        except Exception:
            if self.log_errors:
                self.write_entry("errors", traceback.format_exc())
            raise
        finally:
            for f in self.handles.values():
                f.close()

    def write_entry(self, key, entry, do_print=False):
        if isinstance(entry, dict):
            entry = entry.copy()
        else:
            entry = {"data": entry}
        entry["TIMESTAMP"] = time.time()
        self._write_entry_to_file(key=key, entry=entry)
        if do_print:
            print(entry)

    def write_obj(self, key, obj, entry):
        assert "DATA" not in entry
        if isinstance(entry, dict):
            entry = entry.copy()
        else:
            entry = {"data": entry}
        time_stamp = time.time()
        entry["DATA"] = self._save_obj(key, time_stamp, obj)
        entry["TIMESTAMP"] = time_stamp
        self._write_entry_to_file(key=key, entry=entry)

    def _save_obj(self, key, time_stamp, obj):
        cache_path = self.get_cache_path(key)
        os.makedirs(cache_path, exist_ok=True)
        save_path = os.path.join(cache_path, str(time_stamp))
        torch.save(obj, save_path)
        return save_path

    def check_handle_open(self, key):
        if key in self.handles:
            return
        handle_path = self.get_path(key)
        py_io.create_containing_folder(handle_path)
        self.handles[key] = open(handle_path, self.write_mode)

    def get_path(self, key):
        return os.path.join(self.fol_path, key + ".zlog")

    def get_cache_path(self, key):
        return os.path.join(self.fol_path, key + "___CACHE")

    def flush(self, key=None):
        if key is None:
            for f in self.handles.values():
                f.flush()
        elif isinstance(key, list):
            for k in key:
                self.handles[k].flush()
        else:
            self.handles[key].flush()

    def _write_entry_to_file(self, key, entry):
        self.check_handle_open(key)
        self.handles[key].write(py_io.to_jsonl(entry) + "\n")


class ZBufferedLogger(ZLogger):
    def __init__(
        self,
        fol_path,
        default_buffer_size=1,
        buffer_size_dict=None,
        log_errors=True,
        overwrite=False,
    ):
        super().__init__(fol_path=fol_path, log_errors=log_errors, overwrite=overwrite)
        self.default_buffer_size = default_buffer_size
        self.buffer_size_dict = buffer_size_dict.copy() if buffer_size_dict else {}
        self.buffer_dict = {}

    def check_handle_open(self, key):
        super().check_handle_open(key=key)
        if key not in self.buffer_dict:
            self.buffer_dict[key] = []
            if key not in self.buffer_size_dict:
                self.buffer_size_dict[key] = self.default_buffer_size

    def _write_entry_to_file(self, key, entry):
        self.check_handle_open(key)
        self.buffer_dict[key].append(entry)
        if len(self.buffer_dict[key]) >= self.buffer_size_dict[key]:
            self.flush(key)

    def _write_buffer(self, key):
        if not self.buffer_dict[key]:
            return
        self.handles[key].write(
            "".join(py_io.to_jsonl(entry) + "\n" for entry in self.buffer_dict[key])
        )
        self.buffer_dict[key] = []

    def flush(self, key=None):
        if key is None:
            for k, f in self.handles.items():
                self._write_buffer(k)
                f.flush()
        elif isinstance(key, list):
            for k in key:
                self._write_buffer(k)
                self.handles[k].flush()
        else:
            self._write_buffer(key)
            self.handles[key].flush()


class _VoidZLogger(BaseZLogger):
    def log_context(self):
        yield

    def write_entry(self, key, entry):
        pass

    def write_obj(self, key, obj, entry):
        pass

    def flush(self):
        pass


class _PrintZLogger(BaseZLogger):
    def log_context(self):
        yield

    def write_entry(self, key, entry):
        print(f"{key}: {entry}")

    def write_obj(self, key, obj, entry):
        print(f"{key}: {obj}")

    def flush(self):
        pass


class InMemoryZLogger(BaseZLogger):
    def __init__(self):
        self.entries = {}
        self.data = {}

    def log_context(self):
        yield

    def write_entry(self, key, entry):
        if isinstance(entry, dict):
            entry = entry.copy()
        else:
            entry = {"data": entry}
        entry["TIMESTAMP"] = time.time()
        self._write_entry(key=key, entry=entry)

    def write_obj(self, key, obj, entry):
        assert "DATA" not in entry
        if isinstance(entry, dict):
            entry = entry.copy()
        else:
            entry = {"data": entry}
        time_stamp = time.time()
        entry["DATA"] = obj
        entry["TIMESTAMP"] = time_stamp
        self._write_entry(key=key, entry=entry)

    def _write_entry(self, key, entry):
        if key not in self.entries:
            self.entries[key] = []
        self.entries[key].append(entry)

    def flush(self):
        pass


VOID_LOGGER = _VoidZLogger()
PRINT_LOGGER = _PrintZLogger()


def load_log(fol_path):
    all_paths = filesystem.find_files_with_ext(fol_path, "zlog")
    log_data = {}
    for path in all_paths:
        key = os.path.abspath(path).replace(os.path.abspath(fol_path), "")[1:].replace(".zlog", "")
        log_data[key] = py_io.read_jsonl(path)
    return log_data
