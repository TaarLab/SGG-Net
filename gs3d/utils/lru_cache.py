import os
from collections import OrderedDict

import zarr


class LRUCache:
    def __init__(self, capacity, zarr_path=None):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.zarr_path = zarr_path

        if self.zarr_path:
            self.load_cache_from_disk()

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value
        self.save_cache_to_disk()

    def save_cache_to_disk(self):
        if not self.zarr_path:
            raise ValueError("Zarr path not specified for saving cache.")

        zarr_store = zarr.open(self.zarr_path)

        for key, value in self.cache.items():
            if key in zarr_store.keys():
                continue
            zarr_group = zarr_store.create_group(str(key))
            if len(value) == 2:
                zarr_group['node_features'] = value[0]
                zarr_group['edge_index'] = value[1]
            else:
                zarr_group['data'] = value


    def load_cache_from_disk(self):
        if not self.zarr_path:
            raise ValueError("Zarr path not specified for loading cache.")

        if not os.path.exists(self.zarr_path):
            print(f"No existing cache found at {self.zarr_path}. Starting with an empty cache.")
            return

        zarr_store = zarr.open(self.zarr_path, mode='r')

        for key in zarr_store.keys():
            zarr_group = zarr_store[str(key)]
            value = (zarr_group['node_features'][...], zarr_group['edge_index'][...])
            self.cache[key] = value

        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def clear_cache(self):
        self.cache.clear()
        if self.zarr_path:
            zarr_store = zarr.open(self.zarr_path, mode='w')
            zarr_store.clear()