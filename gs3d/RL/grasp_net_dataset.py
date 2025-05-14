import json

import numpy as np
import zarr
import os
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial.transform import Rotation
from graspnetAPI import GraspNet


class GraspDatasetToZarr:
    def __init__(self, grasp_net, zarr_file="grasp_net_dataset.zarr"):
        self.grasp_net = grasp_net
        self.zarr_file = zarr_file

        # Check if Zarr file exists; if not, create a new Zarr group
        if os.path.exists(self.zarr_file):
            self.zarr_store = zarr.open(self.zarr_file, mode='a')  # Open in append mode if it exists
        else:
            self.zarr_store = zarr.open(self.zarr_file, mode='w')  # Create a new Zarr store

    def save_grasp_data_concurrently(self):
        # Process tasks concurrently
        with ProcessPoolExecutor(max_workers=5) as executor:
            futures = []
            for scene_number in range(100):
                for ann_id in range(1):
                    futures.append(executor.submit(self.process_and_save_scene_ann, scene_number, ann_id))

            # Wait for all futures to complete
            for future in futures:
                try:
                    future.result()  # Raise any exceptions caught during processing
                except Exception as e:
                    print(f"Error in concurrent execution: {e}")

        print(f"Grasp data saved to {self.zarr_file}.")

    def process_and_save_scene_ann(self, scene_number, ann_id):
        # Process and save data chunk for a scene and annotation
        data_chunk = self.process_scene_annotation(scene_number, ann_id)
        if len(data_chunk) > 0:
            self.write_data_chunk(scene_number, ann_id, data_chunk)

    def process_scene_annotation(self, scene_number, ann_id):
        # Process each scene and annotation pair
        if self.scene_annotation_exists(scene_number, ann_id):
            return []

        new_data = []
        try:
            grasp_group = self.grasp_net.loadGrasp(scene_number, ann_id)

            for grasp_index, grasp in enumerate(grasp_group):
                grasp_translation = grasp.translation.tolist()  # Convert translation to list
                grasp_rotation = Rotation.from_matrix(
                    grasp.rotation_matrix).as_quat().tolist()  # Convert rotation to quaternion
                grasp_score = grasp.score

                new_data.append([
                    scene_number, ann_id, grasp_index,
                    grasp_translation[0], grasp_translation[1], grasp_translation[2],
                    grasp_rotation[0], grasp_rotation[1], grasp_rotation[2], grasp_rotation[3],
                    grasp_score
                ])

        except Exception as e:
            print(f"Error processing scene {scene_number}, ann_id {ann_id}: {e}")
            return []

        return new_data

    def scene_annotation_exists(self, scene_number, ann_id):
        # Check if the combination of scene_number and ann_id already exists in the Zarr store
        if f"{scene_number}_{ann_id}" in self.zarr_store:
            return True
        return False

    def write_data_chunk(self, scene_number, ann_id, data_chunk):
        group_name = f"{scene_number}_{ann_id}"

        if group_name in self.zarr_store:
            del self.zarr_store[group_name]

        self.zarr_store.create_dataset(group_name, data=data_chunk, chunks=True, compression='zlib')


    def concat_all_groups_into_one(self):
        # Initialize an empty list to hold the data from all groups
        all_data = []

        # Iterate over all groups (each scene and ann_id combination)
        for group_name in self.zarr_store.array_keys():
            if group_name == 'all':
                continue
            # Load each dataset corresponding to a group
            data = self.zarr_store[group_name][:]
            all_data.append(data)

        # Concatenate all arrays
        all_data_combined = np.vstack(all_data)

        # Create the 'all' dataset if it exists, delete it first
        if 'all' in self.zarr_store:
            del self.zarr_store['all']

        # Save the concatenated dataset into a new group 'all'
        self.zarr_store.create_dataset('all', data=all_data_combined, chunks=True, compression='zlib')

        print(f"All groups concatenated into 'all' dataset in {self.zarr_file}.")

def main():
    with open('pathes.json', 'r') as f:
        config = json.load(f)
    graspnet_root = config['graspnet_root']
    grasp_net = GraspNet(graspnet_root, camera="kinect", split='custom')

    # Create the GraspDatasetToZarr object
    dataset_to_zarr = GraspDatasetToZarr(grasp_net)

    # Save grasp data into Zarr file using concurrent processing
    dataset_to_zarr.save_grasp_data_concurrently()

    # Concatenate all groups into one 'all' group
    dataset_to_zarr.concat_all_groups_into_one()


if __name__ == "__main__":
    main()
