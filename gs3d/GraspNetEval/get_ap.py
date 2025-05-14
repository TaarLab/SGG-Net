import os
import csv
import json

import numpy as np

from graspnetAPI.graspnet_eval import GraspNetEval


def extract_grasp_data(grasp_list, score_list):
    grasp_data = []
    for grasp, score in zip(grasp_list, score_list):
        translation = grasp[:3]
        rotation = grasp[3:12]
        grasp_data.append({
            'score': score,
            'translation_x': translation[0],
            'translation_y': translation[1],
            'translation_z': translation[2],
            'rotation_1': rotation[0], 'rotation_2': rotation[1], 'rotation_3': rotation[2],
            'rotation_4': rotation[3], 'rotation_5': rotation[4], 'rotation_6': rotation[5],
            'rotation_7': rotation[6], 'rotation_8': rotation[7], 'rotation_9': rotation[8],
        })
    return grasp_data


def write_header(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=[
            'scene_id', 'annotation_id', 'score',
            'translation_x', 'translation_y', 'translation_z',
            'rotation_1', 'rotation_2', 'rotation_3',
            'rotation_4', 'rotation_5', 'rotation_6',
            'rotation_7', 'rotation_8', 'rotation_9'
        ])
        writer.writeheader()


def append_to_csv(data, csv_file):
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=[
            'scene_id', 'annotation_id', 'score',
            'translation_x', 'translation_y', 'translation_z',
            'rotation_1', 'rotation_2', 'rotation_3',
            'rotation_4', 'rotation_5', 'rotation_6',
            'rotation_7', 'rotation_8', 'rotation_9'
        ])
        for row in data:
            writer.writerow(row)


if __name__ == "__main__":
    # graspnet_root = r"E:\GraspNet"
    # dump_folder_path =r"E:\TaarLab\3D-Skeleton\dump_folder"
    with open('pathes.json', 'r') as f:
        config = json.load(f)
    dump_folder_path = config["dump_folder"]
    graspnet_root = config['graspnet_root']
    camera = 'realsense'

    grasp_net_eval = GraspNetEval(root=graspnet_root, camera=camera, split='custom')

    csv_output_path = os.path.join(graspnet_root, 'grasp_data.csv')

    # Write the header to the CSV file before processing
    # write_header(csv_output_path)

    for scene_id in range(100, 190):
        acc, grasp_list_list, score_list_list, collision_list_list = grasp_net_eval.eval_scene(
            scene_id=scene_id,
            dump_folder=dump_folder_path,
            return_list=True,
            vis=True
        )

        for annotation_id, (grasp_list, score_list) in enumerate(zip(grasp_list_list, score_list_list)):
            grasp_data = extract_grasp_data(grasp_list, score_list)

            for entry in grasp_data:
                entry['scene_id'] = scene_id
                entry['annotation_id'] = annotation_id

            # Append to CSV in real-time
            # append_to_csv(grasp_data, csv_output_path)
        print()
        np_acc = np.array(acc)
        print('Mean Accuracy: {}'.format(np.mean(np_acc)))
        print(f"Scene {scene_id} processed and data appended to CSV.")

    print(f"Grasp data successfully saved to {csv_output_path}")
