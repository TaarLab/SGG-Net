import os
import csv
import json
import numpy as np
from graspnetAPI.graspnet_eval import GraspNetEval
from concurrent.futures import ProcessPoolExecutor, as_completed


def evaluate_and_save_scene(grasp_net_eval, scene_id, dump_folder_path, scene_csv_folder):
    """
    Evaluates a single scene and saves the AP, AP_08, and AP_04 metrics to a CSV file.
    If the scene's CSV already exists, it skips the evaluation.
    """
    scene_csv_path = os.path.join(scene_csv_folder, f'scene_{scene_id}.csv')

    # Skip if this scene has already been processed
    if os.path.exists(scene_csv_path):
        print(f"Scene {scene_id} already processed, skipping.")
        return scene_csv_path

    # Evaluate the scene
    acc = grasp_net_eval.eval_scene(scene_id=scene_id, dump_folder=dump_folder_path, return_list=False, vis=False,
                                    last_ann_id=5)
    np_acc = np.array(acc) * 100  # Convert to percentage
    AP = np.mean(np_acc)
    AP_08 = np.mean(np_acc[..., 3])  # Assuming 3 corresponds to AP_08
    AP_04 = np.mean(np_acc[..., 1])  # Assuming 1 corresponds to AP_04

    # Classify scene as 'Seen', 'Unseen', or 'Novel'
    if 100 <= scene_id <= 129:
        category = 'Seen'
    elif 130 <= scene_id <= 159:
        category = 'Unseen'
    elif 160 <= scene_id <= 189:
        category = 'Novel'
    else:
        raise ValueError(f"Scene ID {scene_id} is out of expected range")

    # Save results for the scene to a CSV file
    with open(scene_csv_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Scene_ID', 'Category', 'AP', 'AP_08', 'AP_04'])
        writer.writeheader()
        writer.writerow({
            'Scene_ID': scene_id,
            'Category': category,
            'AP': AP,
            'AP_08': AP_08,
            'AP_04': AP_04
        })

    return scene_csv_path

def aggregate_results(scene_csv_folder):
    """
    Aggregates the AP results from individual scene CSVs into overall metrics.
    """
    AP_data = {'Seen': [], 'Unseen': [], 'Novel': []}

    # Iterate through all scene CSV files and aggregate the data
    for csv_file in os.listdir(scene_csv_folder):
        if csv_file.endswith('.csv'):
            scene_csv_path = os.path.join(scene_csv_folder, csv_file)
            with open(scene_csv_path, mode='r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    category = row['Category']
                    AP_data[category].append({
                        'AP': float(row['AP']),
                        'AP_08': float(row['AP_08']),
                        'AP_04': float(row['AP_04'])
                    })

    # Calculate mean AP, AP_08, and AP_04 for each category
    results = {}
    for category, data in AP_data.items():
        AP_mean = np.mean([d['AP'] for d in data])
        AP_08_mean = np.mean([d['AP_08'] for d in data])
        AP_04_mean = np.mean([d['AP_04'] for d in data])
        results[category] = {
            'AP': AP_mean,
            'AP_08': AP_08_mean,
            'AP_04': AP_04_mean
        }

    return results


def save_final_results_to_csv(results, methods_name, output_csv):
    """
    Saves the final aggregated AP results to a CSV file in the table format.
    """
    fieldnames = ['Methods', 'Seen AP', 'Seen AP_08', 'Seen AP_04', 'Unseen AP', 'Unseen AP_08', 'Unseen AP_04',
                  'Novel AP', 'Novel AP_08', 'Novel AP_04']

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        writer.writerow({
            'Methods': methods_name,
            'Seen AP': results['Seen']['AP'],
            'Seen AP_08': results['Seen']['AP_08'],
            'Seen AP_04': results['Seen']['AP_04'],
            'Unseen AP': results['Unseen']['AP'],
            'Unseen AP_08': results['Unseen']['AP_08'],
            'Unseen AP_04': results['Unseen']['AP_04'],
            'Novel AP': results['Novel']['AP'],
            'Novel AP_08': results['Novel']['AP_08'],
            'Novel AP_04': results['Novel']['AP_04']
        })

def collect_ap_data(grasp_net_eval, dump_folder_path, scene_range, scene_csv_folder):
    """
    Collects Average Precision (AP), AP_08, AP_04 for different scenes (Seen, Unseen, Novel) in parallel.
    Saves each scene's result to a separate CSV file.
    """
    # Create directory for saving scene-wise CSVs if it doesn't exist
    os.makedirs(scene_csv_folder, exist_ok=True)

    # Use ProcessPoolExecutor for parallel processing of scene evaluations
    with ProcessPoolExecutor(max_workers=15) as executor:
        future_to_scene = {executor.submit(evaluate_and_save_scene, grasp_net_eval, scene_id, dump_folder_path,
                                           scene_csv_folder): scene_id for scene_id in scene_range}

        for future in as_completed(future_to_scene):
            scene_id = future_to_scene[future]
            try:
                scene_csv_path = future.result()
                print(f"Scene {scene_id} processed and saved to {scene_csv_path}")
            except Exception as exc:
                print(f'Scene {scene_id} generated an exception: {exc}')


if __name__ == "__main__":
    # Load configuration
    with open('pathes.json', 'r') as f:
        config = json.load(f)
    dump_folder_path = config["dump_folder"]
    graspnet_root = config['graspnet_root']
    camera = 'realsense'

    scene_csv_folder = os.path.join(graspnet_root, 'scene_csvs')

    grasp_net_eval = GraspNetEval(root=graspnet_root, camera=camera, split='custom')

    # Collect AP data for seen (100-129), unseen (130-159), and novel (160-189) scenes
    collect_ap_data(grasp_net_eval, dump_folder_path, scene_range=range(100, 190), scene_csv_folder=scene_csv_folder)

    # Aggregate results from individual scene CSVs and calculate overall AP values
    results = aggregate_results(scene_csv_folder)

    # Save final results to CSV
    output_csv = os.path.join(graspnet_root, 'evaluation_results.csv')
    save_final_results_to_csv(results, methods_name="Our Method", output_csv=output_csv)

    print(f"Final evaluation results saved to {output_csv}")
