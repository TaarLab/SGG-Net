import os
import glob
import asyncio
import pandas as pd

async def main():
    result_pattern = "2024-12-14_*"

    results_base_dir = "results"

    result_dirs = glob.glob(os.path.join(results_base_dir, result_pattern))

    object_success_map = {}

    for result_dir in result_dirs:
        csv_file_path = os.path.join(result_dir, "object_success_rates.csv")

        if os.path.exists(csv_file_path):
            df = pd.read_csv(csv_file_path)

            df['DatasetName'] = df['Object File'].apply(lambda x: x.split(os.sep)[1])

            for _, row in df.iterrows():
                obj_file = row['Object File']
                is_successful = row['Successful Grasps'] > 0
                dataset_name = row['DatasetName']

                if obj_file not in object_success_map:
                    object_success_map[obj_file] = {
                        "Object File": obj_file,
                        "IsGraspedSuccessful": is_successful,
                        "DatasetName": dataset_name
                    }
                else:
                    object_success_map[obj_file]["IsGraspedSuccessful"] |= is_successful

    final_df = pd.DataFrame.from_dict(object_success_map, orient='index')

    output_path = os.path.join(results_base_dir, "combined_object_success_rates.csv")
    final_df.to_csv(output_path, index=False)

    print(f"Combined results saved to {output_path}")

    create_latex_table(final_df)


def create_latex_table(df):
    dataset_stats = df.groupby('DatasetName').agg(
        TotalObjects=('Object File', 'count'),
        SuccessfulGrasps=('IsGraspedSuccessful', 'sum')
    ).reset_index()

    dataset_stats['SuccessRate'] = dataset_stats['SuccessfulGrasps'] / dataset_stats['TotalObjects']

    # Start LaTeX table using tabularray for horizontal image layout
    latex_code = "\\begin{table*}[tp]\n\\centering\n\\caption{Comparison of grasp success rates across datasets.}\n\\label{tab:grasp_results}\n\\begin{tblr}{colspec={X[1,c,m]X[1,c,m]X[1,c,m]X[1,c,m]X[5,c,m]},rowsep=6pt,hlines={black,0.8pt}}\n"
    latex_code += "Dataset & Total Objects & Successful Grasps & Success Rate & Objects\\\\\n"
    latex_code += "\\SetCell{c,m}\\hline\n"

    for _, row in dataset_stats.iterrows():
        dataset_name = row['DatasetName']
        total_objects = row['TotalObjects']
        successful_grasps = row['SuccessfulGrasps']
        success_rate = f"{row['SuccessRate']:.2%}".replace('%', '\\%')

        # Example images in horizontal layout using tabular inside the cell
        example_images = "\\begin{tabular}{cccc}"
        example_images += " & ".join([f"\\includegraphics[width=0.1\\textwidth,height=0.05\\textheight,keepaspectratio]{{Figures/datasets/{dataset_name}_{i}.png}}" for i in range(1, 5)])
        example_images += "\\end{tabular}"

        latex_code += f"{dataset_name} & {total_objects} & {successful_grasps} & {success_rate} & {example_images}\\\\\n"

    latex_code += "\\end{tblr}\n\\end{table*}"

    # Save LaTeX code to a file
    latex_output_path = os.path.join("results", "grasp_results_table.tex")
    with open(latex_output_path, "w") as latex_file:
        latex_file.write(latex_code)

    print(f"LaTeX table saved to {latex_output_path}")



if __name__ == "__main__":
    asyncio.run(main())
