# Structured data for RealSense and Kinect cameras
data = {
    # "GPD 2017": {
    #     "RealSense": {"Seen": [22.87, 28.53, 12.84], "Unseen": [21.33, 27.83, 9.64], "Novel": [8.24, 8.89, 2.67]},
    #     "Kinect": {"Seen": [24.38, 30.16, 13.46], "Unseen": [23.18, 28.64, 11.32], "Novel": [9.5, 10.14, 3.16]}
    # },
    # "GG-CNN 2018": {
    #     "RealSense": {"Seen": [15.48, 21.84, 10.25], "Unseen": [13.26, 18.37, 4.62], "Novel": [5.52, 5.93, 1.86]},
    #     "Kinect": {"Seen": [16.89, 22.47, 11.23], "Unseen": [15.05, 19.76, 6.19], "Novel": [7.38, 8.78, 1.32]}
    # },
    "PointNetGPD 2019~\cite{b3}": {
        "RealSense": {"Seen": [25.96, 33.01, 15.37], "Unseen": [22.68, 29.15, 10.76], "Novel": [9.23, 9.89, 2.74]},
        "Kinect": {"Seen": [27.59, 34.21, 17.83], "Unseen": [24.38, 30.84, 12.83], "Novel": [10.66, 11.24, 3.21]}
    },
    "GraspNet 2020~\cite{b27}": {
        "RealSense": {"Seen": [27.56, 33.43, 16.95], "Unseen": [26.11, 34.18, 14.23], "Novel": [10.55, 11.25, 3.98]},
        "Kinect": {"Seen": [29.88, 36.19, 19.31], "Unseen": [27.84, 33.19, 16.62], "Novel": [11.51, 12.92, 3.56]}
    },
    # "GSNet 2021": {
    #     "RealSense": {"Seen": [67.12, 78.46, 60.9], "Unseen": [54.81, 66.72, 46.17], "Novel": [24.31, 30.52, 14.23]},
    #     "Kinect": {"Seen": [63.5, 74.54, 58.11], "Unseen": [49.18, 59.27, 41.89], "Novel": [19.78, 24.6, 11.17]}
    # },
    "RGBMatter 2021~\cite{b28}": {
        "RealSense": {"Seen": [27.98, 33.47, 17.75], "Unseen": [27.23, 36.34, 15.6], "Novel": [12.25, 12.45, 5.62]},
        "Kinect": {"Seen": [32.08, 39.46, 20.85], "Unseen": [30.4, 37.87, 18.72], "Novel": [13.08, 13.79, 6.01]}
    },
    "TransGrasp 2022~\cite{b29}": {
        "RealSense": {"Seen": [39.81, 47.54, 36.42], "Unseen": [29.32, 34.8, 25.19], "Novel": [13.83, 17.11, 7.67]},
        "Kinect": {"Seen": [35.97, 41.69, 31.86], "Unseen": [29.71, 35.67, 24.19], "Novel": [11.41, 14.42, 5.84]}
    },
    "GraNet 2023~\cite{b30}": {
        "RealSense": {"Seen": [43.33, 52.56, 34.03], "Unseen": [39.98, 48.66, 32.0], "Novel": [14.9, 18.66, 7.76]},
        "Kinect": {"Seen": [41.48, 49.84, 33.86], "Unseen": [35.29, 43.15, 26.89], "Novel": [11.57, 14.31, 5.24]}
    },
    # "Duanmu et al. 2024": {
    #     "RealSense": {"Seen": [46.92, 55.54, 40.58], "Unseen": [41.73, 50.41, 34.65], "Novel": [16.67, 21.12, 8.51]},
    #     "Kinect": {"Seen": [None, None, None], "Unseen": [None, None, None], "Novel": [None, None, None]}
    # },
    # "Tang et al. 2024": {
    #     "RealSense": {"Seen": [75.39, 86.75, 70.6], "Unseen": [65.75, 78.82, 57.52], "Novel": [27.38, 34.17, 14.56]},
    #     "Kinect": {"Seen": [None, None, None], "Unseen": [None, None, None], "Novel": [None, None, None]}
    # },
    # "HGGD 2024": {
    #     "RealSense": {"Seen": [64.45, 72.81, 61.16], "Unseen": [53.59, 64.12, 45.91], "Novel": [24.59, 30.46, 15.58]},
    #     "Kinect": {"Seen": [61.17, 69.82, 56.52], "Unseen": [47.02, 56.78, 38.86], "Novel": [19.37, 23.95, 12.14]}
    # },
    # "Wang et al. 2024": {
    #     "RealSense": {"Seen": [74.33, 85.77, 63.89], "Unseen": [64.36, 76.76, 55.25], "Novel": [27.56, 34.09, 20.23]},
    #     "Kinect": {"Seen": [None, None, None], "Unseen": [None, None, None], "Novel": [None, None, None]}
    # },
    # "EconomicGrasp 2024": {
    #     "RealSense": {"Seen": [62.59, 73.89, 55.99], "Unseen": [51.73, 62.7, 43.45], "Novel": [19.54, 24.24, 11.12]},
    #     "Kinect": {"Seen": [68.21, 79.6, 63.54], "Unseen": [61.19, 70.63, 53.77], "Novel": [25.48, 31.46, 13.85]}
    # },
    # "FlexLoG 2024": {
    #     "RealSense": {"Seen": [72.81, None, None], "Unseen": [65.21, None, None], "Novel": [30.04, None, None]},
    #     "Kinect": {"Seen": [69.44, None, None], "Unseen": [59.01, None, None], "Novel": [23.67, None, None]}
    # },
    "3D StSkel": {
        "RealSense": {"Seen": [45.42, 56.86, 29.58], "Unseen": [40.46, 51.16, 25.69], "Novel": [16.59, 20.37, 7.16]},
        "Kinect": {"Seen": [46.98689358,58.55193033,32.21658931	], "Unseen": [42.26838491,52.20644929,30.39443761], "Novel": [21.1449314,25.54790213,11.92058552]}
    }
}

from collections import defaultdict

def get_ranked_latex(data, camera_name):
    """
    Generate LaTeX code for a specified camera (RealSense or Kinect)
    with rankings for each metric and category.
    """
    # Initialize rank data
    rankings = defaultdict(lambda: defaultdict(dict))

    # Calculate ranks for each category and metric
    categories = ["Seen", "Unseen", "Novel"]
    metrics = ["AP", "AP_0.8", "AP_0.4"]

    for category in categories:
        for metric_index, metric in enumerate(metrics):
            # Extract data for current category and metric for the specified camera
            scores = []
            for method, camera_data in data.items():
                if camera_data[camera_name][category][metric_index] is not None:
                    score = camera_data[camera_name][category][metric_index]
                    scores.append((method, score))

            # Sort scores in descending order and assign ranks
            scores.sort(key=lambda x: x[1], reverse=True)
            for rank, (method, score) in enumerate(scores, start=1):
                rankings[method][category][metric] = (score, rank)

    # Begin LaTeX table generation
    latex_code = f"""
    \\begin{{table*}}[tp]
        \\centering
        \\caption{{Detailed results on the GraspNet dataset ({camera_name}), showing average precision (AP) scores. Indices indicate the ranking position of each method in the respective category.}}
        \\label{{tab:comparison_methods_{camera_name.lower()}}}
        \\begin{{tabular}}{{lccccccccc}}
            \\toprule
            Methods & \\multicolumn{{3}}{{c}}{{Seen}} & \\multicolumn{{3}}{{c}}{{Unseen}} & \\multicolumn{{3}}{{c}}{{Novel}} \\\\
            \\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-7}} \\cmidrule(lr){{8-10}}
                                       & AP (\\%)              & AP$_{{0.8}}$ (\\%)      & AP$_{{0.4}}$ (\\%)      & AP (\\%)              & AP$_{{0.8}}$ (\\%)      & AP$_{{0.4}}$ (\\%)       & AP (\\%)                 & AP$_{{0.8}}$ (\\%)         & AP$_{{0.4}}$ (\\%)          \\\\
            \\midrule
    """

    # Populate table rows with scores and ranks
    for method, camera_data in data.items():
        if method == "3D StSkel":
            latex_code += "            \\midrule\n"  # Add \midrule before "3D StSkel" line
        latex_code += f"            {method} "
        for category in categories:
            for metric in metrics:
                score, rank = rankings[method][category].get(metric, (None, None))
                if score is not None:
                    latex_code += f"& {score:.2f}$^{{{rank}}}$ "
                else:
                    latex_code += "& - "
        latex_code += "\\\\\n"

    latex_code += """            \\bottomrule
        \\end{tabular}
    \\end{table*}
    """

    return latex_code


# Generate LaTeX code for RealSense and Kinect
latex_realsense = get_ranked_latex(data, "RealSense")
latex_kinect = get_ranked_latex(data, "Kinect")

# Display or save LaTeX code for RealSense and Kinect tables
print(latex_realsense)
print(latex_kinect)
