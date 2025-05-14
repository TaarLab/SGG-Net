<h1 align="center">SGG-Net: Skeleton and Graph-Based Neural
Network Approaches for Grasping Objects</h1>

## 📌Overview
This project includes the full implementation of the 3D-Skeleton-based grasp pair generation algorithm and the SGG-Net model for robotic grasp scoring. Specifically, it provides code for generating grasp candidates from 3D point clouds using geometric skeletonization (3D-StSkel), along with pairing strategies and grasp pose estimation methods. The repository contains modules for point cloud slicing, 2D polygon formation via alpha shapes, straight skeleton construction, and scoring of grasp candidates using the EGD metric.

Additionally, the codebase includes the implementation of the Heatmap Graph Grasp Quality Network (HGGQ-Net), a graph neural network that evaluates the quality of grasp candidates. There are also evaluation scripts for benchmarking performance on widely used datasets such as GraspNet-1Billion, Dex-Net, and EGAD. These scripts handle data loading, preprocessing (including segmentation and voxelization), model inference, and quantitative evaluation using metrics like grasp success rate and average precision.

## 🚧 Updates
The full version will be released soon.