
### Aref Hashemi \& Aliakbar Izadkhah (2024)  
This repository includes python codes, data, and results for a project on the use of a Graph Neural Network (GNN) for simulation of the dymamics of a multidisperse suspension of partices in a box of fluid.

<p align="center"><img src="sample-rollout.gif" alt="Sample Rollout GIF"></p>

This work is inspired by the following prior works:

*   A\. Sanchez-Gonzalez, J. Godwin, T. Pfaff, R. Ying, J. Leskovec, and P. W. Battaglia. Learning to Simulate Complex Physics with Graph Networks. ICML 2020. Github repository: https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate  
*   K\. Kumar and J. Vantassel. GNS: A generalizable Graph Neural Network-based simulator for particulate and fluid modeling. Journal of Open Source Software 2023. Github repository: https://github.com/geoelements/gns

**Note:** parts of the codes in this repo have been borrowed from the repository https://github.com/geoelements/gns and heavily modified for our own problem. In particular, we use the data_loader script and the corresponding parts of the interface from https://github.com/geoelements/gns.

### Installation (These specific versions work):

1. Install **CUDA 12.4** and the appropriate **NVIDIA driver**.  
   *(CUDA 11.7 would also work with PyTorch 1.13.1 and related dependencies.)*

2. Install the required Python packages with the following commands:  
   *   pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --index-url https://download.pytorch.org/whl/cu117  
   *   pip3 install torch-scatter==2.1.0 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html  
   *   pip3 install torch-sparse==0.6.17 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html  
   *   pip3 install torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html  
   *   pip3 install torch-geometric==2.2.0  
   *   pip3 install numpy==1.23.5  

### How to run the code:
Use bash script *run.bash* to perform a full training along with rendering and generating animations. Use bash script *rollout_render.bash* to generate outputs from a pretrained model.

### How to cite:  
Aref Hashemi \& Aliakbar Izadkhah, A graph neural network simulation of dispersed systems,  
*Mach. Learn.: Sci. Technol.* **6** 015044 (2025)  
DOI: 10.1088/2632-2153/adb0a0
