**Aref Hashemi \& Aliakbar Izadkhah (2024)**  
This repository includes python codes, data, and results for a project on the use of a Graph Neural Network (GNN) for simulation of the dymamics of a multidisperse suspension of partices in a box of fluid.

<p align="center"><img src="sample-rollout.gif" alt="Sample Rollout GIF"></p>

This work is inspired by the following prior works:

*   A\. Sanchez-Gonzalez, J. Godwin, T. Pfaff, R. Ying, J. Leskovec, and P. W. Battaglia. Learning to Simulate Complex Physics with Graph Networks. ICML 2020. Github repository: https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate  
*   K\. Kumar and J. Vantassel. GNS: A generalizable Graph Neural Network-based simulator for particulate and fluid modeling. Journal of Open Source Software 2023. Github repository: https://github.com/geoelements/gns

**Note:** parts of the codes in this repo have been borrowed from the repository https://github.com/geoelements/gns and heavily modified for our own problem. In particular, we use the data_loader script and the corresponding parts of the interface from https://github.com/geoelements/gns.

**How to run the code:**
Use bash script *run.bash* to perform a full training along with rendering and generating animations. Use bash script *rollout_render.bash* to generate outputs from a pretrained model.

**How to cite:**  
A. Hashemi \& A. Izadkhah, A Graph Neural Network Approach to Dispersed Systems. arXiv:2412.02967 (2024); https://doi.org/10.48550/arXiv.2412.02967
