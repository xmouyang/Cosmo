# Cosmo: Contrastive Fusion Learning with Small Data
This is a repo for MobiCom 2022 paper: " <a href="https://github.com/xmouyang/xmouyang.github.io/blob/master/data/Cosmo-MobiCom22-camera-ready-0903.pdf"> Cosmo: Contrastive Fusion Learning with Small Data for Multimodal Human Activity Recognition </a>".

# Requirements
The program has been tested in the following environment: 
* Python 3.9.7
* Pytorch 1.8.1
* torchvision 0.9.1
* sklearn 0.24.2
* opencv-python 4.5.5
* numpy 1.20.3

# Cosmo Overview
<p align="center" >
	<img src="https://github.com/xmouyang/Cosmo/blob/main/materials/Overview.png" width="700">
</p>

* Cosmo on the cloud: 
	* contrastive fusion learning for capturing consistent information from unlabeled multimodal data;
	* evaluate data quality for each modality according to unlabeled multimodal data.
* Cosmo on edge: 
	* initialize the feature encoders and the fusion weights of data quality learned on the cloud; 
	* iterative fusion learning for combining complementary information from limited labeled multimodal data.


# Project Strcuture
```
|--sample-code-UTD // sample code of each approach on the UTD dataset

 |-- Cosmo                    // codes of our approach Cosmo
    |-- main_con.py/	// main file of contrastive fusion learning on cloud 
    |-- main_linear_iterative.py/	// main file of supervised learning on edge
    |-- data_pre.py/		// prepare for the data
    |-- cosmo_design.py/ 	// fusion-based feature augmentation and contrastive fusion loss
    |-- cosmo_model_guide.py/	// models of unimodal feature encoders and the quality-quided attention based classifier
    |-- util.py/	// utility functions

 |-- CMC                    // codes of the baseline Contrastive Multi-view Learning (CMC)
    |-- main_con.py/	// main file of contrastive multi-view learning
    |-- main_linear_iterative.py/	// main file of supervised learning
    |-- data_pre.py/		// prepare for the data
    |-- cosmo_design.py/ 	//  feature augmentation and contrastive loss
    |-- cosmo_model_guide.py/	// models of unimodal feature encoders and the attention based classifier
    |-- util.py/	// utility functions

 |-- supervised-baselines                    // codes of the supervised learning baselines
    |-- attnsense_main_ce.py/	// main file of AttnSense
    |-- attnsense_model.py/	// models of AttnSense
    |-- deepsense_main_ce.py/	// main file of DeepSense
    |-- deepsense_model.py/	// models of DeepSense
    |-- single_main_ce.py/	// main file of single modal learning
    |-- single_model.py/	// models of single modal learning (IMU and skeleton)
    |-- data_pre.py/		// prepare for the multimodal and singlemodal data
    |-- util.py/	// utility functions
    
|--UTD-data 	// zipped folder for the processed and splited data in the UTD dataset

|-- README.md

|-- materials               // figures and materials used this README.md
```
<br>

# Quick Start
* Download the `sample-code-UTD` folder and `UTD-data.zip` to your machine, then put `UTD-data.zip` the folder `sample-code-UTD` and unzipped.
* Run the following code for our approach Cosmo
    ```bash
    cd ./sample-code-UTD/Cosmo/
    python3 main_con.py --batch_size 32 --label_rate 5 --learning_rate 0.01
    python3 main_linear_iterative.py --batch_size 16 --label_rate 5 --learning_rate 0.001 --guide_flag 1 --method iterative
    ```
* Run the following code for the baseline Contrastive Multi-view Learning (CMC)
    ```bash
    cd ./sample-code-UTD/CMC/
    python3 main_con.py --batch_size 32 --label_rate 5 --learning_rate 0.01
    python3 main_linear.py --batch_size 16 --label_rate 5 --learning_rate 0.001
    ```
    
* Run the following code for the supervised learning baselines
    ```bash
    cd ./sample-code-UTD/supervised-baselines/
    python3 attnsense_main_ce.py --batch_size 16 --label_rate 5 --learning_rate 0.001
    python3 deepsense_main_ce.py --batch_size 16 --label_rate 5 --learning_rate 0.001
    python3 single_main_ce.py --modality inertial --batch_size 16 --label_rate 5 --learning_rate 0.001
    python3 single_main_ce.py --modality skeleton --batch_size 16 --label_rate 5 --learning_rate 0.001
    ```
    
 * Note: For the CPC baseline on the USC dataset, please refer to <a href="https://github.com/harkash/contrastive-predictive-coding-for-har">this repo</a>. 
 
    ---

# Repositories utilized in this project
This project is based on the implementations of supervised contrastive learning and contrastive multi-view learning detailed in the following repositories: 
<a href="https://github.com/HobbitLong/SupContrast">SupContrast</a> and <a href="https://github.com/HobbitLong/CMC">CMC</a>. They are very useful for this project.


# Citation
The code of this project are made available for non-commercial, academic research only. If you would like to use the code of this project, please cite the following paper:
```
@inproceedings{ouyang2022cosmo,
  title={Cosmo: contrastive fusion learning with small data for multimodal human activity recognition},
  author={Ouyang, Xiaomin and Shuai, Xian and Zhou, Jiayu and Shi, Ivy Wang and Xie, Zhiyuan and Xing, Guoliang and Huang, Jianwei },
  booktitle={Proceedings of the 28th Annual International Conference On Mobile Computing And Networking},
  year={2022}
}
```
    
