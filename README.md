# Cosmo: Contrastive Fusion Learning with Small Data
This is a repo for MobiCom 2022 paper: " <a href="https://dl.acm.org/doi/10.1145/3458864.3467681"> Cosmo: Contrastive Fusion Learning with Small Data for Multimodal Human Activity Recognition </a>".

# Requirements
The program has been tested in the following environment: 
* Python 3.9.7
* Pytorch 1.8.1
* torchvision 0.9.1
* sklearn 0.24.2
* opencv-python 4.5.5
* numpy 1.20.3
<br>

# Cosmo Overview
<p align="center" >
	<img src="https://github.com/xmouyang/ClusterFL/blob/main/figures/ClusterFL-system-overview.png" width="500">
</p>

* Cosmo on the cloud: 
	* do local training with the collabrative learning variables;
	* communicate with the server.
* Cosmo on edge: 
	* recieve model weights from the clients;
	* learn the relationship of clients;
	* update the collabrative learning variables and send them to each client.


# Project Strcuture
```
|-- client                    // code in client side
    |-- client_cfmtl.py/	// main file of client 
    |-- communication.py/	// set up communication with server
    |-- data_pre.py/		// prepare for the FL data
    |-- model_alex_full.py/ 	// model on client 
    |-- desk_run_test.sh/	// run client 

|-- server/    // code in server side
    |-- server_cfmtl.py/        // main file of client
    |-- server_model_alex_full.py/ // model on server 

|-- README.md

|-- pictures               // figures used this README.md
```
<br>

# Quick Start
* Download the `dataset` folders (collected by ourself) from [FL-Datasets-for-HAR](https://github.com/xmouyang/FL-Datasets-for-HAR) to your client machine.
* Chooose one dataset from the above four datasets and change the "read-path" in 'data_pre.py' to the path on your client machine.
* Change the 'server_x_test.txt' and 'server_y_test.txt' according to your chosen dataset, default as the one for "imu_data_7".
* Change the "server_addr" and "server_port" in 'client_cfmtl.py' as your true server address. 
* Run the following code on the client machine
    ```bash
    cd client
    ./desk_run_test.sh
    ```
* Run the following code on the server machine
    ```bash
    cd server
    python3 server_cfmtl.py
    ```
    ---

# Citation
If you find this work or the datasets useful for your research, please cite this paper:
```
@inproceedings{ouyang2021clusterfl,
  title={ClusterFL: a similarity-aware federated learning system for human activity recognition},
  author={Ouyang, Xiaomin and Xie, Zhiyuan and Zhou, Jiayu and Huang, Jianwei and Xing, Guoliang},
  booktitle={Proceedings of the 19th Annual International Conference on Mobile Systems, Applications, and Services},
  pages={54--66},
  year={2021}
}
```
    
