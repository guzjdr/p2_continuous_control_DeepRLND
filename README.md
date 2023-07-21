# Project2: Continuous Control ( Deep RL NanoDegree )

This repository contains material related to Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program. 

## Project Description

Implemented the Twin Delay DDPG ( TD3 ) Algorithm to solve the task provided. 

Task:
    - Implement the TD3 Algorithm to control the two link arm reacher ( unity environment ).
    - The agent is able to receive an average reward of +13 (or higher) over a 100 episode epoch.


## Environment

Unity Reacher Environment - Used to train an agent to solve the task described above. The main idea is to train an agent to learn to move to target locations within the environment. 
[image1]: ./data_files/exp0_mutli_agent_training.png "Multi_Agent_Training"

![alt text][image1]<center>**Unity Reacher Multi-Agent-Environment**</center>


## Getting Started 

Note the following conda environment was created on a Linux machine ( __Linux Ubuntu 20.04.6 LTS__ ) 

1. Please clone the following Udacity Repo: [deep-reinforcement-learning](https://github.com/udacity/deep-reinforcement-learning) Repo.

2. Follow the instructions - inside the repo - to set up the necessary dependencies. 
* Create (and activate) a new conda environment with Python 3.6 : 
```bash
conda create --name drlnd python=3.6
source activate drlnd
```
* Install OpenAI gym
```bash
pip install gym
```
* Install the necessary Python dependencies 
```bash
cd deep-reinforcement-learning/python
```
- Modified Requirements.txt 
```text
Pillow>=4.2.1
matplotlib
numpy>=1.11.0
jupyter
pytest>=3.2.2
docopt
pyyaml
protobuf==3.5.2
grpcio==1.11.0
torch==1.7.0
pandas
scipy
ipykernel
```
```bash
pip install .
```
* Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment. 
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```
* Download the Unity Environment(s):
Please refer to this repo to download the environment files: [p2_continuous-control](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control)
