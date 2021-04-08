# Navigation Project Reinforcement Learning Nanodegree 

## Project

This projects trains a DQN Agent that navigates through a world and collects bananas. 

![image gif](banana.gif)


A reward of +1 is provided for collecting a yellow banana, 
and a reward of -1 is provided for collecting a blue banana. 
Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, 
along with ray-based perception of objects around the agent's forward direction. 
Given this information, the agent has to learn how to best select actions. 
Four discrete actions are available, corresponding to:   


0 - move forward   
1 - move backward   
2 - turn left     
3 - turn right 
   
The task is episodic, and in order to solve the environment, your agent must get an average 
score of +13 over 100 consecutive episodes.

## Setup Instructions

### Python 

Setup your Python Enviornment according to the instructions by [Udacity](https://github.com/udacity/deep-reinforcement-learning#dependencies) (as shown below). This will install important frameworks like PyTorch and the ML-Agents toolkit.   

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 




## Unity Banana Framework 

Select the file that corresponds to your operating system (see bewlo).  
Then place the file in the DRLND GitHub repository, in the p1_navigation/ folder, and unzip (or decompress) the file.

[Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)     

[Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip) 

[Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)


## Instructions 
For running the code refer to the markdown comments in the Navigation.ipynb file. 

A checkpoint files trained for 624 episodes is provided. It can be loaded by calling : 
```python 
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth')
```
