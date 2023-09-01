# Cost Effective Transfer of Reinforcement Learning Policies
## Abstract
Many challenging real-world problems require the deployment of ensembles to reach acceptable performance levels. While effective, applying the entire ensemble to every sample is costly and often unnecessary. <br>
Deep Reinforcement Learning (DRL) offers a cost-effective alternative, where detectors are dynamically chosen based on the output of their predecessors, with their usefulness weighted against their computational cost. <br>
Despite their potential, DRL-based solutions are not widely used in ensemble management. <br>
This can be attributed to the difficulties in configuring the reward function for each new task, the unpredictable reactions of the DRL agent to changes in the data, and the inability to use common performance metrics (e.g., True and False-Positive Rates, TPR/FPR) to guide the DRL model in a multi-objective environment. <br> 
We propose methods for fine-tuning and calibrating DRL-based policies to meet multiple performance goals. Moreover, we present a method for transferring effective security policies from one dataset to another. Finally, we demonstrate that our approach is highly robust against adversarial attacks. <br>

## Folder Tree
1. base - base CETRA method: <br>
envs - gym environment<br>
pretrained_weights - dir for the pretrained model files <br>

2. metric - metric CETRA method: <br>
envs - gym environment<br>
pretrained_weights - dir for the pretrained model files <br>

3. attack - attack CETRA method: <br>
envs - gym environment<br>
pretrained_weights - dir for the pretrained model files <br>
Note! in attack there is currently only test without train. <br>

4. data - dir for Bahsen/Wang dataset file <br>
Download the datasets: <br>
[Click Here](https://drive.google.com/drive/folders/1rg5Fs638uM2eei5Z9o0xbDMTkKSkgXuB?usp=sharing) <br>

5. Exp. 2 of CETRA for all 3 versions of both datasets is available there <br>
Note! Bahnsen versions are already inside pretrained_weights dir of base/metric/attack for easier use  <br>

6. requirements.txt <br>

## HOW TO Run

1. Create conda env (recommended) and run the cmd pip install -r requirements.txt <br>
2. Download the dataset you want to run CETRA on and put it in the data folder (see no.4 in Folder Tree)  <br>
3. cd into the dir of the version of CETRA you want to run (base/metric/attack) and run the cmd: <br>
	python main.py <DATASET_NAME> <EXPERIMENT_NUMBER> (e.g., python main.py Wang 2 or python main.py Bahnsen 2)<br>
4. If you want to change any of the hyper parameters, update config.py accordingly in each version dir (comment near each hyper parameter offers the available options) <br>
5. Have fun!



