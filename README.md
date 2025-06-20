# 🤖 **HumanoidGen: Data Generation for Bimanual Dexterous Manipulation via LLM Reasoning**

<div align="center">

[[Website]](https://humanoidgen.github.io/)
[[Arxiv(Coming Soon)]]()
[[Dataset(Coming Soon)]]()

<img src="./web/main_pipline.png"/>

</div>

## ⚙️ **Setup Environment**

### **Requirements**

* Supported platform: Linux (Ubuntu 20.04)
* Python 3.9

### **Installation**

#### 1. **Clone the repository**

```sh
git clone git@github.com:TeleHuman/HumanoidGen.git
cd HumanoidGen
```

#### 2. **Create & Build conda env**

```sh
conda create --name humanoidgen python=3.9 -y
conda activate humanoidgen
pip install -r requirements.txt
pip install -e .
```

#### 3. **Install pytorch3d & dp & dp3**

Install pytorch3d:

```sh
cd third_party/pytorch3d_simplified && pip install -e . && cd ../..
```

Install dp:

```sh
cd humanoidgen/policy/Diffusion-Policy && pip install -e . && cd ../../..
```

Install dp3:

```sh
cd humanoidgen/policy/3D-Diffusion-Policy/3D-Diffusion-Policy && pip install -e . && cd ../../../..
```

## 🚀 **Getting Started**

### 1. **Change Execution Path**

```sh
cd humanoidgen
```

### 2. **Show Scene & Run Task**

In this project, we provide standard scenes and execution code for 20 tasks, which can be quickly run using the script files below:

```sh
bash scripts/run_scene.sh
bash scripts/run_solve.sh
```

In addition, you can specify main parameters either by directly modifying the sh script file or by using the commands below:

```sh
python process/run_scene.py -env block_handover -render False
python process/run_solve.py -env block_handover -solve block_handover -render False
```

To set more parameters, configure the config file. For details, see [Configuration Instructions](./CONFIGURATION.md).

### 3. **Collect & Visualize Data**

To collect data, open the configuration file [config_run_solve.yml](./humanoidgen/config/config_run_solve.yml) and set "record\_data" to true. Then run the following command (example for ‘block_handover‘ task):

```sh
python process/run_solve.py -env block_handover -solve block_handover -render False
```

The datasets are generated in folder [datasets](./humanoidgen/datasets) and can be visualized using the commands below:

```sh
python process/show_datasets.py
```

The visualization parameters are set in the configuration file [config\_show\_datasets.yml](./humanoidgen/config/config_show_datasets.yml).

### 4. **Train & Deploy Policy**

Firstly, pre-process the generated datasets for training policy. 

```sh
python process/pkl2zarr.py
```

Datasets path, policy model and more parameters are set in the configuration file [config\_show\_datasets.yml](./humanoidgen/config/config_show_datasets.yml).

Dp3 policy train:

```sh
bash scripts/train.sh dp3
```



### 5. **Generate Task Execution Code**

This project supports two generation methods: direct generation and using MCTS. The execution commands are as follows:

```sh
bash scripts/run_generate_solve.sh block_handover 5
bash scripts/run_generate_solve_with_mcts.sh blocks_stack_hard_mcts 5
```

The first argument specifies the name of the generated task, and the second argument is the number of parallel threads to run. If it is necessary to interrupt the generation process, run the following command:

```sh
bash scripts/kill_all_generate_processes.sh
```

## 📦 **Code to be released**

1. Policy (DP & DP3) Training and Deployment.
2. Release of Benchmark Datasets and Checkpoints.
3. Scene augmentation using Robocasa.
4. Scene Generation.
5. More complex tasks.

