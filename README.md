# 🤖 **HumanoidGen: Data Generation for Bimanual Dexterous Manipulation via LLM Reasoning**

<div align="center">
<p align="center">
<a href="https://github.com/jingzhi-git" target="_blank" style="color:black">Zhi Jing</a><sup>1,2</sup>,
<a href="#" target="_blank" style="color:black">Siyuan Yang</a><sup>1,3</sup>,
<a href="https://github.com/ProNeverFake" target="_blank" style="color:black">Jicong Ao</a><sup>1</sup>,
<a href="#" target="_blank" style="color:black">Ting Xiao</a><sup>4</sup>,
<a href="#" target="_blank" style="color:black">Yugang Jiang</a><sup>2</sup>,
<a href="https://baichenjia.github.io/" target="_blank" style="color:black">Chenjia Bai</a><sup>✉ 1</sup>
<br>
<sup>1</sup>Institute of Artificial Intelligence (TeleAI), China Telecom <sup>†</sup>
<sup>2</sup>Fudan University <sup>†</sup>
<br>
<sup>3</sup>University of Science and Technology of China
<sup>4</sup>East China University of Science and Technology
<br>
<sup>†</sup>Equally leading organizations
<sup>✉</sup> Corresponding Author

</p>

[🔥 Homepage](https://openhumanoidgen.github.io/)
[📄 Paper](https://arxiv.org/abs/2507.00833)
[⛁ Dataset](https://huggingface.co/datasets/TeleEmbodied/humanoidgen_dataset/tree/main/task_datasets)
[🤗 Model](https://huggingface.co/TeleEmbodied/humanoidgen_model/tree/main)

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

After installing the `mplib` library, change the parameter `n_init_qpos`from the default value of 20 to 50 in `mplib/planner.py`. To locate the file path, you can use the following command within the `humanoidgen` conda environment:

```sh
python -c "import mplib; print(mplib.planner.__file__)"
```

#### 3. **Install pytorch3d & dp & dp3**

Install pytorch3d:

```sh
cd third_party/pytorch3d_simplified && pip install -e . && cd ../..
```

Install dp3:

```sh
cd humanoidgen/policy/3D-Diffusion-Policy/3D-Diffusion-Policy && pip install -e . && cd ../../../..
```

Install dp:

```sh
cd humanoidgen/policy/Diffusion-Policy && pip install -e . && cd ../../..
```

#### 4. **Download Assets**

The assets are provided in our datasets lab [datasets](https://huggingface.co/datasets/TeleEmbodied/humanoidgen_dataset/tree/main/assets). Download the files `assets.zip` and `table_assets.zip`, extract them to the [humanoidgen](./) and [scene_builder/table](./scene_builder/table) directories respectively, and name both extracted folders as `assets`.

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

Additionally, you can specify main parameters by directly modifying the shell script files or by using the following commands:

```sh
python process/run_scene.py -env blocks_stack_easy -render False
python process/run_solve.py -env blocks_stack_easy -solve blocks_stack_easy -render False
```

To configure additional parameters, edit the config files [config_run_scene.yml](./config/config_run_scene.yml) and [config_run_solve.yml](humanoidgen/config/config_run_solve.yml).

### 3. **Collect & Visualize Data**

To collect data, open the configuration file [config_run_solve.yml](./humanoidgen/config/config_run_solve.yml) and set `record_data` to `true`. Then run the following command (example for ‘block_handover‘ task):

```sh
python process/run_solve.py -env block_handover -solve block_handover -render False
```

The datasets are generated in [datasets](./humanoidgen/datasets) folder and can be visualized using the following command:

```sh
python process/show_datasets.py
```

The visualization parameters are set in the configuration file [config\_show\_datasets.yml](./humanoidgen/config/config_show_datasets.yml).

### 4. **Train & Deploy Policy**

Firstly, pre-process the generated datasets for training policy.

```sh
python process/pkl2zarr.py
```

The dataset path, policy model, and additional parameters are set in the configuration file [config_pkl2zarr.yml](./config/config_pkl2zarr.yml).

For DP and DP3 policy training and evaluation, we also provide the corresponding the [datasets](https://huggingface.co/datasets/TeleEmbodied/humanoidgen_dataset/tree/main/task_datasets) and [models](https://huggingface.co/TeleEmbodied/humanoidgen_model/tree/main).

Dp3 policy train ([Configuration File Location](./policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config)):

```sh
bash scripts/train.sh dp3
```

Dp policy train ([Configuration File Location](./policy/Diffusion-Policy/diffusion_policy/config)):

```sh
bash scripts/train.sh dp
```

Dp3 policy eval ([Configuration File Location](./config/config_eval_dp3.yml)):

```sh
bash scripts/eval_dp3.sh
```

Dp policy eval ([Configuration File Location](./config/config_eval_dp.yml)):

```sh
bash scripts/eval_dp.sh
```

### 5. **Generate Task Execution Code**

This project supports two generation methods: direct generation and using MCTS. The execution commands are as follows:

```sh
bash scripts/run_generate_solve.sh block_handover 5
bash scripts/run_generate_solve_with_mcts.sh blocks_stack_hard_mcts 5
```

The first argument specifies the name of the generated task, and the second argument specifies the number of parallel threads to run. To interrupt the generation process, run:

```sh
bash scripts/kill_all_generate_processes.sh
```

## 📦 **Code to be released**

- Scene scaling with Robocasa
- Scene generation
- Generation of additional tasks (both MCTS and non-MCTS)

## 🔖 **Citation**

If you find our work helpful, please cite:

```bibtex
@article{jing2025humanoidgen,
  title={HumanoidGen: Data Generation for Bimanual Dexterous Manipulation via LLM Reasoning},
  author={Zhi Jing, Siyuan Yang, Jicong Ao, Ting Xiao, Yugang Jiang, Chenjia Bai},
  journal={arXiv preprint arXiv:2507.00833},
  year={2025}
}
```

## 📄 **License**

This codebase is under [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/deed.en). You may not use the material for commercial purposes, e.g., to make demos to advertise your commercial products.

## Acknowledgements

* [DeepSeek-Prover-V1.5](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): We referred to its MCTS module.
* [ManiSkill](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): Used as the simulation platform.
* [Unitree](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): We use the Unitree H1\_2 as our robot.
* [Gensim2](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): We referred to its constraints module.
* [RoboTwin](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html): We referred to its DP and DP3 modules.

## 📬 **Contact**

**Feel free to contact us!**

- Zhi Jing: [jingzhi2021@qq.com](mailto:jingzhi2021@qq.com) or WeChat `JZhi2024`
- Chenjia Bai (Corresponding Author): [baicj@chinatelecom.cn](mailto:baicj@chinatelecom.cn)

