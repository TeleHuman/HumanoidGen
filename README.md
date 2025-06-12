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

1. **Clone the repository**

```sh
git clone git@github.com:TeleHuman/HumanoidGen.git
cd HumanoidGen
```

2. **Create & Build conda env**

```sh
conda create --name humanoidgen python=3.9 -y
conda activate humanoidgen
pip install -r requirements.txt
pip install -e .
```

3. **Install pytorch3d**

```sh
cd third_party/pytorch3d_simplified && pip install -e . && cd ../..
```

## 🚀 **Getting Started**

### 1. Change execution path

```sh
cd humanoidgen
```

### 2. Show scnce & Run task

In this project, we provide standard scenes and execution code for 20 tasks, which can be quickly run using the script files below.

```sh
bash scripts/run_scene.sh
bash scripts/run_solve.sh
```

In addition, you can specify main parameters either by directly modifying the sh script file or by using the commands below.

```sh
python process/run_scene.py -env block_handover -render False -video True
python process/run_solve.py -env block_handover -solve block_handover -render False -video True
```

In addition to specifying parameters on the command line, you can also run the corresponding Python file by setting the config file. For detailed configuration file parameters, see [**Configuration Instructions**](./CONFIGURATION.md)

### 2. Data collection & Visualization




