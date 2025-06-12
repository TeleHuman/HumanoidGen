# 🤖 HumanoidGen: Data Generation for Bimanual Dexterous Manipulation via LLM Reasoning

<div align="center">

[[Website]](https://humanoidgen.github.io/)
[[Arxiv(Coming Soon)]]()
[[Dataset(Coming Soon)]]()

<img src="./web/main_pipline.png"/>

</div>

## 1. Setup Environment

### Requirements

* Supported platform: Linux (Ubuntu 20.04)
* Python 3.9

### Installation

1. **Clone the repository**

```sh
git clone git@github.com:TeleHuman/HumanoidGen.git
cd HumanoidGen
```

2. **Create & Build Conda Env**

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

