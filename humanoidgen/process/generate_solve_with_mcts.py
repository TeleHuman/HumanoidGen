from humanoidgen.mcts.rmax_tree_search import RMaxTS
from humanoidgen.mcts.utils import load_config, AttrDict
from humanoidgen import ROOT_PATH
import yaml

mcts_config_file = ROOT_PATH/"mcts/config.py"
mcts_cfg = load_config(str(mcts_config_file))

mcts_tree=RMaxTS(
    cfg=AttrDict({
        **mcts_cfg.sampler,
    })
)

with open(ROOT_PATH/"config/config_generate_solve_with_mcts.yml", "r") as file:
    config = yaml.safe_load(file)
mcts_tree.sample(config=config)