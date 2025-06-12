from humanoidgen.mcts.utils import AttrDict
from humanoidgen.mcts.rmax_tree_search import RMaxTS

# algorithm
n_search_procs = 256
sampler = dict(
    gamma=0.99,
    sample_num=6400,
    concurrent_num=32,
    tactic_state_comment=True,
    ckpt_interval=128,
    log_interval=32,
)
