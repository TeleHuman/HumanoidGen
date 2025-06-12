import os
import gc
import time
import math
import random
import pickle
import subprocess

import numpy as np

# from prover.lean.proof import ProofSummarizer
from humanoidgen.mcts.utils import ConcurrentJob
import sys
import gymnasium as gym
import mani_skill.envs
import humanoidgen.envs
import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union
import numpy as np
import time
from enum import Enum
from mani_skill.utils.wrappers.record import RecordEpisode
from humanoidgen.motion_planning.h1_2.utils import images_to_video
from humanoidgen import ROOT_PATH
from datetime import datetime
from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
import yaml
from humanoidgen.envs.example.table_scene import TableSetting
from humanoidgen.llm.api_util import APIUTIL
from humanoidgen.mcts.solver import Solver
from humanoidgen.mcts.log import LogRecord



class TreeNode(object):
    def __init__(self, parent=None, code=None, **kwargs):
        self.parent = parent # root 父亲节点为none
        self.children = dict() #初始化时没有子节点
        self._info = {key: val for key, val in kwargs.items()}  # {'depth': 0,"state_info","node_id"}

        if code is not None:
            self.update_code(code)   # solve_code=str(), prohibited_code=list()
        
        if '_discounted_rewards' not in self._info:
            self._info['_discounted_rewards'] = 0.0
        if '_discounted_visitation' not in self._info:
            self._info['_discounted_visitation'] = 0.0
        if '_subtree_discounted_rewards' not in self._info:
            self._info['_subtree_discounted_rewards'] = 0.0
        if '_subtree_discounted_visitation' not in self._info:
            self._info['_subtree_discounted_visitation'] = 0.0

        self._num_running_jobs = 0
        self._subtree_num_running_jobs = 0
        self.value = None
        self.visitation = None
        self.subtree_value = None
        self.subtree_visitation = None
        self._update_value(gamma=0.0)  # gamma=0.0 is okay for initialization

    # basic tree supports
    @property
    def code(self):
        return random.choice(self._info['_code_list'])

    def update_code(self, code):
        if '_code_list' not in self._info:
            self._info['_code_list'] = []
        if code not in self._info['_code_list']:
            self._info['_code_list'].append(code)
    
    def __getitem__(self, key):
        return self._info[key]
    
    def to_node_list(self):
        return sum([child.to_node_list() for _, child in self.children.items()], start=[self])
    
    def to_dict(self):
        return dict(
            info=self._info,
            children={
                edge: child_node.to_dict()
                for edge, child_node in self.children.items()
            }
        )
    
    @classmethod
    def from_dict(cls, dict_data, parent=None):
        node = cls(
            parent=parent,
            **dict_data['info'],
        )
        node.children = {
            edge: cls.from_dict(child_dict, parent=node)
            for edge, child_dict in dict_data['children'].items()
        }
        return node
    
    # algorithm supports
    def update_reward(self, reward, gamma, first_node=True):
        print(f"\n📈 [Reward Update] Node {self.get_node_id()}")
        if first_node:
            self._info['_discounted_rewards'] = self._info['_discounted_rewards'] * gamma + reward
            self._info['_discounted_visitation'] = self._info['_discounted_visitation'] * gamma + 1.0
        self._info['_subtree_discounted_rewards'] = self._info['_subtree_discounted_rewards'] * gamma + reward
        self._info['_subtree_discounted_visitation'] = self._info['_subtree_discounted_visitation'] * gamma + 1.0
        self._update_value(gamma)

        if self.parent is not None:
            self.parent.update_reward(reward, gamma, first_node=False)
    
    def start_new_job(self, gamma, first_node=True):
        
        # 记录更新前的值
        orig_jobs = self._num_running_jobs
        orig_subtree_jobs = self._subtree_num_running_jobs

        if first_node:
            self._num_running_jobs += 1
        self._subtree_num_running_jobs += 1

        print(f"\n🆕 [Job Start] Node {self.get_node_id()}")
        print(f"   ├─ Active Jobs: {orig_jobs} → {self._num_running_jobs}" + (" (root)" if first_node else ""))
        print(f"   ├─ Subtree Jobs: {orig_subtree_jobs} → {self._subtree_num_running_jobs}")

        self._update_value(gamma)
        if self.parent is not None:
            self.parent.start_new_job(gamma, first_node=False)
    
    def complete_job(self, gamma, first_node=True):

        # 记录更新前的值
        orig_jobs = self._num_running_jobs
        orig_subtree_jobs = self._subtree_num_running_jobs

        if first_node:
            self._num_running_jobs -= 1
        self._subtree_num_running_jobs -= 1

        # 生成完成日志
        print(f"\n✅ [Job Complete] Node {self.get_node_id()}")
        print(f"   ├─ Active Jobs: {orig_jobs} → {self._num_running_jobs}" + (" (root)" if first_node else ""))
        print(f"   ├─ Subtree Jobs: {orig_subtree_jobs} → {self._subtree_num_running_jobs}")

        self._update_value(gamma)

        if self.parent is not None:
            self.parent.complete_job(gamma, first_node=False)
    
    def _update_value(self, gamma):
        
        # 在更新前记录原始值用于显示差异
        original_value = self.value
        original_visitation = self.visitation
        original_subtree_value = self.subtree_value
        original_subtree_visitation = self.subtree_visitation

        discounted_rewards = self._info['_discounted_rewards'] * (gamma ** self._num_running_jobs)   #0*(0.99**1)
        discounted_visitation = \
            self._info['_discounted_visitation'] * (gamma ** self._num_running_jobs) \
            + (1.0 - (gamma ** self._num_running_jobs)) / (1.0 - gamma)  
        self.value = discounted_rewards / max(discounted_visitation, 1e-2)
        self.visitation = discounted_visitation

        subtree_discounted_rewards = self._info['_subtree_discounted_rewards'] * (gamma ** self._subtree_num_running_jobs)
        subtree_discounted_visitation = \
            self._info['_subtree_discounted_visitation'] * (gamma ** self._subtree_num_running_jobs) \
            + (1.0 - (gamma ** self._subtree_num_running_jobs)) / (1.0 - gamma)
        self.subtree_value = subtree_discounted_rewards / max(subtree_discounted_visitation, 1e-2)
        self.subtree_visitation = subtree_discounted_visitation

        # 构建输出信息
        update_info = [
            f"📊 [Value Update] Node {self.get_node_id()}",
            f"   ├─ Parent: {self.parent.get_node_id() if self.parent else 'ROOT'}",
            f"   ├─ Value: {original_value} → {self.value}",
            f"   ├─ Visitation: {original_visitation} → {self.visitation}",
            f"   ├─ Subtree Value: {original_subtree_value} → {self.subtree_value}",
            f"   └─ Subtree Visits: {original_subtree_visitation} → {self.subtree_visitation}"
        ]
        print("\n".join([line for line in update_info]))

    # 获取当前节点的Asset State
    def get_asset_state_info(self):
        state_info = self._info["state_info"]
        asset_state_info = state_info.assets
        return asset_state_info
    
    # 获取当前节点的序号
    def get_node_id(self):
        return self._info["node_id"]

    def add_new_prohibited_code(self,prohibited_code):
        if prohibited_code[0] =="":
            return
        else:
            self._info['_code_list'][0]["prohibited_code"].append(prohibited_code)

class RMaxTS(object):
    
    def __init__(self,cfg):
        self.cfg = cfg
        self.gamma = self.cfg.get('gamma', 0.99)
        self.sample_num = self.cfg.get('sample_num', 6400)
        self.concurrent_num = self.cfg.get('concurrent_num', 1)
        self.tactic_state_comment = self.cfg.get('tactic_state_comment', True)

        # self.ckpt_filename = 'checkpoint.pkl'
        self.node_cls:TreeNode = TreeNode
        self.algorithm_pipeline = [
            self._tactic_tree_generate_proof,
            self._tactic_tree_parse_proof,
            self._rmax_exploration_summarize_results,
        ]
        # 新添加的参数
        self.node_num = 0
        self.solver_num = 0
        
    # basic supports
    def _save_ckpt(self, ckpt_dict: dict):
        # save a backup before overwriting the checkpoint file
        if os.path.exists(self.ckpt_path):
            subprocess.run(['cp', self.ckpt_path, self.ckpt_path + '.backup'])
        # overwrite the checkpoint file
        with open(self.ckpt_path, 'wb') as pkl_f:
            pickle.dump(ckpt_dict, pkl_f)
    
    # tree structure supports
    def _tree_setup(self):
        # initialize tree
        # ckpt_info = None
        # for _ckpt_path in [self.ckpt_path, self.ckpt_path + '.backup']:
        #     if os.path.exists(_ckpt_path):
        #         try:
        #             with open(_ckpt_path, 'rb') as pkl_f:
        #                 ckpt_info = pickle.load(pkl_f)
        #         except:
        #             self.process_print(f'Checkpoint saved at {_ckpt_path} is broken.')
        # if ckpt_info is not None:
        #     root = self.node_cls.from_dict(ckpt_info['root'])
        #     sample_count = ckpt_info['sample_count']
        #     yield_cache = ckpt_info['yield_cache']
        #     self.process_print(f'Load checkpoint from sample_count={sample_count}')
        # else:
        print("\n" + "═" * 50)
        print("[TREE INIT] Starting tree structure initialization".ljust(49) + "═")
        state_info = self.planner.get_state_info_now()
        print("Creating the root node...")
        root:TreeNode = self.node_cls(code=dict(solve_code=str(), prohibited_code=list()), depth=0,state_info=state_info,node_id=0) #创建根节点
        print(f"\n🆕 Tree initialized with root node ID: {self.node_num}") 
        self.node_num = 1  # 当前树节点数
        self.solver_num = 0  
        sample_count = 0
        yield_cache = []
        
        # compile the root node with `sorry`
        # self.proof_summarizer = ProofSummarizer(data=data, scheduler=self.scheduler)
        # root_sorry = self.proof_summarizer.analyze('  sorry', require_verification=True)
        # assert root_sorry.result['pass'], "Cannot parse a `sorry` tactic on root."
        # self.root_goal = root_sorry.result['sorries'][-1]['goal']
        # other initialization

        self._last_selected_node = root
        print("\n🔗 [Status] Root node linked to selection system")
        print("═" * 50 + "\n")

        return root, sample_count, yield_cache
    
    def _tree_new_child(self, parent,code, state_info, node_id):
        return self.node_cls(
            parent=parent,
            code=code,
            depth=parent['depth'] + 1,
            state_info=state_info,
            node_id=node_id
        )
    
    def judge_code_in_children(self, parent_node:TreeNode, code):
        print(f"parent_node: {parent_node.get_node_id()}")
        for edge, child_node in parent_node.children.items():
            print(f"child_node: {child_node.get_node_id()}")
            child_node_code = child_node.code['solve_code']
            if code['solve_code'].strip() == child_node_code.strip():
                return True
        return False

    def get_same_code_edge(self, parent_node:TreeNode, code):
        print(f"parent_node: {parent_node.get_node_id()}")
        for edge, child_node in parent_node.children.items():
            print(f"child_node: {child_node.get_node_id()}")
            child_node_code = child_node.code['solve_code']
            if code['solve_code'].strip() == child_node_code.strip():
                return edge

    def _tree_step(self, node:TreeNode, code,state_info):
        
        if self.judge_code_in_children(node, code):
            edge=self.get_same_code_edge(node, code)
            child_node:TreeNode = node.children[edge]
        else:
            parent_id = node.get_node_id()
            edge = f"{parent_id}_{self.node_num}"
            # if edge not in node.children:
            new_node = self._tree_new_child(node, code=code, state_info=state_info, node_id=self.node_num)
            node.children[edge] = new_node
            self.node_list.append(new_node)
            child_node:TreeNode = node.children[edge]
            # child_node.update_code(code)
            self.node_num += 1
        return child_node
    
    # 创建新节点
    def _tree_update(self, node:TreeNode, solver:Solver):
        node_walk, partial_code = node, str()

        solver.generate_exe_folder() # 生成执行代码的文件夹
        exe_result=solver.exe_code() # 将分布的代码执行多次

        for i in range(exe_result["add_milestone_num"]+1):
            
            if i == 0:
                print("\n" + "═" * 50)
                print(f"[PROHIBITION] Updating constraints for Node {node.get_node_id()}".ljust(49) + "═")
            else:
                print("\n" + "─" * 50)
                print(f"[NODE CREATION] Step {i}/{exe_result['add_milestone_num']}".ljust(49) + "─")

            if i==0:
                # if exe_result["add_node_num"]==0:
                prohibited_code = solver.get_milestone_prohibited_code(i)
                print(f"🛑 Adding primary prohibited code:")
                print(f"   ├─ Code : {prohibited_code}")
                print(f"   ├─ Parent Node: {node.get_node_id()}")
                node.add_new_prohibited_code(solver.get_milestone_prohibited_code(i))
                print(f"   └─ Total prohibitions: {len(node.code['prohibited_code'])}")
                continue

            print(f"🌱 Creating child node {self.node_num}:")
            print(f"   ├─ Parent: {node_walk.get_node_id()}")
            node_walk = self._tree_step(
                node=node_walk, 
                code=dict(solve_code=solver.get_milestone_solve_code(i), prohibited_code=[solver.get_milestone_prohibited_code(i)]),
                state_info=solver.get_milestone_new_state_info(i),  # 这里需要传入当前节点的状态
            )
            print(f"✅ Node created: {self.node_num-1}")
            print(f"   └─ Current tree size: {self.node_num} nodes")

        print("\n" + "═" * 50)
        print(f"[COMPLETED] Added {exe_result['add_milestone_num']} new nodes".ljust(49) + "═")
        print(f"   ├─ Final node counter: {self.node_num}")
        print(f"   └─ Current focus node: {node_walk.get_node_id()}")

        # 这里需要写一个给当前节点添加solver.get_prohibited_code的函数
        # prev_goal = self.root_goal
        # for info in segments:
        #     partial_code += info.tactic_code
        #     code = partial_code + info.state_comment if self.tactic_state_comment else partial_code
        #     if self._encode_length(code) < self.max_tokens:
        #         if info.goal != prev_goal:
        #             node_walk = self._tree_step(
        #                 node=node_walk, edge=info.goal,
        #                 code=dict(tactic_code=partial_code, state_comment=info.state_comment)
        #             )
        #             prev_goal = info.goal
        return node_walk

    def _select_node(self):
        print("\n" + "═" * 50)
        print("[NODE SELECTION] Starting Node Selection".ljust(49) + "═")
        node = self.root
        selection_depth = 0  # 跟踪选择深度
        
        while len(node.children) > 0:
            # 打印当前节点信息
            current_layer_str = f"[Selection Layer {selection_depth}] Node {node.get_node_id()} "
            current_layer_str += f"(V={node.visitation}, val={node.value:.2f}) "
            current_layer_str += f"TotalN={node.visitation + sum(c.subtree_visitation for c in node.children.values())}"
            print(current_layer_str)
            
            # 计算总访问次数和UCB系数
            total_visitation = node.visitation + np.sum([child.subtree_visitation for _, child in node.children.items()])
            log_total = max(total_visitation, 2)
            
            # 构建选项列表并打印
            choice_list = []

            # 遍历子节点
            for idx, (_, child) in enumerate(node.children.items(), 1):
                child_ucb = math.sqrt(2.0 * math.log(log_total) / max(child.subtree_visitation, 1e-2))
                child_total = child.subtree_value + child_ucb
                choice_list.append((child_total, child))
                print(f"   Child {idx}: {child.subtree_value:.2f} + {child_ucb:.3f} = {child_total:.3f}", end="")
                print(" ◀━" if idx == 1 else "")  # 仅第一个子节点标记起始符

            self_option = node.value + math.sqrt(2.0 * math.log(log_total) / max(node.visitation, 1e-2))
            choice_list.append((self_option, None))
            print(f"   UCB_SELF: {node.value:.2f} + {math.sqrt(2.0 * math.log(log_total)/max(node.visitation,1e-2)):.3f} = {self_option:.3f}")
            print("   └━")  # 结束符
            # 排序并选择最佳选项
            choice_list.sort(reverse=True, key=lambda x: x[0])
            best_choice = choice_list[0]
            
            # 打印选择结果
            arrow = "✔" if best_choice[1] else "✖"
            print(f"=> Selected: {best_choice[0]:.3f} {arrow}")
            
            if best_choice[1] is None:
                node.start_new_job(gamma=self.gamma)
                print("========== Node Selection Process Completed ==========")
                break
            else:
                node = best_choice[1]
                selection_depth += 1
                print(f"━━━┓\n┃ Proceeding to child node {node.get_node_id()}\n━━━┛")
        
        # 最终扩展输出
        print("\n" + "═"*40)
        print(f"[Final Expansion] Node {node.get_node_id()} (depth {selection_depth})")
        print(f"Trigger new job with gamma={self.gamma}")
        print("═"*40 + "\n")
        node.start_new_job(gamma=self.gamma)
        print(f"SELECTED NODE ID: {node.get_node_id()}")
        print("\n" + "═" * 50)
        return node

    def _tactic_tree_generate_proof(self, node:TreeNode):
        print("\n" + "═" * 50)
        print(f"[Code GEN] Start generating for Node {node.get_node_id()}".ljust(49) + "═")
        code_prefix = node.code
        # extra_prompt = code_prefix['solve_code']
        # if self.tactic_state_comment:
        #   extra_prompt += code_prefix['state_comment']
        # 这里需要使用node_info生成prompt，并将场景复现为node的状态
        ## 场景基本信息、初始场景状态、当前场景状态、已经执行的代码、不可以执行的代码
        # 准备生成prompt的信息(任务目标,初始状态，当前状态，已经执行的代码,不可以执行的代码),除此之外物品的基本属性信息额外获取
        task_goal = self.config["task_goal"] # 任务目标 
        task_note = self.config["task_note"] # 任务备注
        asset_attribute = self.env.get_asset_attribute() # 物品的基本属性信息
        initial_asset_state=self.root.get_asset_state_info() # 获取root节点的Asset State
        current_asset_state=node.get_asset_state_info() # 获取当前节点的Asset State
        executed_code = code_prefix['solve_code'] # 已经执行的代码
        children_nodes = node.children

        prohibited_codes = []
        # 添加子结点的作为不能扩展动作
        # for edge, children_node in children_nodes.items():
        #     prohibited_code = Solver.get_latest_action_code_from_prompt(children_node.code['solve_code'])
        #     reason = "Please use another method."
        #     prohibited_codes.append([prohibited_code,reason])
        for prohibited_code in code_prefix["prohibited_code"]:
            prohibited_codes.append(prohibited_code)

        prohibited_code = Solver.generate_prohibited_prompt(prohibited_codes) # 不可以执行下一步代码
        node_id = node.get_node_id() # 获取当前节点的序号
        llm_model = APIUTIL()

        print("\n📊 [States] Asset State Comparison")
        print(f"   ├─ Current State: {current_asset_state}")
        print("\n📝 [Code Tracking]")
        print(f"   ├─ Executed Code:\n {executed_code}")
        print(f"\n🚫 [Code Tracking]  \n├─ Prohibited Code:\n {prohibited_code}")

        unexecuted_code , _ = llm_model.generate_solve_mcts(task_goal,asset_attribute,initial_asset_state,current_asset_state,executed_code,prohibited_code,self.env_name,node_id,task_note)
        return dict(
            node=node,
            unexecuted_code=unexecuted_code,
            # generator_request_id=self.scheduler.generator_submit_request(
            #     self._preprocess_data({**data, '_extra_prompt': extra_prompt}),
            # ),
            # generator_request_id="none"
        )
    
    def _tactic_tree_parse_proof(self, node:TreeNode, unexecuted_code):
        # code = None
        # code = self.scheduler.generator_get_request_status(generator_request_id)
        # if code is None:
        #     return None
        # 将prompt传给llm生成执行代码，并提取其中的代码部分进行合并
        # code = "test"
        # code = code_prefix['tactic_code'] + code
        # proof = self.proof_summarizer.analyze(code, require_verification=True)
        # proof ="test"
        executed_code = node.code['solve_code']
        solver=Solver(unexecuted_code,executed_code,self.env,node.get_node_id(),self.solver_num,self.solver_folder,self.config)
        self.solver_num += 1
        return dict(node=node, solver=solver)
    
    def _rmax_exploration_summarize_results(self, node:TreeNode, solver:Solver):
        # if not proof.is_result_ready():
        #     return None        
        num_nodes_before = len(self.node_list)
        self._tree_update(node,solver)
        # RMax reward
        # node.update_reward(solver.check_new_milestone(), gamma=self.gamma) # 是否达到新的里程碑
        node.update_reward(num_nodes_before< len(self.node_list), gamma=self.gamma) # 是否达到新的里程碑
        node.complete_job(gamma=self.gamma)
        
        return dict(
            code=solver.generate_success_python(),
            result=solver.check_success(),
        )

    def create_env(self):
        config_file = ROOT_PATH / "config/config_generate_solve_with_mcts.yml"
        with open(config_file, "r") as file:
            run_config = yaml.safe_load(file)

        @dataclass
        class Args:
            env_id: Annotated[Optional[str], tyro.conf.arg(aliases=["-e"])] = run_config["env_id"]
        
        args = tyro.cli(Args)
        print(f"Running environment: {args.env_id}")
        
        run_config.update(vars(args))
        
        if run_config["render_mode"] == "auto":
            if run_config["default"]["render_scene"]:
                run_config["render_mode"]="human"
            else:
                run_config["render_mode"]="rgb_array"

        env_kwargs = dict(
            obs_mode=run_config["obs_mode"],
            reward_mode=run_config["reward_mode"],
            control_mode=run_config["control_mode"],
            render_mode=run_config["render_mode"],
            sensor_configs=dict(shader_pack=run_config["shader"]),
            human_render_camera_configs=dict(shader_pack=run_config["shader"]),
            viewer_camera_configs=dict(shader_pack=run_config["shader"]),
            num_envs=run_config["num_envs"],
            sim_backend=run_config["sim_backend"],
            enable_shadow=run_config["enable_shadow"],
            parallel_in_single_scene=run_config["parallel_in_single_scene"],
        )

        extra_kwargs = dict(
            config_file_path=config_file
        )
        env_kwargs.update(extra_kwargs)
        env = gym.make(args.env_id,**env_kwargs)
        env=env.unwrapped
        return env

    # sampler interface   config运行参数
    def sample(self, config):
        self.env_name = config["env_id"]
        
        mcts_file_name = ROOT_PATH/f"mcts/task_info/{self.env_name}_mcts.yml"
        with open(mcts_file_name, "r") as file:
            mcts_config = yaml.safe_load(file)
        self.config = {**config, **mcts_config}   
        self.env = self.create_env() # 
        self.planner = HumanoidMotionPlanner(
            self.env,
            debug=False,
            vis=True,
            base_pose=self.env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=True,
            print_env_info=False,
            show_key_points=self.config["show_key_points"],
            debug_key_frame=self.config["debug_key_frame"],
        )
        
        test_num = config["exe_num"]
        success_num = 0
        saple_num = []
        for i in range(test_num):
            current_test = i + 1
            print(f"\n{'='*30} TEST {current_test}/{test_num} STARTED {'='*30}")

            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.solver_folder = str(ROOT_PATH/"motion_planning/h1_2/solution/mcts"/f"{self.env_name}_{current_time}")  #存储生成action代码的文件夹
            self.terminal_log_file = os.path.join(self.solver_folder, "terminal_log.txt")
            # LogRecord.update_log_file_path(self.terminal_log_file)
            sys.stdout=LogRecord(self.terminal_log_file)

            # self.env.reset(seed=0)
            self.env.init_task_scene() # 初始化场景，重置(reset)+判断是否初始化成功
            self.root, sample_count, yield_cache = self._tree_setup() #初始化树结构
            
            self.node_list = self.root.to_node_list()
            # for _proposal, _sample_info in yield_cache:
            #     yield _proposal, _sample_info
            gc.collect()  # release memory

            job_slots = [
                ConcurrentJob(self.algorithm_pipeline)
                for _ in range(self.concurrent_num)
            ]

            sample_budget = self.sample_num - sample_count if len(yield_cache) == 0 else 0
            sample_budget = config["sample_num"]
            while (sample_budget > 0) or any([not job.is_idle() for job in job_slots]):
                # print("aaa")
                execute_sccess = False
                for job in job_slots:
                    if job.is_idle() and sample_budget > 0:
                        node = self._select_node() # 0-> start_new_job
                        self._last_selected_node = node
                        job.start(node=node)
                        sample_budget -= 1
                    if not job.is_idle():
                        info = job.get_status()
                        if info is not None:
                            # output samples
                            sample_count += 1
                            if info['result']:
                                file_name = os.path.join(self.solver_folder, f"step0.py")
                                code_content = info['code']
                                
                                # 生成输出信息
                                print("\n" + "═" * 50)
                                print(f"[SAMPLE GENERATION] Creating initial solution".ljust(49) + "═")
                                print(f"📝 Code Summary:")
                                print(f"   ├─ Sample Number: #{sample_count}")
                                print(f"   ├─ File Name: {os.path.basename(file_name)}")
                                print(f"   ├─ Code Length: {len(code_content)} chars")
                                
                                # 显示代码片段预览
                                print("\n🔍 Code Preview:")
                                preview_lines = code_content.split('\n')[:3]
                                for idx, line in enumerate(preview_lines, 1):
                                    print(f"   {idx:02d} │ {line[:60].strip()}")
                                if len(code_content.split('\n')) > 3:
                                    print("      ...")

                                with open(file_name, "w") as file:
                                    file.write(info['code'])
                                # return [True,sample_count]
                                # saple_num.append(sample_count)
                                success_num += 1
                                execute_sccess = True
                                break


                                # _proposal, _sample_info = info['code'], self._post_sample_info(
                                #     cost=sample_count, tree_size=len(self.node_list),
                                # )
                                # yield_cache.append((_proposal, _sample_info))
                                # yield _proposal, _sample_info
                            
                            # # logging
                            # if sample_count % self.log_interval == 0:
                            #     self.process_print('Progress: {} / {}    Tree Size: {}'.format(
                            #         sample_count, self.sample_num, len(self.node_list),
                            #     ))
                            # # saving checkpoints
                            # if sample_count % self.ckpt_interval == 0:
                            #     self._save_ckpt(dict(
                            #         root=self.root.to_dict(),
                            #         sample_count=sample_count,
                            #         yield_cache=yield_cache,
                            #     ))
                            #     if len(yield_cache) > 0:
                            #         # return after saving the checkpoint
                            #         # avoid overestimation caused by interrupt-restart loop
                            #         sample_budget = 0
                if execute_sccess == True:
                    print(f"\n✅ Test {current_test} Succeeded (#{success_num})")
                    break
                elif sample_budget == 0:
                    print(f"\n❌ Test {current_test} Failed (#{success_num})")
                    break

                time.sleep(0.1)
        # return [False, sample_count]
            saple_num.append(sample_count)
        print(f"success_num: {success_num}")
        print(f"success_rate: {success_num/test_num}")
        print(f"sample_num: {saple_num}")
        print(f"average_sample_num: {sum(saple_num)/test_num}")

        # # save the final tree structure
        # self._save_ckpt(dict(
        #     root=self.root.to_dict(),
        #     sample_count=sample_count,
        #     yield_cache=yield_cache,
        # ))
