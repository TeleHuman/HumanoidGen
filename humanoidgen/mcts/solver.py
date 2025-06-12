import re
from humanoidgen import ROOT_PATH
import os
from datetime import datetime
from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
import importlib.util


class Solver(object):
    def __init__(self, code,executed_code,env,node_id,solver_id,solver_folder_path,config):
        self.functions = ["open_hand", "hand_pre_grasp", "hand_grasp", "hand_pre_pinch", "hand_pinch", "move_to_pose_with_screw"]

        self.logical_error = False

        self.code_list = code
        self.merged_code = self.merge(code)
        self.segmented_code = self.segmentation(self.merged_code,self.functions)
        self.code_prompt = self.prompt_code(self.segmented_code)
        
        self.merged_executed_code = self.merge_prompt_code(executed_code)
        self.segmented_executed_code = self.segmentation(self.merged_executed_code,self.functions)
        self.executed_code_prompt = self.prompt_code(self.segmented_executed_code)

        self.segmented_code_all = self.segmented_executed_code + self.segmented_code
        self.step_indices = self.get_function_indices(self.segmented_code_all, self.functions)
        
        self.milestone_code_block_executed,self.milestone_index_executed = self.milestone_merge(self.segmented_executed_code,self.step_indices[:len(self.segmented_executed_code)])
        self.milestone_code_block,self.milestone_index = self.milestone_merge(self.segmented_code,self.step_indices[len(self.segmented_executed_code):])

        self.solver_folder_path = solver_folder_path
        self.env = env
        self.node_id = node_id
        self.solver_id = solver_id
        self.env_name = env.env_name
        self.config = config

    def merge(self, code):
        merged_code = "".join(code)
        return merged_code
    
    def merge_prompt_code(self, code):
        match = re.findall(r'```python\s*(.*?)\s*```', code, re.DOTALL)
        return self.merge(match)
    
    def prompt_code(self, segmented_code):
        code_prompt =[]
        for i in range(len(segmented_code)):
            code_prompt.append("\n```python\n"+segmented_code[i]+"\n```\n")
        return self.merge(code_prompt)
    
    def segmentation(self,merged_code, functions):
        """
        将合并后的代码字符串划分为多个 step,每个 step 保留其他内容，但必须包含且仅包含一个指定函数调用。

        Args:
            merged_code (str): 合并后的代码字符串。
            functions (list): 指定的函数名称列表，用于划分 step。

        Returns:
            list: 分步后的代码列表，每步为一个字符串。
        """

        import re

        # 构建正则表达式，匹配指定函数的调用
        pattern = r'(' + r'|'.join(re.escape(func) + r'\s*\(.*?\)' for func in functions) + r')'

        # 使用正则表达式查找所有指定函数的调用
        matches = list(re.finditer(pattern, merged_code, re.DOTALL))

        steps = []
        start_idx = 0

        # 遍历每个匹配项，将代码划分为 step
        for match in matches:
            func_call = match.group(0)  # 当前匹配的函数调用
            end_idx = match.end()       # 当前函数调用的结束位置

            # 提取从上一个函数结束到当前函数调用结束的代码段
            step_code = merged_code[start_idx:end_idx]

            # 确保 step 中只有一个函数调用
            step_code = re.sub(pattern, func_call, step_code, count=1)

            steps.append(step_code.strip())
            start_idx = end_idx
        
        # 添加最后一段代码（如果有剩余）
        if start_idx < len(merged_code):
            remaining_code = merged_code[start_idx:].strip()
            if remaining_code:
                steps.append(remaining_code)
        indentation_needed = False
        # for i in range(len(steps)):
        #     steps[i] = "\n    "+steps[i]+"\n"
            
        for i in range(len(steps)):
            lines=steps[i].splitlines()
            if any(line.strip() != "" and not line.startswith(" ") for line in lines[1:]):
                # steps[i] = "\n    " + "\n    ".join(lines)
                steps[i] = "\n    ".join(lines)
                indentation_needed = True
                # break
        
        for i in range(len(steps)):
            steps[i] = "\n    "+steps[i]+"\n"
        return steps

    def generate_exe_folder(self):
        if os.path.exists(self.solver_folder_path) and os.path.isdir(self.solver_folder_path):
            pass
        else:
            os.makedirs(self.solver_folder_path, exist_ok=True)
        folder_path =str(self.solver_folder_path+f"/{self.solver_id}_{self.node_id}")
        self.folder_path = str(folder_path)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # 如果文件夹存在，添加时间戳(按照道理不会出现)
            print(f"Error: The folder '{folder_path}' exists.")
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_path = str(ROOT_PATH/"motion_planning/h1_2/solution/mcts"/f"{self.solver_id}_{self.node_id}_{current_time}")
            print(f"Creating a new folder '{folder_path}' with a timestamp.")
            os.makedirs(folder_path, exist_ok=True)
        else:
            print(f"The folder '{folder_path}' does not exist.")
            os.makedirs(folder_path, exist_ok=True)

        file_prefix = "from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *\ndef step(planner:HumanoidMotionPlanner):\n"
        for i, code_block in enumerate(self.segmented_code_all):
            file_name = os.path.join(folder_path, f"step{i}.py")
            with open(file_name, "w") as file:
                file.write(file_prefix)  # 写入前缀
                file.write(code_block)  # 写入代码块
            print(f"Saved: {file_name}")

    def init_planner(self,env):
        seed = 0
        env.reset(seed=seed)
        init_scene_step =40
        for i in range(init_scene_step):
            env.render()
            defalt_pose = env.agent.robot.get_qpos()[0, :38].cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(defalt_pose)
        planner = HumanoidMotionPlanner(
            env,
            debug=False,
            vis=True,
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=True,
            print_env_info=False,
            show_key_points=self.config["show_key_points"],
            debug_key_frame=self.config["debug_key_frame"],
        )
        return planner

    # def run_step(self,task_filder,file_name,planner):
    #     spec = importlib.util.spec_from_file_location(file_name.removesuffix(".py"), str(task_filder+"/"+file_name))
    #     module = importlib.util.module_from_spec(spec)
    #     spec.loader.exec_module(module)
    #     try:
    #         # 尝试执行 module.step(planner)
    #         module.step(planner)
    #         return ""
    #     except Exception as e:
    #         import traceback
    #         # 捕获异常并打印错误信息
    #         print(f"Error occurred while executing step in {file_name}: {e}")
    #         error_traceback = traceback.format_exc()  # 获取完整的错误堆栈信息
    #         print(error_traceback)  # 打印完整的错误堆栈信息
    #         return error_traceback  # 返回完整的错误信息

    def run_step(self, task_folder, file_name, planner):
        try:
            # 将模块加载过程也移到 try 块内
            spec = importlib.util.spec_from_file_location(
                file_name.removesuffix(".py"),
                str(task_folder + "/" + file_name)
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # 可能抛出语法错误的位置
            
            module.step(planner)  # 执行用户代码
            return ""
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            print(f"Error in {file_name}:\n{error_traceback}")
            return error_traceback

    def get_function_indices(self,segmented_code_all, functions):
        """
        查询 segmented_code_all 中每段代码使用的 functions 的顺序。

        Args:
            segmented_code_all (list): 包含代码段的列表。
            functions (list): 函数名称列表。

        Returns:
            list: 每段代码对应的函数在 functions 中的索引（从 1 开始）。
        """
        indices = []
        for code in segmented_code_all:
            for i, func in enumerate(functions, start=0):  # 从 1 开始索引
                if func in code:
                    indices.append(i)
                    break  # 每段代码只包含一个函数，找到后立即跳出循环
        return indices

    # 
    def exe_code(self):
        run_step_nums = [] # 记录每次执行的步数
        success_executions = [] # 记录每次执行的成功与否
        fail_reasons = [] # 记录每次执行的失败原因
        state_infos = []
        
        print("\n" + "═" * 50)
        print(f"[EXECUTION] Starting code validation".ljust(49) + "═")
        

        print("step_indices:", self.step_indices)
        for trial in range(5):
            
            print(f"\n🔁 Trial #{trial+1}")
            print("   ├─ Initializing environment...")

            planner = self.init_planner(self.env)
            while True:
                start_task_flag=self.env.start_task()
                if start_task_flag == True:
                    break
                else:
                    planner=self.init_planner(self.env)
            
            print(f"   ├─ Total steps to execute: {len(self.segmented_code_all)}")
            print("   └─ Execution progress:")

            run_step_num = 0
            fail_reason = ""
            state_info = []
            # 三种情况: 未执行完成发生错误；执行完成, 但是任务没有成功；执行完成并且任务成功
            for step in range(len(self.segmented_code_all)):
                
                step_status = f"      Step {step+1}: "

                error_info = self.run_step(self.folder_path,f"step{step}.py",planner)
                state_info.append(planner.get_state_info_now())
                if error_info != "":
                    fail_reason = error_info
                    
                    step_status += "❌ Runtime Error - "
                    step_status += f"{error_info}..."
                    print(step_status)
                    break
                
                step_result = planner.execution_result[-1]
                if step_result[1] == False or step_result[1] == 0:
                    fail_reason = step_result[2]

                    step_status += "❌ Validation Failed - "
                    fail_detail = step_result[2][:50] + "..." if len(step_result[2]) > 50 else step_result[2]
                    step_status += fail_detail
                    print(step_status)
                    break
                
                run_step_num += 1
                if self.env.check_success():
                    break
            
            planner.end_planner()
            self.env.end_task(save_file=f"generate_mcts/{self.env_name}")
            
            success = self.env.check_success()
            status_icon = "✅" if success else "❌"
            print(f"\n   {status_icon} Trial Result: {'Success' if success else 'Failed'}")
            print(f"      ├─ Steps completed: {run_step_num}/{len(self.segmented_code_all)}")
            print(f"      └─ Failure reason: {fail_reason[:70]}..." if fail_reason else "      └─ No errors detected")

            state_infos.append(state_info)
            run_step_nums.append(run_step_num)
            fail_reasons.append(fail_reason)
            success_executions.append(success)

        print("\n" + "═" * 50)
        print("[EXECUTION] Final Validation Report".ljust(49) + "═")
        for i, (success, steps) in enumerate(zip(success_executions, run_step_nums)):
            status = "✅ FULL SUCCESS" if success else f"❌ PARTIAL ({steps} steps)"
            print(f"   Trial {i+1}: {status}")



        # 检查是否有成功的执行
        for i, success in enumerate(success_executions):
            if success:
                self.state_infos = state_infos[i]
                self.new_state_info = state_infos[i][len(self.segmented_executed_code):]
                self.add_node_num = run_step_nums[i]-len(self.segmented_executed_code)
                self.run_step_num= run_step_nums[i]
                self.success = success
                self.fail_reason = ""

                milestone_num = 0
                milestone_step=0
                for j in range(len(self.milestone_index)):
                    if type(self.milestone_index[j]) == list:
                        milestone_step += len(self.milestone_index[j])
                    else:
                        milestone_step += 1
                    milestone_num += 1
                    if milestone_step >= self.add_node_num:
                        break
                self.add_milestone_num = milestone_num
                return {
                    "add_milestone_num": milestone_num,
                    "state_infos": state_infos[i],
                    "new_state_info": state_infos[i][len(self.segmented_executed_code):],
                    "add_node_num": run_step_nums[i]-len(self.segmented_executed_code),
                    "run_step_num": run_step_nums[i],
                    "success": success,
                    "fail_reason": ""
                }

        # 如果没有成功的执行，找到最大的 run_step_nums
        max_steps_index = run_step_nums.index(max(run_step_nums))
        self.state_infos = state_infos[max_steps_index][:run_step_nums[max_steps_index]]
        if len(self.segmented_executed_code) == run_step_nums[max_steps_index]:
            self.new_state_info = None
        else:
            self.new_state_info = state_infos[max_steps_index][len(self.segmented_executed_code):run_step_nums[max_steps_index]]
        self.add_node_num = run_step_nums[max_steps_index]-len(self.segmented_executed_code)
        self.run_step_num = run_step_nums[max_steps_index]
        self.success = success_executions[max_steps_index]
        self.fail_reason = fail_reasons[max_steps_index]

        milestone_num = 0
        milestone_step=0
        for j in range(len(self.milestone_index)):
            if type(self.milestone_index[j]) == list:
                milestone_step += len(self.milestone_index[j])
            else:
                milestone_step += 1
            if milestone_step > self.add_node_num:
                break
            elif milestone_step == self.add_node_num:
                milestone_num += 1
                break
            milestone_num += 1
        self.add_milestone_num = milestone_num

        return {
            "add_milestone_num": milestone_num,
            "state_infos": self.state_infos,
            "new_state_info": self.new_state_info,
            "add_node_num": run_step_nums[max_steps_index]-len(self.segmented_executed_code),
            "run_step_num": run_step_nums[max_steps_index],
            "success": success_executions[max_steps_index],
            "fail_reason": fail_reasons[max_steps_index]
        }
    
    # 
    def get_solve_code(self,step_num):
        merged_code = self.segmented_executed_code + self.segmented_code[:step_num]
        return self.prompt_code(merged_code)
    
    def get_milestone_solve_code(self,step_num):
        merged_code = self.milestone_code_block_executed + self.milestone_code_block[:step_num]
        return self.prompt_code(merged_code)
    
    def get_prohibited_code(self,step_num):
        # 如果当前步数小于添加节点的步数，或者当前步数等于生成代码段的长度，则返回空字符串
        if step_num < self.add_node_num or step_num == len(self.segmented_code):
            # [code,reason]
            return ["",""] 
        prohibited_code = self.segmented_code[step_num]
        reason = self.fail_reason
        return [prohibited_code,reason]

    def get_milestone_prohibited_code(self,step_num):
        # 如果当前步数小于添加节点的步数，或者当前步数等于生成代码段的长度，则返回空字符串
        if step_num < self.add_milestone_num or step_num == len(self.milestone_code_block):
            # [code,reason]
            return ["",""] 
        prohibited_code = self.milestone_code_block[step_num]
        reason = self.fail_reason
        return [prohibited_code,reason]


    def check_success(self):
        return self.success
    
    def generate_success_python(self):
        merged_code = self.segmented_executed_code + self.segmented_code[:self.add_node_num]
        file_prefix = "from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *\ndef step(planner:HumanoidMotionPlanner):\n"
        success_code = "" 
        success_code += file_prefix
        for i, code_block in enumerate(merged_code):
            success_code += code_block
        return success_code
    
    def check_new_milestone(self):
        target_indices = {0, 2, 4} 
        new_node_indices=self.step_indices[len(self.segmented_executed_code):self.run_step_num]
        new_milestone=bool(any(index in target_indices for index in new_node_indices))
        
        if new_milestone:
            return True
        else:
            return False
        
    @classmethod
    def generate_prohibited_prompt(cls,prohibited_code):
        prohibited_prompt = ""
        for i in range(len(prohibited_code)):
            if prohibited_code[i][0] =="":
                continue
            prohibited_prompt += "\n```python\n"+prohibited_code[i][0]+"\n```\n"
            prohibited_prompt += "reason:\n"+prohibited_code[i][1]+"\n"
        return prohibited_prompt
    
    @classmethod
    def get_latest_action_code_from_prompt(cls,prompt_code):
        match = re.findall(r'```python\s*(.*?)\s*```', prompt_code, re.DOTALL)
        return match[-1]


    def get_new_state_info(self,step_num):
        return self.new_state_info[step_num-1]
    
    def get_milestone_new_state_info(self,milestone_num):
        milestone_num = milestone_num-1
        
        milestone_step=0
        for j in range(milestone_num):
            if type(self.milestone_index[j]) == list:
                milestone_step += len(self.milestone_index[j])
            else:
                milestone_step += 1
        if milestone_step >= len(self.new_state_info):
            if self.new_state_info:  # 检查 self.new_state_info 是否不为空
                return self.new_state_info[-1]
            elif self.state_infos:  # 如果 self.new_state_info 为空，则检查 self.state_info
                return self.state_infos[-1]
            else:
                return ""
        return self.new_state_info[milestone_step]


    def milestone_merge(self,segmented_code, step_indices):
        milestone_code = []
        milestone_indexes = []
        p = 0
        while p < len(step_indices):
            milestone_index = []
            # open_hand
            if step_indices[p] == 0:
                # 将 move+openhand合并
                while  (len(milestone_indexes))>0 and milestone_indexes[-1] == 5:
                    milestone_indexes.pop()
                    milestone_index.append(5)
                milestone_index.append(0)
                milestone_indexes.append(milestone_index)
                p += 1
                continue
            # hand_grasp
            elif step_indices[p] == 2:
                # 将 pre-grasp+move+grasp合并
                while (len(milestone_indexes))>0 and (milestone_indexes[-1] == 5 or milestone_indexes[-1] == 1):
                    milestone_index.append(milestone_indexes.pop())
                milestone_index.append(2)
                milestone_indexes.append(milestone_index)
                p += 1
                continue
            # hand_pinch
            elif step_indices[p] == 4:
                # 将 pre-pinch+move+pinch合并
                while  (len(milestone_indexes))>0 and (milestone_indexes[-1] == 5 or milestone_indexes[-1] == 3):
                    milestone_index.append(milestone_indexes.pop())
                milestone_index.append(4)
                milestone_indexes.append(milestone_index)
                p += 1
                continue
            else:
                milestone_indexes.append(step_indices[p])
                p += 1

            # # hand_pre_grasp
            # if step_indices[p] == 1:
            #     q=p+1
            #     while q < len(segmented_code):
            #         if step_indices[q] != 2 and step_indices[q] != 5:
            #             p+=1  #出现异常忽略pre操作
            #             break
            #         elif step_indices[q] == 2:
            #             milestone_index.append(2)
            #             milestone_indexes.append(milestone_index)
            #         elif  step_indices[q] == 5:
            #             milestone_index.append(5)
            #         milestone_index.append(1)
            #         q += 1
        
        code_p=0
        for i in range(len(milestone_indexes)):
            if type(milestone_indexes[i]) == list:
                milestone_code.append(self.merge(segmented_code[code_p:code_p+len(milestone_indexes[i])]))
                code_p += len(milestone_indexes[i])
            else:
                milestone_code.append(segmented_code[code_p])
                code_p += 1
        return milestone_code,milestone_indexes