from openai import OpenAI
from humanoidgen import ROOT_PATH
import yaml
import humanoidgen.llm.utils as g_utils
from humanoidgen.envs.repository.library import Library
import random
from datetime import datetime
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
import json
import re
import os
from openai import AzureOpenAI
import io
import base64
import numpy as np
from PIL import Image

class APIUTIL:
    def __init__(self):
        self.prompt_folder= ROOT_PATH / "llm/prompt"
        self.llm_config_file=ROOT_PATH / "config/config_llm.yml"

        # Read the configuration file
        with open(self.llm_config_file, "r") as file:
            self.cfg = yaml.safe_load(file)

        self.model=self.cfg["model_name"]
        self.api_key = self.cfg[self.model]["api_key"]
        self.multi_model=self.cfg[self.model]["multi_model"]
        self.reasoning= self.cfg[self.model]["reasoning"]
        self.reply_max_tokens = self.cfg[self.model]["reply_max_tokens"]
        self.retry_max = self.cfg[self.model]["retry_max"]

        self.prompt_hitory = []
        self.chat_log = "" # chat log file, to be saved
        self.library = Library()
        self.save_token_consumed = True
        self.use_history=False

        if self.model == "deepseek-reasoner" or self.model == "deepseek-chat":
            self.client = OpenAI(api_key=self.api_key, base_url=self.cfg[self.model]["base_url"])
        
        elif self.model == "gpt-4o" :
            self.client = AzureOpenAI(
                api_key = self.api_key,
                api_version = self.cfg[self.model]["api_version"],
                azure_endpoint = self.cfg[self.model]["azure_endpoint"],
            )

    def call_llm(self, model, prompt,interaction_txt):
        # self.prompt_hitory=[]
        if not self.use_history:
            self.prompt_hitory=[]
        self.prompt_hitory.append({"role": "user", "content": prompt})
        truncated_messages = g_utils.truncate_message_for_token_limit(
            self.prompt_hitory, self.reply_max_tokens
        )
        for retry in range(self.retry_max):
            try:
                if interaction_txt is not None:
                    interaction_txt=g_utils.add_to_txt(interaction_txt, ">>> Prompt: \n" + prompt, with_print=False)
                print("Task prompt text:",truncated_messages)
                print("calling LLM..........")
                # Call the LLM
                # call_res = self.client.chat.completions.create(
                #     model=model,
                #     messages=truncated_messages
                # )
                print("model(call_llm):",model)
                response = self.client.chat.completions.create(
                    model=model,
                    messages=truncated_messages,
                    stream=False,
                    temperature=0
                )
                res = response.choices[0].message.content

                if self.reasoning:
                    resoning_content=response.choices[0].message.reasoning_content
                else:
                    resoning_content=""
                print("call LLM success!")
            
                with open(str(ROOT_PATH/"Tokens_consumption.txt"), "a") as file:
                    # with open("output.txt", "w") as file:
                    # file.write(f"========>{time.time()}\n")
                    time_now=datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                    file.write(f"========>{time_now}\n")
                    file.write(f"Model: {self.model}\n")
                    # file.write(f"Function: generate_feedback()\n")
                    completion_tokens = response.usage.completion_tokens
                    # file.write(f"Completion Tokens: {completion_tokens}\n")
                    file.write(f"Completion Tokens: {completion_tokens}\n")
                    prompt_tokens = response.usage.prompt_tokens
                    file.write(f"Prompt Tokens: {prompt_tokens}\n")
                    total_tokens = response.usage.total_tokens
                    file.write(f"Total Tokens: {total_tokens}\n")
                

                self.prompt_hitory.append({"role": "assistant", "content": res})
                to_print = highlight(f"{res}", PythonLexer(), TerminalFormatter())
                print(to_print)
                if interaction_txt is not None:
                    interaction_txt=g_utils.add_to_txt(interaction_txt, f"========>{time_now}\n", with_print=False)
                    interaction_txt=g_utils.add_to_txt(interaction_txt, f"Model: {self.model}\n", with_print=False)
                    interaction_txt=g_utils.add_to_txt(interaction_txt, f"Prompt Tokens: {prompt_tokens}\n", with_print=False)
                    interaction_txt=g_utils.add_to_txt(interaction_txt, f"Completion Tokens: {completion_tokens}\n", with_print=False)
                    interaction_txt=g_utils.add_to_txt(interaction_txt, f"Total Tokens: {total_tokens}\n", with_print=False)
                    interaction_txt=g_utils.add_to_txt(interaction_txt, ">>> Reasoning: \n" + resoning_content, with_print=False)
                    interaction_txt=g_utils.add_to_txt(interaction_txt, ">>> Answer: \n" + res, with_print=False)
                return res,interaction_txt
            except Exception as e:
                print("failed chat completion", e)
        raise Exception("Failed to generate")
    
    def call_llm_muti_model(self, model, prompt,interaction_txt,images):
        # self.prompt_hitory=[]
        if not self.use_history:
            self.prompt_hitory=[]
        prompt_all={"role": "user", "content": [{"type": "text", "text": prompt}]}
        for idx, image in enumerate(images):
            
            numpy_image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(numpy_image)
            # cv2.imwrite(f"chat_{len(existing_messages)}_{idx}.png", msg)  # save as logs
            byte_stream = io.BytesIO()
            pil_image.save(byte_stream, format="JPEG")
            image_bytes = byte_stream.getvalue()
            prompt_all["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
                    },
                }
            )
        # self.prompt_hitory.append({"role": "user", "content": prompt})
        self.prompt_hitory.append(prompt_all)
        truncated_messages = g_utils.truncate_message_for_token_limit(
            self.prompt_hitory, self.reply_max_tokens
        )
        for retry in range(self.retry_max):
            try:
                if interaction_txt is not None:
                    interaction_txt=g_utils.add_to_txt(interaction_txt, ">>> Prompt: \n" + prompt, with_print=False)
                print("Task prompt text:",truncated_messages)
                print("calling LLM..........")
                # Call the LLM
                # call_res = self.client.chat.completions.create(
                #     model=model,
                #     messages=truncated_messages
                # )
                print("model(call_llm):",model)
                response = self.client.chat.completions.create(
                    model=model,
                    messages=truncated_messages,
                    stream=False,
                    temperature=0
                )
                res = response.choices[0].message.content

                if self.reasoning:
                    resoning_content=response.choices[0].message.reasoning_content
                else:
                    resoning_content=""
                print("call LLM success!")
            
                with open(str(ROOT_PATH/"Tokens_consumption.txt"), "a") as file:
                    # with open("output.txt", "w") as file:
                    # file.write(f"========>{time.time()}\n")
                    time_now=datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                    file.write(f"========>{time_now}\n")
                    file.write(f"Model: {self.model}\n")
                    # file.write(f"Function: generate_feedback()\n")
                    completion_tokens = response.usage.completion_tokens
                    # file.write(f"Completion Tokens: {completion_tokens}\n")
                    file.write(f"Completion Tokens: {completion_tokens}\n")
                    prompt_tokens = response.usage.prompt_tokens
                    file.write(f"Prompt Tokens: {prompt_tokens}\n")
                    total_tokens = response.usage.total_tokens
                    file.write(f"Total Tokens: {total_tokens}\n")
                

                self.prompt_hitory.append({"role": "assistant", "content": res})
                to_print = highlight(f"{res}", PythonLexer(), TerminalFormatter())
                print(to_print)
                if interaction_txt is not None:
                    interaction_txt=g_utils.add_to_txt(interaction_txt, f"========>{time_now}\n", with_print=False)
                    interaction_txt=g_utils.add_to_txt(interaction_txt, f"Model: {self.model}\n", with_print=False)
                    interaction_txt=g_utils.add_to_txt(interaction_txt, f"Prompt Tokens: {prompt_tokens}\n", with_print=False)
                    interaction_txt=g_utils.add_to_txt(interaction_txt, f"Completion Tokens: {completion_tokens}\n", with_print=False)
                    interaction_txt=g_utils.add_to_txt(interaction_txt, f"Total Tokens: {total_tokens}\n", with_print=False)
                    interaction_txt=g_utils.add_to_txt(interaction_txt, ">>> Reasoning: \n" + resoning_content, with_print=False)
                    interaction_txt=g_utils.add_to_txt(interaction_txt, ">>> Answer: \n" + res, with_print=False)
                return res,interaction_txt
            except Exception as e:
                print("failed chat completion", e)
        raise Exception("Failed to generate")
    
    def generate_solve_mcts(self,task_goal,asset_attribute,initial_asset_state,current_asset_state,executed_code,prohibited_code,env_name,node_id,task_note):
        
        # build the prompt
        original_prompt = open(f"{self.prompt_folder}/prompt_generate_solve_mcts.txt").read()
        prompt = original_prompt.replace(
            "TASK_DESCRIPTION", task_goal
        )
        prompt = prompt.replace(
            "ASSETS_ATTRIBUTES", str(asset_attribute)
        )
        prompt = prompt.replace(
            "INITIAL_ASSET_STATE", str(initial_asset_state)
        )
        prompt = prompt.replace(
            "CURRENT_ASSET_STATE", str(current_asset_state)
        )
        prompt = prompt.replace(
            "EXECUTED_CODE", executed_code
        )
        prompt = prompt.replace(
            "PROHIBITED_ACTION", prohibited_code
        )
        prompt = prompt.replace(
            "TASK_NOTE", task_note
        )
        to_print = highlight(f"{prompt}", PythonLexer(), TerminalFormatter())
        print(to_print) # print the prompt
        
        use_llm = self.cfg["use_llm"]
        # use_llm = False
        if use_llm:
            while True:
                self.chat_log = g_utils.add_to_txt(
                    self.chat_log, "================= Solve Generate!", with_print=True
                )
                res,self.chat_log=self.call_llm(model=self.model,prompt=prompt,interaction_txt=self.chat_log)
                print("res:\n",res)
                match = re.findall(r'```python\s*(.*?)\s*```', res, re.DOTALL)
                if match is not None:
                    break
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{env_name}_{node_id}_{current_time}.txt"
            chat_log_file = ROOT_PATH/"llm/chat_log"/file_name
            with open(chat_log_file, "w") as file:
                file.write(self.chat_log)
        else:
            assumed_res_file_path = ROOT_PATH/"llm/assumed_res"/f"{env_name}.txt"
            with open(assumed_res_file_path, "r") as file:
                res = file.read()
            match = re.findall(r'```python(\s*.*?\s*)```', res, re.DOTALL)
            if match is not None:
                pass
            else:
                print("No code found code in the res")
        return match,prompt
    
    def generate_solve(self,env_name=None,task=None,assets_status=None, robot_status=None,images=None):
        """generate Solve for the task"""

        # check the input info
        obj_names=list(assets_status.keys())
        if task is None or len(obj_names) == 0:
            print("Please provide a task or a list of objects")
            return None
        
        self.chat_log = g_utils.add_to_txt(
            self.chat_log, "================= Solve Generate!", with_print=True
        )
        
        #  get original prompt
        original_prompt = open(f"{self.prompt_folder}/prompt_generate_solve.txt").read()
        
        # get object attributes
        obj_info_file = os.path.join(ROOT_PATH, "assets/objects/assets_info.json")
        obj_info = json.load(open(obj_info_file))
        print("########### objects attributes ######################")
        obj_attribute_prompt=""
        for obj_name in obj_names:
            if obj_name not in obj_info:
                print(f"Warning: Object {obj_name} not found in the asset info")
                continue
            
            status_info=obj_info[obj_name][0]["status"]
            obj_attribute_prompt+=f"{obj_name}:"+"{\n"
            obj_attribute_prompt+="\"status\":"+f"{status_info}\n"
            
            if "bbx" in obj_info[obj_name][0]:
                bbx=obj_info[obj_name][0]["bbx"]
                obj_attribute_prompt+=f'"bbx":'+f"{bbx}\n"
            elif obj_info[obj_name][0]["dataset"] == "local_rigid":
                # bbox
                # humanoidgen/assets/objects/rigidbody_objs/info_raw.json
                bbox_path = os.path.join(ROOT_PATH, "assets/objects/rigidbody_objs/info_raw.json")
                bbox_info = json.load(open(bbox_path))
                bbox_info_name=obj_info[obj_name][0]["info_name"]
                bbox_info_target=bbox_info[bbox_info_name]["bbox"]
                if bbox_info_name in bbox_info:
                    obj_attribute_prompt+=f'"bbox":'+f"{bbox_info_target}\n"
            # obj_attribute_prompt += f'"obj_name": {[obj_i["name"] for obj_i in obj_info[obj_name]]}\n'
            obj_attribute_prompt += "}\n"

        print("obj_prompt:",obj_attribute_prompt)
        print("#########################################")
        print("########### prompt_all ######################")
        
        # set the prompt
        prompt = original_prompt.replace(
            "ASSETS_ATTRIBUTES", obj_attribute_prompt
        )
        prompt = prompt.replace(
            "ASSETS_STATUS", str(assets_status)
        )
        prompt = prompt.replace(
            "ROBOT_END_EFFECTOR", str(robot_status)
        )
        prompt = prompt.replace(
            "TASK_DESCRIPTION", task
        )
        print(prompt)
        print("#########################################")
       
        while True:
            if images is not None and self.multi_model:
                res,self.chat_log=self.call_llm_muti_model(model=self.model,prompt=prompt,interaction_txt=self.chat_log,images=images)
            else:
                res,self.chat_log=self.call_llm(model=self.model,prompt=prompt,interaction_txt=self.chat_log)
            # print("res1:",res)
            # res = res.strip()
            print("res:\n",res)
            match = re.findall(r'```python\s*(.*?)\s*```', res, re.DOTALL)
            if match is not None:
                break
        
        # save chat log
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{env_name}_{current_time}.txt"
        chat_log_file = ROOT_PATH/"llm/chat_log"/file_name
        with open(chat_log_file, "w") as file:
            file.write(self.chat_log)

        folder_path =str(ROOT_PATH/"motion_planning/h1_2/solution/generated"/env_name)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            print(f"The folder '{folder_path}' exists.")
            folder_path = str(ROOT_PATH/"motion_planning/h1_2/solution/generated"/f"{env_name}_{current_time}")
            print(f"Creating a new folder '{folder_path}' with a timestamp.")
            os.makedirs(folder_path, exist_ok=True)
        else:
            print(f"The folder '{folder_path}' does not exist.")
            os.makedirs(folder_path, exist_ok=True)

        file_prefix = """from humanoidgen.motion_planning.h1_2.solution.generated.solver_env import *
def step(planner:HumanoidMotionPlanner):
"""
        for i, code_block in enumerate(match):
            file_name = os.path.join(folder_path, f"step{i}.py")

            lines=code_block.splitlines()
            if any(line.strip() != "" and not line.startswith(" ") for line in lines[1:]):
                # steps[i] = "\n    " + "\n    ".join(lines)
                code_block= "\n    ".join(lines)
                indentation_needed = True
            code_block= "\n    "+code_block+"\n"

            with open(file_name, "w") as file:
                file.write(file_prefix)  # 写入前缀
                file.write("\n")  # 添加换行
                file.write(code_block)  # 写入代码块
            print(f"Saved: {file_name}")

        print("#########################################")

        return folder_path