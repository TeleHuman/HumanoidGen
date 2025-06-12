import operator
import numpy as np
import random

def add_to_txt(log, text, with_print=False):
    """
    Append text to a log string and optionally print it.

    Args:
        log (str): The log string where the text will be appended.
        text (str): The text to append to the log.
        with_print (bool): If True, print the text to the console.

    Returns:
        str: The updated log string.
    """
    # Append the text to the log
    log += text + '\n'
    
    # Optionally print the text
    if with_print:
        print(text)
    
    return log

def format_dict_prompt(task_name_dict, sample_num=-1, sort_items=False):
    """format a saved dictionary into prompt"""
    if sort_items:
        task_name_dict = sorted(task_name_dict.items(), key=operator.itemgetter(0))
    prompt_replacement = ""
    sample_idx = list(range(len(task_name_dict)))
    random.shuffle(sample_idx)

    if sample_num > 0:
        sample_idx = np.random.choice(len(task_name_dict), sample_num, replace=False)

    for idx, (task_name, task_desc) in enumerate(task_name_dict.items()):
        if idx in sample_idx:
            prompt_replacement += f"- {task_name}: {task_desc}\n"

    return prompt_replacement + "\n\n"


def truncate_message_for_token_limit(message_history, max_tokens=6000):
    truncated_messages = []
    tokens = 0

    # reverse
    for idx in range(len(message_history) - 1, -1, -1):
        message = message_history[idx]
        message_tokens = len(message["content"]) / 4  # rough estimate.
        if tokens + message_tokens > max_tokens:
            break  # This message would put us over the limit

        truncated_messages.append(message)
        tokens += message_tokens

    truncated_messages.reverse()
    return truncated_messages