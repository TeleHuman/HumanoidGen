# class LogRecord:
#     log_record_file_path = None  

#     @staticmethod
#     def update_log_file_path(log_file_path):
#         LogRecord.log_record_file_path = log_file_path

#     def write_log(write_info):
#         if LogRecord.log_record_file_path is None:
#             raise ValueError("Log file path is not set. Please call update_log_file_path() first.")
        
#         # 在log文件的基础上追加写入
#         with open(LogRecord.log_record_file_path, 'a') as log_file:
#             log_file.write(write_info + '\n')

import sys
import os

class LogRecord:
    def __init__(self, log_file_path):
        # 确保日志文件所在的目录存在
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)  # 创建目录

        self.terminal = sys.stdout  # 保存原始终端输出
        self.log_file = open(log_file_path, "w")  # 

    def write(self, message):
        self.terminal.write(message)  # 输出到终端
        self.log_file.write(message)  # 写入日志文件

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()