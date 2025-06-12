import os
import json
from humanoidgen import ROOT_PATH
from collections import OrderedDict

class Library:
    def __init__(self):
        self.data_folder = ROOT_PATH / "envs"
        self.online_asset_buffer = OrderedDict()
        assets_info = self.load_library_info()
        self.online_asset_buffer.update(assets_info)

    def load_library_info(self):
        """get the current task descriptions, assets, and code"""
        assets_info_path = os.path.join(ROOT_PATH, "assets/objects/assets_info.json")
        assets_info = json.load(open(assets_info_path))

        return assets_info