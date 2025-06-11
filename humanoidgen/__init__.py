from pathlib import Path
import os

ROOT_PATH= Path(__file__).parent.resolve()
ASSET_DIR = Path(
    os.getenv("MS_ASSET_DIR", os.path.join(os.path.expanduser("~"), ".maniskill/data"))
)
HGENSIM_ASSET_DIR = ROOT_PATH/"assets"
BRIDGE_DATASET_ASSET_PATH = ASSET_DIR / "tasks/bridge_v2_real2sim_dataset/"
ROBOCASA_ASSET_DIR = ASSET_DIR / "scene_datasets/robocasa_dataset/assets/objects/objaverse"
REPLICA_ASSET_DIR = ASSET_DIR / "scene_datasets/replica_cad_dataset"