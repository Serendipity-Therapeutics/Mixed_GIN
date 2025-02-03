# data/tdc_benchmark.py

from tdc.benchmark_group import admet_group
from tqdm import tqdm

class ADMETBenchmarks:
    def __init__(self):
        self.benchmark_config = {
            'caco2_wang': ('regression', False),
            'bioavailability_ma': ('binary', False),
            'lipophilicity_astrazeneca': ('regression', False),
            'solubility_aqsoldb': ('regression', False),
            'hia_hou': ('binary', False),
            'pgp_broccatelli': ('binary', False),
            'bbb_martins': ('binary', False),
            'ppbr_az': ('regression', False),
            'vdss_lombardo': ('regression', True),  # log_scale=True
            'cyp2c9_veith': ('binary', False),
            'cyp2d6_veith': ('binary', False),
            'cyp3a4_veith': ('binary', False),
            'cyp2c9_substrate_carbonmangels': ('binary', False),
            'cyp2d6_substrate_carbonmangels': ('binary', False),
            'cyp3a4_substrate_carbonmangels': ('binary', False),
            'half_life_obach': ('regression', True),
            'clearance_hepatocyte_az': ('regression', True),
            'clearance_microsome_az': ('regression', True),
            'ld50_zhu': ('regression', False),
            'herg': ('binary', False),
            'ames': ('binary', False),
            'dili': ('binary', False)
        }
        self.keys = list(self.benchmark_config.keys())

    def __call__(self, key=None):
        if isinstance(key, int):
            return self.keys[key:key+1]
        elif isinstance(key, slice):
            return self.keys[key]
        elif isinstance(key, list):
            return [self.keys[k] for k in key]
        elif key is None:
            return self.keys
        elif isinstance(key, str):
            return self.benchmark_config[key]

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.benchmark_config[key]
        raise KeyError(f"Unsupported key type: {type(key)}")