from tqdm import tqdm
from tdc.benchmark_group import admet_group

benchmark_config = {
    'caco2_wang': ('regression', False),
    'bioavailability_ma': ('binary', False),
    'lipophilicity_astrazeneca': ('regression', False),
    'solubility_aqsoldb': ('regression', False),
    'hia_hou': ('binary', False),
    'pgp_broccatelli': ('binary', False),
    'bbb_martins': ('binary', False),
    'ppbr_az': ('regression', False),
    'vdss_lombardo': ('regression', True),
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

group = admet_group(path = 'data/')
predictions_list = []

for admet_benchmark in list(benchmark_config.keys()):
    for seed in tqdm([1, 2, 3, 4, 5]):
        benchmark = group.get(admet_benchmark)
        
        predictions = {}
        name = benchmark['name']
        train_val, test = benchmark['train_val'], benchmark['test']
        train, valid = group.get_train_valid_split(benchmark = name, split_type = 'default', seed = seed)
        
            # --------------------------------------------- # 
            #  Train your model using train, valid, test    #
            #  Save test prediction in y_pred_test variable #
            # --------------------------------------------- #
            
        predictions[name] = y_pred_test
        predictions_list.append(predictions)

    results = group.evaluate_many(predictions_list)
    # {'caco2_wang': [6.328, 0.101]}