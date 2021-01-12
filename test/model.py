import os

from config import config
from models.model import Model

conf = config


def test_default_signature():
    os.chdir('..')
    model = Model(conf['system'], conf['model'], conf['hyperparameter'])
    casiab = model._parse_dataset_config(conf['dataset'])
    model._parse_dataloader_config(casiab, conf['dataloader'])
    assert model._log_name == os.path.join(
        'runs', 'logs', 'RGB-GaitPart_80000_64_128_128_64_1_2_4_True_True_32_5_'
                        '3_3_3_3_3_2_1_1_1_1_1_0_2_3_4_16_256_0.2_0.0001_0.9_'
                        '0.999_CASIA-B_74_30_15_3_64_32_8_16')
    assert model._signature == ('RGB-GaitPart_80000_0_64_128_128_64_1_2_4_True_'
                               'True_32_5_3_3_3_3_3_2_1_1_1_1_1_0_2_3_4_16_256_'
                               '0.2_0.0001_0.9_0.999_CASIA-B_74_30_15_3_64_32_'
                               '8_16')
