import os

from config import config
from models.model import Model
from utils.dataset import ClipConditions

conf = config
os.chdir('..')


def test_default_signature():
    model = Model(conf['system'], conf['model'], conf['hyperparameter'])
    casiab = model._parse_dataset_config(conf['dataset'])
    model._parse_dataloader_config(casiab, conf['dataloader'])
    assert model._log_name == os.path.join(
        'runs', 'logs', 'RGB-GaitPart_80000_64_128_128_64_1_2_4_True_True_32_5_'
                        '3_3_3_3_3_2_1_1_1_1_1_0_2_3_4_16_256_0.2_0.0001_0.001_'
                        '500_0.9_CASIA-B_74_30_15_3_64_32_8_16')
    assert model._checkpoint_sig == ('RGB-GaitPart_0_80000_64_128_128_64_1_2_4_'
                                     'True_True_32_5_3_3_3_3_3_2_1_1_1_1_1_0_2_'
                                     '3_4_16_256_0.2_0.0001_0.001_500_0.9_'
                                     'CASIA-B_74_30_15_3_64_32_8_16')


def test_default_signature_with_selector():
    model = Model(conf['system'], conf['model'], conf['hyperparameter'])
    casiab = model._parse_dataset_config(dict(
        **conf['dataset'],
        **{'selector': {'conditions': ClipConditions({r'nm-0\d', r'bg-0\d'})}}
    ))
    model._parse_dataloader_config(casiab, conf['dataloader'])
    assert model._log_name == os.path.join(
        'runs', 'logs', 'RGB-GaitPart_80000_64_128_128_64_1_2_4_True_True_32_5_'
                        '3_3_3_3_3_2_1_1_1_1_1_0_2_3_4_16_256_0.2_0.0001_0.001_'
                        '500_0.9_CASIA-B_74_30_15_3_64_32_bg-0\\d_nm-0\\d_8_16')
    assert model._checkpoint_sig == ('RGB-GaitPart_0_80000_64_128_128_64_1_2_4_'
                                     'True_True_32_5_3_3_3_3_3_2_1_1_1_1_1_0_2_'
                                     '3_4_16_256_0.2_0.0001_0.001_500_0.9_'
                                     'CASIA-B_74_30_15_3_64_32_bg-0\\d_nm-0\\d_'
                                     '8_16')
