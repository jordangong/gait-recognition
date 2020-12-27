from utils.dataset import CASIAB, ClipConditions, ClipViews

CASIAB_ROOT_DIR = '../data/CASIA-B-MRCNN/SEG'


def test_casiab():
    casiab = CASIAB(CASIAB_ROOT_DIR, discard_threshold=0)
    assert len(casiab) == 74 * 10 * 11


def test_casiab_nm():
    nm_selector = {'conditions': ClipConditions({r'nm-0\d'})}
    casiab_nm = CASIAB(CASIAB_ROOT_DIR, selector=nm_selector,
                       discard_threshold=0)
    assert len(casiab_nm) == 74 * 6 * 11


def test_casiab_nm_bg_90():
    nm_bg_90_selector = {'conditions': ClipConditions({r'nm-0\d', r'bg-0\d'}),
                         'views': ClipViews({'090'})}
    casiab_nm_bg_90 = CASIAB(CASIAB_ROOT_DIR,
                             selector=nm_bg_90_selector,
                             discard_threshold=0)
    assert len(casiab_nm_bg_90) == 74 * (6 + 2) * 1
