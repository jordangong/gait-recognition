from utils.dataset import CASIAB, ClipConditions, ClipViews, ClipClasses

CASIAB_ROOT_DIR = '../data/CASIA-B-MRCNN/SEG'


def test_casiab():
    casiab = CASIAB(CASIAB_ROOT_DIR, discard_threshold=0)
    assert len(casiab) == 74 * 10 * 11

    labels = []
    for i in range(74):
        labels += [i] * 10 * 11
    assert casiab.labels.tolist() == labels

    assert casiab.metadata['labels'] == [i for i in range(74)]

    assert casiab.label_encoder.inverse_transform([0, 2]).tolist() == ['001',
                                                                       '003']


def test_casiab_test():
    casiab_test = CASIAB(CASIAB_ROOT_DIR, is_train=False, discard_threshold=0)
    assert len(casiab_test) == (124 - 74) * 10 * 11

    labels = []
    for i in range(124 - 74):
        labels += [i] * 10 * 11
    assert casiab_test.labels.tolist() == labels

    assert casiab_test.label_encoder.inverse_transform([0, 2]).tolist() == [
        '075', '077']


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


def test_caisab_class_selector():
    class_selector = {'classes': ClipClasses({'001', '003'})}
    casiab_class_001_003 = CASIAB(CASIAB_ROOT_DIR,
                                  selector=class_selector,
                                  discard_threshold=0)
    assert len(casiab_class_001_003) == 2 * 10 * 11
