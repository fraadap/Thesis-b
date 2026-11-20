import numpy as np
import pytest

from pianob.data import faced_meta as FM
from pianob.data.adapters import FacedAdapter
from pianob.data.splits import (PROTOCOLS, check_windows_in_split, clip_fold_assignment, make_split, n_folds,
                                permute_clip_labels)
from pianob.data.windows import make_windows


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_no_subject_leakage_any_fold(protocol):
    for f in range(n_folds(protocol)):
        sp = make_split(protocol, f)
        tr, va, te = map(set, (sp.train_subjects, sp.val_subjects, sp.test_subjects))
        assert not (tr & va) and not (tr & te) and not (va & te)
        assert len(tr | va | te) == FM.N_SUBJECTS
        if sp.is_lco:
            assert not set(sp.train_clips) & set(sp.unseen_clips)
            assert sorted(sp.train_clips + sp.unseen_clips) == list(range(28))


def test_split_fm_matches_cbramod():
    sp = make_split("split_fm")
    assert sp.train_subjects == list(range(80))
    assert sp.val_subjects == list(range(80, 100))
    assert sp.test_subjects == list(range(100, 123))
    assert sp.eval_conditions() == {"all": list(range(28))}


def test_lco_folds():
    cf = clip_fold_assignment()
    sizes = [int((cf == k).sum()) for k in range(3)]
    assert sizes == [10, 9, 9]
    lab = np.array(FM.CLIP_LABELS)
    for k in range(3):
        unseen = np.where(cf == k)[0]
        # una clip per ciascuna delle 8 emozioni + neutre 2/1/1
        assert sorted(set(lab[unseen])) == list(range(9))
        assert (lab[unseen] == FM.NEUTRAL_CLASS).sum() == (2 if k == 0 else 1)


def test_lnso10_partition():
    tests = [make_split("lnso10", f).test_subjects for f in range(10)]
    assert [len(t) for t in tests] == [12] * 9 + [15]
    assert sorted(sum(tests, [])) == list(range(123))
    sp = make_split("lnso10", 3)
    assert len(sp.val_subjects) == 11


def test_lnso10_lco_index():
    sp = make_split("lnso10_lco", 7)   # fold soggetti 2, fold clip 1
    assert sp.test_subjects == make_split("lnso10", 2).test_subjects
    assert sp.unseen_clips == make_split("split_fm_lco", 1).unseen_clips


def test_val2_excludes_neutral():
    ad = FacedAdapter(synthetic=True).load()
    clips = ad.task_clips("val2")
    assert len(clips) == 24 and not set(clips) & {12, 13, 14, 15}
    sp = make_split("split_fm_lco", 0, clips)
    assert not set(sp.train_clips + sp.unseen_clips) & {12, 13, 14, 15}


def test_window_leak_detected():
    with pytest.raises(AssertionError, match="LEAKAGE"):
        check_windows_in_split([1, 2, 101], [0, 0, 0], list(range(80)), list(range(28)), "train")
    with pytest.raises(AssertionError, match="LEAKAGE"):
        check_windows_in_split([1, 2], [0, 5], [1, 2], [0, 1], "train")


def test_windows_stay_inside_trials():
    ws = make_windows(np.array([0, 0]), np.array([3, 4]), np.array([1, 1]), 6000, 800, overlap=0.5)
    assert ws.n == 2 * 14
    assert (ws.win_start + ws.win_len <= 6000).all()
    ws10 = make_windows(np.array([0]), np.array([0]), np.array([0]), 6000, 2000)
    assert list(ws10.win_start) == [0, 2000, 4000]


def test_permutation_preserves_counts():
    lab = np.array(FM.CLIP_LABELS)
    p = permute_clip_labels(lab, 3)
    assert sorted(p) == sorted(lab) and not np.array_equal(p, lab)
    v = FacedAdapter(synthetic=True).load().labels("val2")
    pv = permute_clip_labels(v, 1, np.where(v >= 0)[0])
    assert (pv[v < 0] == -1).all() and sorted(pv) == sorted(v)
