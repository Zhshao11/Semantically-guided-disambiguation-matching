import numpy as np

from sdgm.context_formal_analysis import paired_family_comparison, seed_variability


def test_family_pairing_preserves_mask_and_query_weighting_not_family_macro():
    family=np.array(['a','a','b'])
    keep=np.array([[1,1],[1,0],[1,1]],dtype=bool)
    first=np.array([[1,0],[1,1],[0,0]],dtype=bool)
    second=np.array([[0,1],[1,0],[1,1]],dtype=bool)
    result,rows=paired_family_comparison(family,keep,first,second,17,replicates=100)
    assert result['queries']==5 and result['families']==2
    assert result['new_hits']==3 and result['lost_hits']==1
    assert result['delta_coverage']==2/5
    assert result['family_improved']==1 and result['family_tied']==1
    assert rows[0]['queries']==3 and rows[0]['first_hits']==rows[0]['second_hits']==2
    again,_=paired_family_comparison(family,keep,first,second,17,replicates=100)
    assert again==result


def test_identical_predictions_have_exact_zero_paired_interval_even_if_families_unequal():
    hit=np.array([[1,0,1],[1,1,0],[0,0,0]],dtype=bool)
    result,_=paired_family_comparison(['a','a','b'],np.ones_like(hit),hit,hit,3,replicates=100)
    assert result['interval95']==[0.0,0.0]
    assert result['net_hits']==result['new_hits']==result['lost_hits']==0


def test_seed_variability_uses_actual_three_runs_and_sample_standard_deviation():
    result=seed_variability([0.1,0.2,0.3])
    assert result['seeds']==3
    np.testing.assert_allclose(result['mean'],0.2)
    np.testing.assert_allclose(result['sample_std'],0.1)
