"""Independent NumPy endpoint count/quantile audit for all four heads and seeds."""

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from sdgm.io import save_json
from scripts.train_os_context_formal import settings


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/os_context_formal.json');args=p.parse_args()
    spec,cfg=settings(args.config);root=Path(spec['output'])
    rows=list(csv.DictReader((root/'endpoint_metrics_all_seeds.csv').open()))
    contrasts=json.loads((root/'paired_comparisons.json').read_text())
    checked=0;comparisons=0;loops={};initializers={};frozen_states=[]
    for seed in spec['paired_seeds']:
        out=root/f'seed{seed}'
        starts=json.loads((out/'start_audit.json').read_text())['groups']
        assert {v['parameters'] for v in starts.values()}=={393984}
        assert len({v['initial_digest'] for v in starts.values()})==1
        initializers[str(seed)]=next(iter(starts.values()))['initial_digest']
        frozen_states.append(json.loads((out/'completion.json').read_text())['frozen_state'])
        source=(out/'training_source_at_launch.py').read_text()
        environment=json.loads((out/'implementation_environment.json').read_text())
        assert hashlib.sha256(source.encode()).hexdigest()==environment['script_sha256']
        tree=ast.parse(source)
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_seed')
        # Added a source snapshot write after seed1 launch. Exclude this file
        # archival statement; the numerical training/evaluation loop is equal.
        function.body=[n for n in function.body if 'training_source_at_launch.py' not in ast.unparse(n)]
        loops[str(seed)]=hashlib.sha256(ast.dump(function,include_attributes=False).encode()).hexdigest()
        assert json.loads((out/'engineering_verification.json').read_text())['passed']
        gradients=json.loads((out/'gradient_checks.json').read_text())
        assert len(gradients)==4*4
        for g in gradients:
            for side in ['optical','sar']:assert g[side]['gradient_norm']>0 and g[side]['all_parameter_gradients_finite']
        for split in ['training_monitor','validation']:
            data=np.load(out/'evaluation'/('step05000_'+split+'_queries.npz'))
            baseline=np.load(Path(spec['stageAB_root'])/'baseline'/(split+'_queries.npz'))
            for key in ['sample_id','family_id','supervised','structure_indices','structure_nearest','structure_first']:
                np.testing.assert_array_equal(data[key],baseline[key])
            keep=data['supervised'];original=data['structure_nearest']<=8
            for row in [r for r in rows if int(r['seed'])==seed and r['split']==split]:
                k=keep if row['subset']=='all' else keep&~original
                method=row['method'];dist=data[method+'_merged_nearest'];independent=data[method+'_nearest']
                rank=data[method+'_rank_rank_min'];first=data[method+'_first']
                a=original[k];b=dist[k]<=8
                assert int(row['queries'])==k.sum()
                assert int(row['new_hits'])==np.count_nonzero(~a&b)
                assert int(row['lost_hits'])==np.count_nonzero(a&~b)
                for key,value in [('merged4',b.mean()),('independent4',(independent[k]<=8).mean()),('top1',(first[k]<=8).mean()),
                    ('nearest_median',np.median(dist[k])),('nearest_p95',np.percentile(dist[k],95))]:
                    np.testing.assert_allclose(float(row[key]),value,atol=1e-12,rtol=0)
                np.testing.assert_array_equal([float(row[key]) for key in ['rank_median','rank_p75','rank_p95']],np.percentile(rank[k],[50,75,95]))
                for key in [method+'_indices',method+'_merged_indices']:
                    index=data[key]
                    assert index.shape==(952,144,4) and ((index>=-1)&(index<1024)).all()
                    for i in range(4):
                        for j in range(i):assert not ((index[...,i]==index[...,j])&(index[...,i]>=0)).any()
                checked+=1
            for contrast in [r for r in contrasts if r['seed']==seed and r['split']==split]:
                k=keep if contrast['subset']=='all' else keep&~original
                suffix='_nearest' if contrast['metric']=='independent4' else '_merged_nearest'
                a=data[contrast['first']+suffix][k]<=8;b=data[contrast['second']+suffix][k]<=8
                assert contrast['new_hits']==int((~a&b).sum())
                assert contrast['lost_hits']==int((a&~b).sum())
                np.testing.assert_allclose(contrast['delta_coverage'],(b.sum()-a.sum())/k.sum(),atol=1e-12,rtol=0)
                comparisons+=1
    assert len(set(loops.values()))==1
    assert len(set(initializers.values()))==3
    assert all(v==frozen_states[0] for v in frozen_states)
    save_json(root/'training_code_version_audit.json',dict(passed=True,training_function_AST_sha256_by_seed=loops,
        same_numerical_training_and_evaluation_loop=True,excluded_source_archival_statement_only=True,
        interpretation='verification/reload/resource-flag extensions did not change training loop or protocol; launch sources retained'))
    save_json(root/'result_verification.json',dict(passed=True,endpoint_metric_rows_checked=checked,paired_comparisons_checked=comparisons,
        all_three_seeds_all_four_groups=True,independent_NumPy_counts_and_quantiles=True,
        same_baseline_and_fixed_miss_queries=True,candidate_budget_and_no_duplicates=True,
        initializers_same_within_seed_distinct_across_seeds=initializers,
        frozen_model_and_semantic_head_identical_across_seeds=True,
        all_recorded_modality_gradients_finite_nonzero=True,OS_test_evaluated=False,UBC_evaluated=False))
    print('All endpoint metrics and paired comparison counts independently verified',flush=True)


if __name__=='__main__':main()
