"""Query-paired comparisons with whole-family resampling, separate from seed SD."""

import numpy as np


def paired_family_comparison(families, keep, first_hit, second_hit, seed, replicates=2000):
    """Second minus first; resample complete families, preserving query pairing."""
    families=np.asarray(families)
    keep=np.asarray(keep,dtype=bool)
    a=np.asarray(first_hit,dtype=bool)&keep
    b=np.asarray(second_hit,dtype=bool)&keep
    rows=[]
    for family in sorted(set(families)):
        k=keep[families==family];aa=a[families==family];bb=b[families==family]
        if not k.any():continue
        rows.append(dict(family_id=str(family),queries=int(k.sum()),first_hits=int(aa.sum()),second_hits=int(bb.sum()),
            new_hits=int((~aa&bb&k).sum()),lost_hits=int((aa&~bb&k).sum())))
    if not rows:return {},[]
    matrix=np.array([[r[k] for k in ['queries','first_hits','second_hits','new_hits','lost_hits']] for r in rows],dtype=float)
    n,aa,bb,new,lost=matrix.sum(0)
    rng=np.random.default_rng(seed)
    boot=matrix[rng.integers(0,len(matrix),(replicates,len(matrix)))].sum(1)
    differences=(boot[:,2]-boot[:,1])/boot[:,0]
    delta=matrix[:,2]-matrix[:,1]
    return dict(queries=int(n),families=len(rows),first_coverage=aa/n,second_coverage=bb/n,delta_coverage=(bb-aa)/n,
        new_hits=int(new),lost_hits=int(lost),net_hits=int(new-lost),
        family_improved=int((delta>0).sum()),family_degraded=int((delta<0).sum()),family_tied=int((delta==0).sum()),
        interval95=np.percentile(differences,[2.5,97.5]).tolist(),bootstrap_seed=seed,bootstrap_replicates=replicates,
        interval_scope='paired whole source families, query-weighted; conditional on one trained seed; not training randomness or geographic independence'),rows


def seed_variability(values):
    """Describe the actual repeated training runs, without pooling query counts."""
    values=np.asarray(values,dtype=float)
    return dict(seeds=len(values),values=values.tolist(),mean=float(values.mean()),
        sample_std=float(values.std(ddof=1)) if len(values)>1 else None,
        minimum=float(values.min()),maximum=float(values.max()),
        interpretation='descriptive training-seed variation; three seeds do not establish a precise population interval')
