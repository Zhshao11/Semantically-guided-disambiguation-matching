"""All seeds, all four groups, all fixed queries and structural misses, no selection."""

import argparse
import json
from pathlib import Path

import numpy as np

from sdgm.context_formal_analysis import paired_family_comparison,seed_variability
from sdgm.io import save_json,save_csv
from scripts.train_os_context_formal import settings


CONTRASTS=[('structure_point','structure_context','structure_context_effect'),
    ('semantic_point','semantic_context','semantic_context_effect'),
    ('structure_context','semantic_context','semantic_extra_over_context'),
    ('structure_point','semantic_point','semantic_extra_point')]


def summarize(config):
    spec,cfg=settings(config);root=Path(spec['output'])
    metrics=[];comparisons=[];families=[];geometry=[];by_seed={};process=[]
    semantic=json.loads(Path(cfg['semantic_config']).read_text())
    for seed in spec['paired_seeds']:
        out=root/f'seed{seed}'
        assert json.loads((out/'engineering_verification.json').read_text())['passed']
        for evaluation in json.loads((out/'evaluation_history.json').read_text()):
            for method,s in evaluation['summary'].items():
                process.append(dict(seed=seed,step=evaluation['step'],split=evaluation['split'],method=method,loss=s['loss_pair_mean'],
                    merged4=s['recall_at_4'],independent4=s['ranking']['all']['independent_NMS4_coverage'],
                    rank_median=s['ranking']['all']['best_positive_rank_min']['50']))
        by_seed[seed]={}
        for split in ['training_monitor','validation']:
            prefix=out/'evaluation'/('step05000_'+split)
            summaries=json.loads(Path(str(prefix)+'_summary.json').read_text())
            data=np.load(str(prefix)+'_queries.npz');keep=data['supervised'];original=data['structure_nearest']<=8
            by_seed[seed][split]=summaries
            groups=dict(all=keep,structure_miss=keep&~original)
            for method in ['structure']+cfg['groups']:
                summary=summaries[method]
                for subset,k in groups.items():
                    ranks=summary['ranking'][subset]
                    stats=ranks['best_positive_rank_min'];worst=ranks['best_positive_rank_max']
                    a=original[k];b=data[method+'_merged_nearest'][k]<=8
                    metrics.append(dict(seed=seed,split=split,method=method,subset=subset,queries=int(k.sum()),
                        rank_median=stats.get('50'),rank_p75=stats.get('75'),rank_p95=stats.get('95'),
                        rank_max_median=worst.get('50'),rank_max_p75=worst.get('75'),rank_max_p95=worst.get('95'),
                        top1=ranks['top1_coverage'],independent4=ranks['independent_NMS4_coverage'],
                        merged4=float(b.mean()),new_hits=int((~a&b).sum()),lost_hits=int((a&~b).sum()),
                        rescue_rate=float((~a&b).sum()/(~a).sum()) if (~a).any() else None,
                        nearest_median=float(np.median(data[method+'_merged_nearest'][k])),
                        nearest_p95=float(np.percentile(data[method+'_merged_nearest'][k],95)),
                        loss=summary['loss_pair_mean']))
            tests=CONTRASTS+[('structure',m,'candidate_net_'+m) for m in cfg['groups']]
            for first,second,label in tests:
                for score,suffix in [('independent4','_nearest'),('merged4','_merged_nearest')]:
                    for subset,k in groups.items():
                        result,rows=paired_family_comparison(data['family_id'],k,data[first+suffix]<=8,data[second+suffix]<=8,
                            cfg['bootstrap']['seed'],cfg['bootstrap']['replicates'])
                        comparisons.append(dict(seed=seed,split=split,comparison=label,first=first,second=second,metric=score,subset=subset,**result))
                        families.extend(dict(seed=seed,split=split,comparison=label,metric=score,subset=subset,**r) for r in rows)
            plans=[json.loads(l) for l in (Path(semantic['diagnostic_root'])/('monitor_transforms.jsonl' if split=='training_monitor' else 'validation_transforms.jsonl')).read_text().splitlines()]
            assert [p['sample_id'] for p in plans]==data['sample_id'].tolist()
            rotation=np.broadcast_to(np.array([abs(p['parameters']['rotation_degrees']) for p in plans])[:,None],keep.shape)
            scale=np.broadcast_to(np.array([abs(p['parameters']['scale']-1) for p in plans])[:,None],keep.shape)
            fraction=np.minimum(data['source_window_valid_fraction'][...,3],data['target_GT_window_valid_fraction'][...,3])
            axes=[('absolute_rotation_degrees',rotation,cfg['geometry_diagnostics']['absolute_rotation_degrees_bins']),
                ('absolute_scale_deviation',scale,cfg['geometry_diagnostics']['absolute_scale_deviation_bins']),
                ('min_source_targetGT_window9_valid_fraction',fraction,cfg['geometry_diagnostics']['window9_valid_fraction_bins'])]
            for axis,values,bins in axes:
                for lo,hi in zip(bins[:-1],bins[1:]):
                    for subset,k0 in groups.items():
                        k=k0&(values>=lo)&(values<hi)
                        if not k.any():continue
                        for method in ['structure']+cfg['groups']:
                            a=original[k];b=data[method+'_merged_nearest'][k]<=8
                            geometry.append(dict(seed=seed,split=split,axis=axis,lower=lo,upper=hi,subset=subset,method=method,queries=int(k.sum()),
                                rank_median=float(np.median(data[method+'_rank_rank_min'][k])),
                                independent4=float((data[method+'_nearest'][k]<=8).mean()),merged4=float(b.mean()),
                                new_hits=int((~a&b).sum()),lost_hits=int((a&~b).sum()),delta_merged4=float((b.astype(int)-a).mean())))
    save_csv(root/'endpoint_metrics_all_seeds.csv',metrics)
    save_json(root/'paired_comparisons.json',comparisons)
    save_csv(root/'paired_comparisons.csv',[dict(**{k:v for k,v in r.items() if k!='interval95'},interval95_lower=r['interval95'][0],interval95_upper=r['interval95'][1]) for r in comparisons])
    save_csv(root/'paired_family_changes.csv',families)
    save_csv(root/'geometry_strata.csv',geometry)
    save_csv(root/'fixed_process_metrics.csv',process)
    variability=[]
    for split in ['training_monitor','validation']:
        for subset in ['all','structure_miss']:
            for method in cfg['groups']:
                selected=[r for r in metrics if r['split']==split and r['subset']==subset and r['method']==method]
                for key in ['merged4','independent4','top1','rank_median','rank_p75','rank_p95']:
                    variability.append(dict(split=split,subset=subset,method=method,metric=key,**seed_variability([r[key] for r in selected])))
            for label in [r[2] for r in CONTRASTS]+['candidate_net_'+m for m in cfg['groups']]:
                for metric in ['merged4','independent4']:
                    selected=[r for r in comparisons if r['split']==split and r['subset']==subset and r['comparison']==label and r['metric']==metric]
                    variability.append(dict(split=split,subset=subset,comparison=label,metric=metric,**seed_variability([r['delta_coverage'] for r in selected])))
    save_json(root/'training_seed_variability.json',variability)
    make_report(root,spec,cfg,metrics,comparisons,variability,geometry)
    plot_process(root,spec,cfg,process)
    print('All endpoints, three-way contrasts, family intervals and seed variability summarized',flush=True)


def plot_process(root,spec,cfg,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,7),sharex=True)
    for col,seed in enumerate(spec['paired_seeds']):
        for method in cfg['groups']:
            for row,split in enumerate(['training_monitor','validation']):
                selected=[r for r in rows if r['seed']==seed and r['method']==method and r['split']==split]
                axes[row,col].plot([r['step'] for r in selected],[r['merged4']*100 for r in selected],'o-',label=method)
                axes[row,col].set_title(f'{split}, seed{seed}');axes[row,col].grid(alpha=.2);axes[row,col].set_ylabel('Fixed2+2 coverage (%)')
    axes[0,0].legend(fontsize=7)
    for ax in axes[-1]:ax.set_xlabel('Update (fixed5000 endpoint, process points diagnostic only)')
    fig.tight_layout();fig.savefig(root/'fixed_process_curves.png',dpi=180);plt.close(fig)


def make_report(root,spec,cfg,metrics,comparisons,variability,geometry):
    names=dict(structure='原4结构',structure_point='结构逐点',structure_context='结构上下文',semantic_point='语义逐点',semantic_context='语义上下文')
    labels=dict(structure_context_effect='结构上下文－结构逐点',semantic_context_effect='语义上下文－语义逐点',
        semantic_extra_over_context='语义上下文－结构上下文',semantic_extra_point='语义逐点－结构逐点')
    def contrast(label):
        return next(v for v in variability if v.get('comparison')==label and v['split']=='validation' and v['subset']=='all' and v['metric']=='merged4')
    structural=contrast('candidate_net_structure_context');semantic=contrast('candidate_net_semantic_context');extra=contrast('semantic_extra_over_context')
    lines=['# 冻结前端上下文描述子：正式三配对种子结果','',
        '四组均重新初始化描述子头和AdamW；先完成首个配对种子的全部四组5,000次更新并通过工程核验，再完成另外两种子的全部四组。终点固定5,000步，没有按验证选步数或按收益追加种子。OS最终测试与UBCv2测试池继续封存。','',
        '本轮分开考察上下文效果、语义额外贡献和实际四候选净收益。指标相对合成标签，来源family隔离不等于地理独立。','',
        '## 1. 冻结条件与执行审计','',
        '第30轮结构前端、运行统计、DINO教师及32原型、KL续训终点总10,000步SAR语义头全部冻结。输入原生256，32×32目标网格、144源点，8 px正集合与召回门槛、24 px顺序NMS不变。每组393,984个新头参数，两模态各独立640→256→128 MLP；相同配对种子内四组初始权重完全一致，单次数据读取／增强／冻结前端输出共享。无语义组沿用同一冻结行正交结构投影。','',
        'AdamW lr=1e-3恒定、weight_decay=1e-4、betas=(0.9,0.999)、梯度裁剪1.0，batch=16，保留epoch末尾短批；损失为tau=0.1全有效1024网格正邻域概率质量，每对先平均有效查询。四组不加入其他损失，不融合结构分数。数据随机排列使用各配对种子；样本/epoch增强种子沿用原数据链，因此种子间只改变初始化和样本顺序，不额外改增强范围。','',
        '过程评价固定0/1,000/2,500/5,000步，每次使用原952对训练监测集和952对验证集；各自116,953有效查询，验证238个family。固定原结构漏检组为验证46,631查询，训练监测34,952查询，所有模型/种子共用。排名保留1e-6平局的乐观/保守两端，NMS规则保持既有实现。','',
        '| 种子 | GPU | 每组成功更新 | 尝试/跳过批次 | 含过程评价耗时(s) | 峰值已分配显存(MiB) |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    extra_rows=[r for r in comparisons if r['comparison']=='semantic_extra_over_context' and r['split']=='validation' and r['subset']=='all' and r['metric']=='merged4']
    extra_positive=sum(x>0 for x in extra['values']);extra_negative=sum(x<0 for x in extra['values'])
    cross_zero=sum(r['interval95'][0]<=0<=r['interval95'][1] for r in extra_rows)
    context_positive=all(x>0 for label in ['structure_context_effect','semantic_context_effect'] for x in contrast(label)['values'])
    semantic_confirmed=extra_positive==3 and all(r['interval95'][0]>0 for r in extra_rows)
    headline=('上下文的三种子对照均为正' if context_positive else '上下文对比须见逐种子结果')+'，'+('语义有稳定额外贡献' if semantic_confirmed else '尚未证实语义在结构上下文之外的稳定额外贡献')
    verdict=(f"**本轮结论：{headline}。** "
        f"结构上下文的2＋2相对原4结构三个种子平均增加{structural['mean']*100:.4f}个百分点，"
        f"语义上下文增加{semantic['mean']*100:.4f}个百分点；"
        f"语义上下文－结构上下文仅为{extra['mean']*100:+.4f}±{extra['sample_std']*100:.4f}个百分点（均值±训练种子样本标准差），"
        f"三个种子中{extra_positive}正、{extra_negative}负，{cross_zero}/3个种子的family配对95%区间跨0。")
    lines[4:4]=[verdict,'']
    for seed in spec['paired_seeds']:
        c=json.loads((root/f'seed{seed}'/'completion.json').read_text())
        env=json.loads((root/f'seed{seed}'/'implementation_environment.json').read_text())
        lines.append(f"| {seed} | {env['cuda_device']} | {c['updates_each']} | {c['attempted_batches']} / {c['skipped_batches']} | {c['wall_seconds']:.1f} | {c['peak_allocated_mib']:.1f} |")
    lines+=['','上述耗时不含进程初始化与数据/教师审计；共享GPU负载会影响时间。语义组另有既有DINO缓存/原型拟合及SAR头10,000步训练成本，本轮等新增预算不等于相同总预训练成本。','',
        '## 2. 验证终点：全部查询与固定结构漏检组','',
        '原结构验证正名次中位/P75/P95为2/5/42；固定结构漏检组6/17/111。原4结构@1=47.8337%、@4=60.1284%，每个种子、每个过程点的结构候选/距离/有效查询均精确核对旧基线。','',
        '| 种子 | 组别 | 全部正名次 中位/P75/P95 | 漏检组正名次 中位/P75/P95 | 独立四峰 全部/漏检组 | 2＋2@4 | 新增/丢失 | 补救率 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for seed in spec['paired_seeds']:
        for group in cfg['groups']:
            a=next(r for r in metrics if r['seed']==seed and r['split']=='validation' and r['method']==group and r['subset']=='all')
            b=next(r for r in metrics if r['seed']==seed and r['split']=='validation' and r['method']==group and r['subset']=='structure_miss')
            ranks=lambda r:'/'.join(f"{r[k]:g}" for k in ['rank_median','rank_p75','rank_p95'])
            lines.append(f"| {seed} | {names[group]} | {ranks(a)} | {ranks(b)} | {a['independent4']*100:.4f}% / {b['independent4']*100:.4f}% | {a['merged4']*100:.4f}% | {a['new_hits']}/{a['lost_hits']} | {a['rescue_rate']*100:.4f}% |")
    lines+=['','完整CSV另列训练监测、独立最高分覆盖、最近候选距离中位/P95，以及平局保守排名。漏检组的2＋2覆盖即补救率，其相对结构丢失必为0，不把这一条件子组当作总体净收益。','',
        '![三个种子固定终点](three_seed_validation.png)','',
        '## 3. 三种收益分开比较','',
        '下表为每个训练种子内的验证family配对95%重采样区间，单位为百分点。每次整family重采样并保留查询配对，2,000次，按查询总数加权。区间条件于本训练种子，不估计训练随机性。','',
        '| 种子 | 比较 | 独立四峰Δ [family区间] | 2＋2Δ [family区间] | 2＋2配对新增/丢失 |',
        '| --- | --- | ---: | ---: | ---: |']
    for seed in spec['paired_seeds']:
        for first,second,label in CONTRASTS+[('structure',m,'candidate_net_'+m) for m in cfg['groups']]:
            selected=[r for r in comparisons if r['seed']==seed and r['split']=='validation' and r['subset']=='all' and r['comparison']==label]
            a=next(r for r in selected if r['metric']=='independent4');b=next(r for r in selected if r['metric']=='merged4')
            fmt=lambda r:f"{r['delta_coverage']*100:+.4f} [{r['interval95'][0]*100:+.4f}, {r['interval95'][1]*100:+.4f}]"
            lines.append(f"| {seed} | {labels.get(label,names.get(second,second)+'－原4结构')} | {fmt(a)} | {fmt(b)} | {b['new_hits']}/{b['lost_hits']} |")
    lines+=['','独立检索比较与2＋2比较分列：后者保留前2结构峰并按原24 px排重补峰，收益受同一结构前缀约束。不能把不同候选协议混为同一个指标。','',
        '## 4. 训练随机性：三个种子的单独值、均值与样本标准差','',
        '| 验证指标／比较 | 三种子值（按登记顺序，%或百分点） | 均值 | 样本标准差 |',
        '| --- | --- | ---: | ---: |']
    for v in variability:
        if v['split']!='validation' or v['subset']!='all' or v['metric']!='merged4':continue
        if 'method' in v:
            label=names[v['method']]+' 2＋2@4'
        elif v['comparison'] in labels:
            label=labels[v['comparison']]
        else:
            label=names[v['comparison'].removeprefix('candidate_net_')]+'－原4结构'
        lines.append(f"| {label} | {', '.join(f'{x*100:.4f}' for x in v['values'])} | {v['mean']*100:.4f} | {v['sample_std']*100:.4f} |")
    lines+=['','三个训练种子是有限重复，均值±样本标准差只描述实际三次运行；不把family区间替代这一层变动，也不从三次重复构造过强总体置信结论。','',
        '![固定过程评价](fixed_process_curves.png)','',
        '## 5. 上下文的几何限制','',
        'geometry_strata.csv按绝对旋转[0,5)、[5,10)、[10,15)、[15,20)度，尺度偏离，以及源/GT目标9×9窗口有效比例的较小值分组。GT目标位置只用于诊断分组，没有在前向对齐窗口或搬运光学信息。正方形均值窗口并不仿射不变，当前生成器仅含旋转、均匀尺度和平移，不包含各向异性变换，不能宣称已验证各向异性鲁棒性。','',
        '| 验证分组（3种子均值） | 每种子查询数 | 组别 | 独立四峰 | 2＋2@4 | 相对4结构Δ（百分点） |',
        '| --- | ---: | --- | ---: | ---: | ---: |']
    strata={}
    for r in geometry:
        if r['split']=='validation' and r['subset']=='all':
            strata.setdefault((r['axis'],r['lower'],r['upper'],r['method']),[]).append(r)
    for (axis,lo,hi,method),rows in strata.items():
        if method=='structure':continue
        lines.append(f"| {axis} [{lo:.8g},{hi:.8g}) | {rows[0]['queries']} | {names[method]} | {np.mean([r['independent4'] for r in rows])*100:.4f}% | {np.mean([r['merged4'] for r in rows])*100:.4f}% | {np.mean([r['delta_merged4'] for r in rows])*100:+.4f} |")
    boundary=[r for r in geometry if r['split']=='validation' and r['subset']=='all' and r['method']=='structure_context' and r['axis']=='min_source_targetGT_window9_valid_fraction' and r['lower']==0]
    interior=[r for r in geometry if r['split']=='validation' and r['subset']=='all' and r['method']=='structure_context' and r['axis']=='min_source_targetGT_window9_valid_fraction' and r['lower']==0.999999]
    lines+=['',f"边界与内部需分开判断：结构上下文在有效比例<0.5的{boundary[0]['queries']:,}个查询上平均{np.mean([r['delta_merged4'] for r in boundary])*100:+.4f}个百分点，在近完整窗口的{interior[0]['queries']:,}个查询上平均{np.mean([r['delta_merged4'] for r in interior])*100:+.4f}个百分点。不同组的地物、基线难度和剩余提升空间不同；这些是配对描述性结果，不能单独归因为窗口仿射变化。",'',
        '## 6. 结论的判据','']
    for label in [r[2] for r in CONTRASTS]+['candidate_net_'+m for m in cfg['groups']]:
        v=next(r for r in variability if r.get('comparison')==label and r['split']=='validation' and r['subset']=='all' and r['metric']=='merged4')
        title=labels.get(label,names[label.removeprefix('candidate_net_')]+'－原4结构') if label.startswith('candidate_net_') else labels[label]
        positive=sum(x>0 for x in v['values'])
        lines.append(f"- {title}：三个种子中{positive}/3个2＋2差值为正，均值{v['mean']*100:+.4f}个百分点，范围[{v['minimum']*100:+.4f}, {v['maximum']*100:+.4f}]。")
    lines+=['','应结合上面的上下文对照、语义额外贡献对照、实际净覆盖及family区间作结论，不用训练损失或排名单项宣布成功；本轮不接后续关系模块。','',
        '91项单测通过。独立结果核验复算60行终点指标和192项配对比较，并检查候选预算/去重、三种子初始权重差异、冻结前端一致性、启动源码中数值训练循环一致。各seed工程验收还完整复放epoch采样计划，并重载四个终点头对两种固定评估集的前32对逐项复现候选/排名/距离。','',
        '完整交付：endpoint_metrics_all_seeds.csv；paired_comparisons.json/.csv；paired_family_changes.csv；training_seed_variability.json；geometry_strata.csv；各seed目录的共享批次计划、冻结前后校验、逐步训练日志、0/1000/2500/5000逐对/family/查询评价和工程核验。大权重留在zzvr，各最终checkpoint哈希在各seed/completion.json。GitHub归档包含终点查询数组及全周期汇总，18份过程查询大数组完整保存在zzvr和本地outputs，路径及校验值见publication_index.json。','',
        '```bash','export PYTHONPATH=.','.venv/bin/python scripts/train_os_context_formal.py --seed 2026112706',
        '.venv/bin/python scripts/train_os_context_formal.py --seed 2026112706 --verify',
        '# 首种子工程核验通过后，依次对2707和2708重复训练与--verify。',
        '.venv/bin/python scripts/summarize_os_context_formal.py','```','']
    (root/'REPORT_zh.md').write_text('\n'.join(lines))
    # Standalone figures use explicit seeds and endpoints, never selected steps.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for i,group in enumerate(cfg['groups']):
        rows=[r for r in metrics if r['split']=='validation' and r['subset']=='all' and r['method']==group]
        for ax,key in zip(axes,['independent4','merged4']):
            ax.plot(range(3),[r[key]*100 for r in rows],'o-',label=group)
    for ax in axes:
        ax.axhline(60.12842765897412,color='black',linestyle='--',label='original4structure')
        ax.set_xticks(range(3),[str(s) for s in spec['paired_seeds']],fontsize=8);ax.set_xlabel('Paired training seed');ax.set_ylabel('Validation coverage (%)');ax.grid(alpha=.2)
    axes[0].set_title('Independent descriptor NMS4');axes[1].set_title('Fixed2structure +2descriptor');axes[0].legend(fontsize=7)
    fig.suptitle('Fixed5000-update endpoints, native256,8px recall, same1024-grid search');fig.tight_layout();fig.savefig(root/'three_seed_validation.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/os_context_formal.json');args=p.parse_args();summarize(args.config)
