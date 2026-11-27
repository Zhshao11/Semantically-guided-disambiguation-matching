"""Paired full-split descriptor training; fixed front, head capacity and endpoint."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from sdgm.coarse import StructuralRetriever
from sdgm.context_descriptors import ContextDescriptorHead, full_grid_pair_mass_loss, orthogonal_projection
from sdgm.context_descriptor_data import frozen_inputs, group_maps
from sdgm.context_descriptor_evaluation import evaluate_context
from sdgm.continuation_diagnostics import state_digest
from sdgm.io import save_json
from sdgm.os_family_data import OSFamilyDataset
from sdgm.raw_lineage import sha256_file
from sdgm.semantic_evaluation import frozen_views
from sdgm.semantic_routing import SARPrototypeHead
from sdgm.semantic_teacher import TeacherView, verify_teacher_cache
from scripts.run_os_context_stageAB import gradient_check
from scripts.train_structure_smoke import prepare, geometry_for
from scripts.train_os_semantic_routing import frozen_provenance


def settings(path):
    spec = json.loads(Path(path).read_text())
    cfg = json.loads(Path(spec['base_config']).read_text())
    assert spec['paired_seeds'] == cfg['formal_preregistration']['paired_seeds']
    assert spec['steps'] == cfg['formal_preregistration']['steps_per_group_seed']
    assert spec['evaluation_steps'] == cfg['formal_preregistration']['evaluation_steps']
    cfg = copy.deepcopy(cfg)
    cfg['output'] = spec['output']
    return spec, cfg


def assert_same_baseline(queries, reference):
    a, b = np.load(queries), np.load(reference)
    for key in ['sample_id', 'family_id', 'supervised', 'structure_indices', 'structure_first', 'structure_nearest']:
        np.testing.assert_array_equal(a[key], b[key])


def build_runtime(cfg):
    semantic = json.loads(Path(cfg['semantic_config']).read_text())
    parent = json.loads(Path(semantic['parent_config']).read_text())
    verify_teacher_cache(Path(semantic['output']) / 'teacher', semantic)
    for key in ['structural_checkpoint', 'SAR_head_checkpoint']:
        assert sha256_file(cfg[key]) == cfg[key + '_sha256']
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = torch.device(f"cuda:{cfg['cuda_device']}")
    torch.cuda.set_device(device)
    model = StructuralRetriever(parent['temperature'], image_hw=(256, 256)).to(device)
    structural = torch.load(cfg['structural_checkpoint'], map_location='cpu', weights_only=False)
    assert structural['epoch'] == 30
    model.load_state_dict(structural['model'])
    model.requires_grad_(False).eval()
    semantic_head = SARPrototypeHead().to(device)
    sar = torch.load(cfg['SAR_head_checkpoint'], map_location='cpu', weights_only=False)
    assert sar['total_step'] == 10000
    semantic_head.load_state_dict(sar['head'])
    semantic_head.requires_grad_(False).eval()
    projection = orthogonal_projection(cfg['projection_seed']).to(device)
    train = OSFamilyDataset(semantic['family_root'], 'train', parent, training=True)
    val = OSFamilyDataset(semantic['family_root'], 'val', parent)
    monitor, validation = frozen_views(semantic, train, val)
    teacher_root = Path(semantic['output']) / 'teacher'
    views = dict(training_monitor=TeacherView(monitor, teacher_root), validation=TeacherView(validation, teacher_root))
    return semantic, parent, device, model, semantic_head, projection, train, TeacherView(train, teacher_root), views


def provenance(cfg, semantic):
    result = frozen_provenance(semantic)
    for p in [cfg['SAR_head_checkpoint'], str(Path(semantic['output'])/'teacher/cache_receipt.json'),
              'configs/os_context_descriptor.json', 'sdgm/context_descriptors.py',
              'sdgm/context_descriptor_data.py', 'sdgm/context_descriptor_evaluation.py',
              'sdgm/semantic_supplement.py', 'sdgm/semantic_bottleneck.py']:
        result[p] = sha256_file(p)
    return result


def run_seed(spec, cfg, config_path, seed):
    root = Path(spec['output'])
    root.mkdir(exist_ok=True, parents=True)
    protocol = root/'locked_protocol.json'
    locked = dict(formal=spec, base_config_sha256=sha256_file(spec['base_config']))
    if protocol.exists():
        assert json.loads(protocol.read_text()) == locked
    else:
        # Confirm A/B admission before formal initialization.
        receipt = json.loads((Path(spec['stageAB_root'])/'verification/verification_summary.json').read_text())
        assert receipt['passed']
        save_json(protocol, locked)
    if seed != spec['paired_seeds'][0]:
        assert json.loads((root/'first_seed_engineering_gate.json').read_text())['passed']
    out = root/f'seed{seed}'
    if out.exists():
        raise FileExistsError('Never overwrite/restart a completed or partial formal seed')
    semantic, parent, device, model, semantic_head, projection, train, training, views = build_runtime(cfg)
    out.mkdir()
    before = provenance(cfg, semantic)
    save_json(out/'frozen_provenance_before.json', before)
    frozen_state = dict(structural=state_digest(model.state_dict()), semantic=state_digest(semantic_head.state_dict()))
    old_projection = np.load(Path(spec['stageAB_root'])/'structural_projection.npy')
    np.testing.assert_array_equal(projection.cpu().numpy(), old_projection)
    torch.manual_seed(seed)
    initial = ContextDescriptorHead()
    initial_state = copy.deepcopy(initial.state_dict())
    initial_digest = state_digest(initial_state)
    heads, optimizers, starts = {}, {}, {}
    opt = cfg['optimizer']
    for group in cfg['groups']:
        head = ContextDescriptorHead().to(device)
        head.load_state_dict(copy.deepcopy(initial_state))
        heads[group] = head
        optimizer = torch.optim.AdamW(head.parameters(), lr=opt['learning_rate'], weight_decay=opt['weight_decay'], betas=tuple(opt['betas']))
        optimizers[group] = optimizer
        starts[group] = dict(initial_digest=state_digest(head.state_dict()), parameters=sum(p.numel() for p in head.parameters()),
            new_optimizer_empty=not optimizer.state, smoke_weights_loaded=False,
            optical_sar_independent=head.optical_head[0].weight.data_ptr()!=head.sar_head[0].weight.data_ptr())
    assert {r['initial_digest'] for r in starts.values()} == {initial_digest}
    assert {r['parameters'] for r in starts.values()} == {393984}
    save_json(out/'start_audit.json', dict(seed=seed,groups=starts,training_pairs=len(train),training_families=len({r['family_id'] for r in train.records}),
        sampler_seed=seed,augmentation='original parent sample/epoch transform rules, unchanged',fresh_initialization=True))
    save_json(out/'implementation_environment.json', dict(python=__import__('platform').python_version(), torch=torch.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(device),cuda_device=cfg['cuda_device'],
        script_sha256=sha256_file(__file__),formal_config_sha256=sha256_file(config_path),
        evaluation='eval()+no_grad with descriptor requires_grad flags preserved; frozen front/semantic requires_grad=False'))
    history=[]
    def evaluate(step):
        for split, view in views.items():
            tag=f'step{step:05d}_{split}'
            result=evaluate_context(model,semantic_head,heads,view,parent,cfg,projection,device,out/'evaluation',tag)
            assert_same_baseline(out/'evaluation'/(tag+'_queries.npz'),Path(spec['stageAB_root'])/'baseline'/(split+'_queries.npz'))
            history.append(dict(step=step,split=split,summary=result))
            print(json.dumps(dict(stage='evaluation',seed=seed,step=step,split=split,
                groups={k:dict(loss=v['loss_pair_mean'],rank=v['ranking']['all']['best_positive_rank_min']['50'],independent4=v['ranking']['all']['independent_NMS4_coverage'],merged4=v['recall_at_4']) for k,v in result.items()})),flush=True)
        save_json(out/'evaluation_history.json',history)
    start=time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    evaluate(0)
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(list(range(len(train))),batch_size=opt['batch_size'],shuffle=True,generator=generator,num_workers=0)
    iterator=None;epoch=0;updates=0;attempts=0;skipped=0
    schedule_hash=hashlib.sha256();gradients=[]
    stream=(out/'training.jsonl').open('w');schedule=(out/'shared_schedule.jsonl').open('w')
    while updates < spec['steps']:
        if iterator is None:
            epoch+=1;train.set_epoch(epoch);iterator=iter(loader)
        try: indices=next(iterator).tolist()
        except StopIteration:
            epoch+=1;train.set_epoch(epoch);iterator=iter(loader);indices=next(iterator).tolist()
        batch=prepare(default_collate([training[i] for i in indices]),device)
        attempts+=1
        plans=[train.transform_record(i) for i in indices]
        record=dict(attempt=attempts,planned_update=updates+1,epoch=epoch,indices=indices,sample_ids=batch['id'],
            transform_seeds=[p['transform_seed'] for p in plans],matrix_sha256=state_digest(batch['matrix']))
        line=json.dumps(record,sort_keys=True)
        schedule.write(line+'\n');schedule_hash.update(line.encode())
        with torch.no_grad():
            inputs=frozen_inputs(model,semantic_head,batch)
            geom=geometry_for(model,batch,parent)
        # Same one decoded/augmented batch and one frozen-front invocation are
        # shared across all four heads. Position/mask selection is not trainable.
        row=dict(attempt=attempts,step=updates+1,epoch=epoch,batch_pairs=len(indices),groups={})
        valid_any=bool(geom['supervised'].any())
        for group in cfg['groups']:
            head=heads[group];head.train();optimizer=optimizers[group];optimizer.zero_grad(set_to_none=True)
            om,sm,_,_=group_maps(inputs,projection,group)
            result=head(om,sm,model.source_pixel_xy,model.target_pixel_xy)
            loss,detail=full_grid_pair_mass_loss(result['scores'],geom,cfg['temperature'])
            assert bool(detail['valid_pair'].any()) == valid_any
            if not valid_any:
                row['groups'][group]=dict(skipped=True);continue
            loss.backward()
            check=gradient_check(head) if updates+1 in [1,1000,2500,5000] else None
            norm=torch.nn.utils.clip_grad_norm_(head.parameters(),opt['gradient_clip'],error_if_nonfinite=True)
            optimizer.step()
            if check:gradients.append(dict(step=updates+1,group=group,**check))
            row['groups'][group]=dict(loss=float(loss.detach()),gradient_norm=float(norm),valid_pairs=int(detail['valid_pair'].sum()),
                effective_queries=int(detail['supervised'].sum()),empty_positive=int(detail['empty_positive'].sum()),nonfinite=False)
        assert not model.training and not semantic_head.training
        assert all(p.grad is None and not p.requires_grad for p in list(model.parameters())+list(semantic_head.parameters()))
        if valid_any:updates+=1
        else:skipped+=1
        stream.write(json.dumps(row)+'\n')
        if attempts%100==0:
            stream.flush();schedule.flush();save_json(out/'progress.json',dict(seed=seed,**row,elapsed_seconds=time.perf_counter()-start))
            print(json.dumps(dict(stage='training',seed=seed,**row)),flush=True)
        if valid_any and updates in spec['evaluation_steps']:
            evaluate(updates)
    stream.close();schedule.close()
    save_json(out/'gradient_checks.json',gradients)
    assert frozen_state == dict(structural=state_digest(model.state_dict()),semantic=state_digest(semantic_head.state_dict()))
    assert before == provenance(cfg,semantic)
    save_json(out/'frozen_provenance_after.json',provenance(cfg,semantic))
    checkpoints={}
    for group in cfg['groups']:
        path=out/'checkpoints'/(group+'_step05000.pt');path.parent.mkdir(exist_ok=True)
        torch.save(dict(head=heads[group].state_dict(),optimizer=optimizers[group].state_dict(),step=updates,seed=seed,
            initial_digest=initial_digest,projection_sha256=sha256_file(Path(spec['stageAB_root'])/'structural_projection.npy'),
            shared_schedule_digest=schedule_hash.hexdigest(),sampler_state=generator.get_state(),config=cfg,formal=spec),path)
        checkpoints[group]=dict(path=str(path),sha256=sha256_file(path))
        assert {int(v['step']) for v in optimizers[group].state.values()} == {5000}
    save_json(out/'completion.json',dict(completed=True,seed=seed,updates_each=updates,attempted_batches=attempts,skipped_batches=skipped,
        same_batch_augmentation_and_initial_heads=True,fresh_heads_and_optimizers=True,groups=cfg['groups'],
        schedule_digest=schedule_hash.hexdigest(),frozen_state=frozen_state,frozen_sources_unchanged=True,
        wall_seconds=time.perf_counter()-start,peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
        final_checkpoints=checkpoints,OS_test_evaluated=False,UBC_evaluated=False,fixed_endpoint_not_validation_selected=True))
    print(json.dumps(dict(stage='seed_complete',seed=seed,wall_seconds=time.perf_counter()-start)),flush=True)


def verify_seed(spec,cfg,seed):
    root=Path(spec['output']);out=root/f'seed{seed}'
    complete=json.loads((out/'completion.json').read_text());assert complete['completed'] and complete['updates_each']==5000
    starts=json.loads((out/'start_audit.json').read_text())['groups']
    before=json.loads((out/'frozen_provenance_before.json').read_text())
    assert before==json.loads((out/'frozen_provenance_after.json').read_text())
    for p,d in before.items():assert sha256_file(p)==d,p
    assert all(v['new_optimizer_empty'] and not v['smoke_weights_loaded'] for v in starts.values())
    assert len({v['initial_digest'] for v in starts.values()})==1
    rows=[json.loads(l) for l in (out/'training.jsonl').read_text().splitlines()]
    assert len(rows)==complete['attempted_batches']
    effective=[r for r in rows if not r['groups'][cfg['groups'][0]].get('skipped')]
    assert len(effective)==5000
    for row in effective:
        assert set(row['groups'])==set(cfg['groups'])
        assert len({v['effective_queries'] for v in row['groups'].values()})==1
        assert all(not v['nonfinite'] for v in row['groups'].values())
    digest=hashlib.sha256()
    for line in (out/'shared_schedule.jsonl').read_text().splitlines():digest.update(line.encode())
    assert digest.hexdigest()==complete['schedule_digest']
    for group,row in complete['final_checkpoints'].items():
        assert sha256_file(row['path'])==row['sha256']
        saved=torch.load(row['path'],map_location='cpu',weights_only=False)
        assert saved['step']==5000 and saved['seed']==seed
        assert saved['initial_digest']==starts[group]['initial_digest']
        assert {int(v['step']) for v in saved['optimizer']['state'].values()}=={5000}
    for step in spec['evaluation_steps']:
        for split in ['training_monitor','validation']:
            tag=f'step{step:05d}_{split}'
            assert_same_baseline(out/'evaluation'/(tag+'_queries.npz'),Path(spec['stageAB_root'])/'baseline'/(split+'_queries.npz'))
            summaries=json.loads((out/'evaluation'/(tag+'_summary.json')).read_text())
            assert set(summaries)=={'structure',*cfg['groups']}
            assert all(s['effective_queries']==116953 for s in summaries.values())
            assert all(not s['OS_test_evaluated'] and not s['UBC_evaluated'] for s in summaries.values())
    receipt=dict(passed=True,seed=seed,all_four_updates5000=True,fresh_initializers_and_optimizers=True,
        exact_fixed_structural_baseline_all_process_points=True,same_queries_support_and_miss_group=True,
        finite_losses_and_gradients=True,frozen_weights_stats_and_sources_unchanged=True,final_test_still_sealed=True,
        gate_uses_engineering_integrity_not_validation_gain=True)
    save_json(out/'engineering_verification.json',receipt)
    if seed==spec['paired_seeds'][0]:save_json(root/'first_seed_engineering_gate.json',receipt)
    print(json.dumps(receipt),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/os_context_formal.json');p.add_argument('--seed',type=int,required=True);p.add_argument('--verify',action='store_true')
    args=p.parse_args();spec,cfg=settings(args.config)
    assert args.seed in spec['paired_seeds']
    if args.verify:verify_seed(spec,cfg,args.seed)
    else:run_seed(spec,cfg,args.config,args.seed)


if __name__=='__main__':main()
