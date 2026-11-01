"""Validation-only paired benchmark; both production modules stay unchanged."""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import traceback
import torch
from tilelang.utils.tensor import torch_assert_close

parser=argparse.ArgumentParser()
for key in ['baseline','candidate','reference','cases','output']:
    parser.add_argument('--'+key,required=True)
parser.add_argument('--ids',default='1,2,3,4,5,6')
parser.add_argument('--trials',type=int,default=7)
parser.add_argument('--repeats',type=int,default=100)
parser.add_argument('--warmup',type=int,default=200)
args=parser.parse_args()
output_path=Path(args.output)
report={'status':'running','method':{'preallocated_outputs_both':True,'alternating_order':True,
        'trials':args.trials,'repeats':args.repeats,'warmup_each':args.warmup,
        'seeds':[0,42],'rtol':2e-4,'atol':1e-4,'max_mismatched_ratio':0.01,
        'timing':'synchronized CUDA events around whole repeat batch; median over trials'},'artifacts':{},'cases':[]}
def save():
    output_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
def load(path,name):
    path=Path(path)
    report['artifacts'][name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module
def diagnostics(out,ref):
    finite=bool(torch.isfinite(out).all().item())
    assert finite,'Output contains NaN/Inf'
    torch_assert_close(out,ref,rtol=2e-4,atol=1e-4,max_mismatched_ratio=0.01)
    mismatches=int((~torch.isclose(out,ref,rtol=2e-4,atol=1e-4,equal_nan=False)).sum().item())
    return {'accepted':True,'finite':finite,'mismatch_fraction':mismatches/out.numel(),
            'max_abs_error':float((out.float()-ref.float()).abs().max().item())}
def timed(call):
    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.repeats):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)/args.repeats
try:
    baseline=load(args.baseline,'score96_baseline')
    candidate=load(args.candidate,'score96_candidate')
    reference=load(args.reference,'score96_reference')
    report['device']=str(torch.cuda.get_device_properties(0))
    assert torch.cuda.is_available()
    selected={int(value) for value in args.ids.split(',')}
    cases=[case for case in json.loads(Path(args.cases).read_text()) if int(case['case_id']) in selected]
    assert len(cases)==len(selected)
    save()
    for case in cases:
        shape=tuple(int(case[key]) for key in ['batch','heads','kv_heads','kv_ctx','dim','pe_dim'])
        b,h,kh,n,d,p=shape
        row={'case_id':case['case_id'],'shape':dict(zip(['batch','heads','kv_heads','kv_ctx','dim','pe_dim'],shape)),
             'status':'running','seeds':[],'baseline_ms':None,'candidate_ms':None}
        report['cases'].append(row)
        save()
        print('CASE '+str(case['case_id'])+' '+str(shape),flush=True)
        try:
            for role,module in [('baseline',baseline),('candidate',candidate)]:
                try:
                    module._get_kernel(*shape)
                    torch.cuda.synchronize()
                except Exception:
                    row['error_stage']=role+'_compile'
                    raise
            seed0=None
            for seed in [0,42]:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                tensors=[torch.randn(s,device='cuda',dtype=torch.float16) for s in [(b,h,d),(b,h,p),(b,n,kh,d),(b,n,kh,p)]]
                ref=reference.ref_program(*tensors)
                torch.cuda.synchronize()
                assert bool(torch.isfinite(ref).all().item())
                outputs={role:torch.full((b,h,d),float('nan'),device='cuda',dtype=torch.float16) for role in ['baseline','candidate']}
                record={'seed':seed}
                for role,module in [('baseline',baseline),('candidate',candidate)]:
                    module.run_kernel(*tensors,outputs[role],*shape)
                    torch.cuda.synchronize()
                    record[role]=diagnostics(outputs[role],ref)
                row['seeds'].append(record)
                save()
                if seed==0:
                    seed0=(tensors,outputs)
                del ref,tensors,outputs
            tensors,outputs=seed0
            calls={
                'baseline':lambda:baseline.run_kernel(*tensors,outputs['baseline'],*shape),
                'candidate':lambda:candidate.run_kernel(*tensors,outputs['candidate'],*shape),
            }
            for call in calls.values():
                for _ in range(args.warmup):
                    call()
            torch.cuda.synchronize()
            samples={'baseline':[],'candidate':[]}
            for trial in range(args.trials):
                order=['baseline','candidate'] if trial%2==0 else ['candidate','baseline']
                for role in order:
                    value=timed(calls[role])
                    assert value>0
                    samples[role].append(value)
                row['samples_ms']=samples
                save()
            row['baseline_ms']=statistics.median(samples['baseline'])
            row['candidate_ms']=statistics.median(samples['candidate'])
            row['speedup']=row['baseline_ms']/row['candidate_ms']
            row['paired_trial_speedups']=[x/y for x,y in zip(samples['baseline'],samples['candidate'])]
            row['status']='passed'
            print(json.dumps({key:row[key] for key in ['case_id','baseline_ms','candidate_ms','speedup','status']}),flush=True)
            del tensors,outputs,calls,seed0
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as error:
            row.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
            save()
            print(row['traceback'],flush=True)
            report['status']='failed'
            raise
        save()
    report['status']='passed'
except Exception as error:
    report.update(status='failed',error_type=type(error).__name__,error=str(error))
    save()
    sys.exit(1)
save()
