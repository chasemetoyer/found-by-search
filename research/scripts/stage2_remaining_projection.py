"""Remaining Stage2 costs including retained canceled calls, retries and remote Sandbox overhead."""
import argparse,collections,json
from decimal import Decimal as D
from pathlib import Path


def project(ledger,replay):
    tasks=('DoorKey-8x8','UnlockPickup');units={};remaining={}
    complete=set(replay['recorded_fitness_areas'])
    for task in tasks:
        samples=[D(r['guard_charge_usd']) for k,r in ledger['calls'].items()
                 if k.startswith(('search_cma-','search_random-')) and '-'+task+'-' in k
                 and r['status']=='complete' and not r.get('recovery_note')]
        units[task]={'n':len(samples),'mean_guard_usd':str(sum(samples)/len(samples))}
        remaining[task]=780-sum('-'+task+'-' in k for k in complete)
    future=sum(D(units[t]['mean_guard_usd'])*remaining[t] for t in tasks)
    base=D(ledger['guard_total_usd'])
    worker_reserves=32*D('0.06419640')  # registered3600s +60s termination allowance
    # Current official Sandbox rates differ from Function worker rates.
    cpu_rate=D('0.00003942');ram_rate=D('0.00000667')
    rate=cpu_rate+2*ram_rate  # forecast envelope1physicalcore+2GiB, not a verified configured limit
    nominal=rate*8*3600;reservation=rate*12*3600
    setup=D('0.10')  # additional verification/setup allowance, beyond retained ledger charges
    old={r['function_call_id'] for r in ledger.get('infrastructure_history',[])}
    old.update(r['function_call_id'] for r in ledger['calls'].values() if r['status'] in ('retry_pending','uncertain') and r.get('provider_terminal_status')=='TERMINATED')
    return {'prelaunch_review_threshold_usd':33,'runtime_cap_removal_approved':True,
            'remaining_logical_training_inputs':sum(remaining.values()),'completed_recoverable':len(complete),
            'ordinary_completed_guard_unit_costs':units,'remaining_by_task':remaining,
            'current_guard_including_prior_calls_usd':str(base),
            'remaining_training_guard_forecast_usd':str(future),
            'max_worker_reservations_usd':str(worker_reserves),
            'controller_sandbox_cpu_usd_per_core_second':str(cpu_rate),'controller_sandbox_ram_usd_per_gib_second':str(ram_rate),
            'controller_forecast_resource_envelope':{'physical_cores':1,'memory_gib':2,'not_yet_provisioned':True},
            'controller_expected_8h_usd':str(nominal),'controller_reserved_12h_usd':str(reservation),
            'verification_setup_allowance_usd':str(setup),
            'expected_guard_at_end_usd':str(base+future+nominal+setup),
            'nominal_guard_peak_usd':str(base+future+nominal+setup+worker_reserves),
            'conservative_guard_peak_usd':str(base+future+reservation+setup+worker_reserves),
            'launch_gate':'STOP' if base+future+reservation+setup+worker_reserves>D(33) else 'PASS',
            'pricing_source':'https://modal.com/pricing','pricing_checked_date':'2026-10-28',
            'retained_history_rows':len(ledger.get('infrastructure_history',[])),
            'basis':'Current guard retained in full, including every prior completed charge, canceled-call reservation, history and0.50allowance. Add1068remaining training costs (including36authorized fresh retries), controller lifetime reservation, extra0.10setup and max32worker reservations using the same conservative additive peak convention as the earlier projection. No old canceled charge released. No monetary cap is imposed by this forecast.',
            'limitations':['Projection, not invoice or guarantee. Actual controller provisioning/usage and full terminated-attempt CPU metadata still need validation.','No training launch permitted while protective peak exceeds33 without review, even though the runtime35cap was removed by user instruction.']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ledger',type=Path,required=True);p.add_argument('--replay',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=project(json.loads(a.ledger.read_text()),json.loads(a.replay.read_text()));a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['remaining_logical_training_inputs','completed_recoverable','remaining_by_task','conservative_guard_peak_usd','nominal_guard_peak_usd','launch_gate']},indent=2))
