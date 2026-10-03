from pathlib import Path
from unittest.mock import patch
import json

import pytest

from csgo_seen10.cli import parser, inference_profile, validate_prediction_identity
from csgo_seen10.config import load_config


def test_profile_and_parser_commands():
    for command in ('train', 'infer', 'eval', 'smoke', 'check', 'coverage', 'benchmark', 'compare', 'plot'):
        a = parser().parse_args([command])
        assert a.checkpoint == 'late'
    a = parser().parse_args(['infer', '--inference-engine', 'compiled', '--batch-size', '2'])
    assert inference_profile(a) == 'compiled-default-b2-vae1'


def test_eval_refuses_checkpoint_and_profile_mixing(tmp_path):
    cfg = load_config()
    args = parser().parse_args(['eval', '--task', 'discrete'])
    metadata={'smoke':False,'identity':{'protocol':{'data_sha256':'test'}}}
    marker = {'checkpoint': {'adapter_sha256': 'a','metadata':metadata,'protocol':{'data_sha256':'test'}}, 'science_sha256': cfg['identity']['science_sha256'],
              'task': 'discrete', 'engine': 'eager', 'compile_mode': None,
              'batch_size': 1, 'vae_batch_size': 1, 'selection': {'partial_debug': False}}
    pred = tmp_path / 'gen_imgs'
    pred.mkdir()
    path = tmp_path / 'prediction_identity.json'
    path.write_text(json.dumps(marker))
    with patch('csgo_seen10.checkpoint.resolve_checkpoint', return_value=tmp_path), \
         patch('csgo_seen10.checkpoint.checkpoint_metadata', return_value=metadata), \
         patch('csgo_seen10.data.protocol_identity', return_value={'data_sha256':'test'}), \
         patch('csgo_seen10.cli.sha256_file', return_value='a'):
        validate_prediction_identity(cfg,args,'discrete',pred)
        marker['selection']['partial_debug']=True
        path.write_text(json.dumps(marker))
        with pytest.raises(ValueError, match='Partial'):
            validate_prediction_identity(cfg,args,'discrete',pred)
        args.eval_smoke=True
        validate_prediction_identity(cfg,args,'discrete',pred)
        marker['batch_size']=2
        path.write_text(json.dumps(marker))
        with pytest.raises(ValueError, match='batch_size'):
            validate_prediction_identity(cfg,args,'discrete',pred)
        marker['batch_size']=1
        marker['checkpoint']['adapter_sha256']='b'
        path.write_text(json.dumps(marker))
        with pytest.raises(ValueError, match='checkpoint'):
            validate_prediction_identity(cfg,args,'discrete',pred)
        marker['checkpoint']['adapter_sha256']='a'
        marker['checkpoint']['protocol']={'data_sha256':'changed'}
        path.write_text(json.dumps(marker))
        with pytest.raises(ValueError, match='protocol'):
            validate_prediction_identity(cfg,args,'discrete',pred)


def test_eval_all_keeps_output_directories_distinct(tmp_path):
    from csgo_seen10.cli import evaluate
    cfg=load_config()
    args=parser().parse_args(['eval','--task','all','--output-root',str(tmp_path),'--device','cpu'])
    with patch('csgo_seen10.cli.validate_prediction_identity'), patch('csgo_seen10.cli.subprocess.run') as run:
        evaluate(cfg,args)
    outputs=[Path(call.args[0][call.args[0].index('--output')+1]) for call in run.call_args_list]
    assert outputs == [tmp_path/'discrete',tmp_path/'continuous']


def test_asset_check_passes_resolved_cli_paths(tmp_path):
    from csgo_seen10.cli import check
    overrides={n:str(tmp_path/n) for n in ('base_checkpoint','gemma_path','tokenizer_path','vae_path')}
    cfg=load_config(overrides=overrides)
    with patch('csgo_seen10.checks.check_data', return_value={}), \
         patch('csgo_seen10.cli.gpu_resource_check', return_value={'available':False}), \
         patch('csgo_seen10.cli.subprocess.run') as run:
        run.return_value.returncode=0
        check(cfg,tmp_path/'check')
    command=run.call_args.args[0]
    for name,value in overrides.items():
        assert command[command.index('--'+name.replace('_','-'))+1] == value


def test_benchmark_partial_cannot_pollute_formal_run(tmp_path):
    from csgo_seen10.cli import require_isolated_output
    cfg=load_config(overrides={'run_root':str(tmp_path/'formal')})
    with pytest.raises(ValueError,match='outside'):
        require_isolated_output(cfg,tmp_path/'formal/predictions/late/new_profile/discrete')
    assert require_isolated_output(cfg,tmp_path/'benchmark') == tmp_path/'benchmark'


def test_smoke_failure_stops_before_gpu_training(tmp_path):
    from csgo_seen10.cli import smoke
    cfg = load_config(overrides={'run_root': str(tmp_path/'smoke')})
    args = parser().parse_args(['smoke', '--run-root', cfg['paths']['run_root']])
    with patch('csgo_seen10.cli.check'), \
         patch('csgo_seen10.cli.subprocess.run') as run, \
         patch('csgo_seen10.checks.evaluator_fixture_smoke', return_value={'discrete': {'returncode': 0}}), \
         patch('csgo_seen10.cli.gpu_resource_check', return_value={'available': True}), \
         patch('csgo_seen10.training.run_training') as train:
        run.return_value.returncode = 1
        with pytest.raises(RuntimeError, match='Smoke check failed'):
            smoke(cfg, args)
        train.assert_not_called()
    assert json.loads((tmp_path/'smoke/smoke_summary.json').read_text())['gpu'] == 'untested'


def test_smoke_rejects_formal_experiment_tree():
    from csgo_seen10.cli import smoke
    from csgo_seen10.config import PROJECT_ROOT
    from csgo_seen10.training import run_training
    cfg = load_config()
    root = PROJECT_ROOT/'outputs'/cfg['experiment']/'smoke_nested'
    cfg['paths']['run_root'] = str(root)
    args = parser().parse_args(['smoke', '--run-root', str(root)])
    with pytest.raises(ValueError, match='outside'):
        smoke(cfg, args)
    with patch('csgo_seen10.training.distributed_context', return_value=(0, 1, __import__('torch').device('cpu'))):
        with pytest.raises(ValueError, match='outside'):
            run_training(cfg, micro_batch_size=1, smoke=True)


def test_code_identity_keeps_path_only_config_migration_valid(tmp_path):
    import csgo_seen10.config as config
    (tmp_path/'configs').mkdir()
    path = tmp_path/'configs'/f'{config.EXPERIMENT}.json'
    payload = {'experiment': config.EXPERIMENT, 'seed': 42, 'paths': {'data_root': '/old/data'}}
    path.write_text(json.dumps(payload))
    with patch.object(config, 'PROJECT_ROOT', tmp_path), patch.object(config.subprocess, 'check_output', return_value='upstream\n'):
        before = config.code_identity()
        payload['paths']['data_root'] = '/new/server/data'
        path.write_text(json.dumps(payload))
        assert config.code_identity() == before
        payload['seed'] = 43
        path.write_text(json.dumps(payload))
        assert config.code_identity() != before
