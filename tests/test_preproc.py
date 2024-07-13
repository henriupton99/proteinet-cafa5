import subprocess
path_config = 'config.yml'

def test_preproc():
    cmd = [
        'poetry',
        'run',
        'python3',
        '-m',
        'src.proteinet',
        '--config',
        f'{path_config}',
        '--mode',
        'preproc'
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode != 1, result.stderr