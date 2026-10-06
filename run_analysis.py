#!/usr/bin/env python3
"""Run a retained script with explicit roots and author dependencies."""
import argparse
import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true', help='Check prerequisites without executing analysis')
    p.add_argument('script', help='Relative path to a retained .py or .R script')
    p.add_argument('arguments', nargs=argparse.REMAINDER)
    a = p.parse_args()
    script = (ROOT / a.script).resolve()
    if ROOT not in script.parents or not script.is_file() or script.suffix not in {'.py', '.R'}:
        p.error('Select a retained .py or .R script inside this repository.')
    missing = []
    for key in ('PROJECT_ROOT', 'LEGACY_DATA_ROOT', 'HOME_ROOT'):
        value = os.environ.get(key, '')
        if not value or not Path(value).is_absolute() or not Path(value).is_dir():
            missing.append(f'{key}: set an existing absolute data directory')
    text = script.read_text()
    env = os.environ.copy()
    dirs = [str(d) for d in (ROOT / 'analysis').rglob('*') if d.is_dir()]
    if script.suffix == '.py':
        tree = ast.parse(text)
        needs_common = any(isinstance(n, ast.ImportFrom) and n.module == 'common' for n in ast.walk(tree))
        if needs_common:
            helper = Path(env.get('AUTHOR_COMMON_DIR') or (ROOT / 'src'))
            if not (helper / 'common.py').is_file():
                missing.append('common.py: bundled module or AUTHOR_COMMON_DIR override is required')
        dirs.insert(0, str(ROOT / 'src'))
        if env.get('AUTHOR_COMMON_DIR'):
            dirs.insert(0, env['AUTHOR_COMMON_DIR'])
        env['PYTHONPATH'] = os.pathsep.join([str(script.parent), *filter(None, dirs), str(ROOT / 'src'), env.get('PYTHONPATH', '')])
        command = [sys.executable, str(script)]
    else:
        rscript = shutil.which('Rscript')
        if not rscript:
            missing.append('Rscript: install the required R environment')
        for key in ('SCISSOR_HELPER_R', 'SCHARD_FUNCTIONS_R', 'SCHARD_H5AD_R'):
            if f'Sys.getenv("{key}")' in text and not Path(env.get(key, '__missing__')).is_file():
                missing.append(f'{key}: supply the missing author R helper')
        if 'BULK_METADATA_ROOT' in text and not Path(env.get('BULK_METADATA_ROOT', '__missing__')).is_dir():
            missing.append('BULK_METADATA_ROOT: supply the metadata working directory')
        command = [rscript or 'Rscript', str(script)]
    if missing:
        print('Prerequisites not satisfied:\n' + '\n'.join('- ' + m for m in missing), file=sys.stderr)
        return 2
    if a.check:
        print('Entry-point prerequisites satisfied; data files and scientific results have not been validated.')
        return 0
    return subprocess.call(command + a.arguments, env=env, cwd=ROOT)

if __name__ == '__main__':
    raise SystemExit(main())
