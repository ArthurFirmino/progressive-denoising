import numpy as np
import os
import re
import OpenImageIO as oiio
import argparse
from glob import glob
from collections import defaultdict
from tqdm.contrib.concurrent import process_map
from tabulate import tabulate
import subprocess

global args

BASE_DIR = '.'
OUTPUT_DIR = BASE_DIR + '/output'
REPORTS_DIR = BASE_DIR + '/reports'
TEST_IMAGES_DIR = BASE_DIR + '/data/interim/rt_test'
PATH_TO_FLIP_CUDA = ''  # https://github.com/NVlabs/flip

def compute_errors(scene_model):
    scene, model = scene_model

    ref_image_path = os.path.join(OUTPUT_DIR, 'rt_test', f'{scene}_{args.ref_spp:06d}spp.input.hdr.exr')
    assert os.path.exists(ref_image_path)
    ref_image = oiio.ImageBufAlgo.channels(oiio.ImageBuf(ref_image_path), ('R', 'G', 'B')).get_pixels()

    errors = defaultdict(lambda: [])
    for f in glob(OUTPUT_DIR + f'/rt_test/{scene}_*spp.{model}.hdr.exr'):

        spp = int(re.search(r'_(\d+)spp', f.split('/')[-1]).group(1))
        if spp < args.min_spp or spp > args.max_spp:
            continue
        image = oiio.ImageBuf(f).get_pixels()

        for metric in args.metrics:
            if metric == 'smape':
                error = (np.abs(image - ref_image) / (np.abs(image) + np.abs(ref_image) + 1e-2)).mean()
            elif metric == 'rmse':
                error = ((image - ref_image) ** 2)
                error = error.mean() ** 0.5
            elif metric == 'flip':
                proc = subprocess.Popen([PATH_TO_FLIP_CUDA,
                                  '-r', ref_image_path, '-t', f, '-nexm', '-nerm'],
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                proc.wait()
                out, _ = proc.communicate()
                error = float(re.search(r'Mean: ([-.0-9]+)', str(out)).group(1))
            else:
                assert False
            errors[metric].append((spp, error))

    for metric in args.metrics:
        errors[metric] = sorted(errors[metric], key=lambda a: a[0])

    return scene, model, dict(errors)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Analyze model results.')
    parser.add_argument('--models', type=str, nargs='*', default=[])
    parser.add_argument('--all-models', type=bool, default=False)
    parser.add_argument('--metrics', type=str, nargs='*', default=['rmse', 'smape'])
    parser.add_argument('--scenes', type=str, nargs='*', default=[])
    parser.add_argument('-s', '--min-spp', type=int, default=32)
    parser.add_argument('-S', '--max-spp', type=int, default=8192)
    parser.add_argument('-r', '--ref-spp', type=int, default=32768)
    args = parser.parse_args()

    if args.all_models:
        models = set()
        for f in glob(OUTPUT_DIR + f'/rt_test/*.*.hdr.exr'):
            m = re.search(r'[a-z0-9-]+\.([a-zA-Z0-9-_]+)\.hdr\.exr', f.split('/')[-1]).group(1)
            models.add(m)
        args.models = list(models)

    compute_tasks = []
    for m in args.models:
        scenes = set()
        for f in glob(OUTPUT_DIR + f'/rt_test/*.{m}.hdr.exr'):
            scenes.add(re.search(r'([a-z0-9-]+)_\d+spp', f.split('/')[-1]).group(1))
        for s in scenes:
            if len(args.scenes) == 0 or s in args.scenes:
                compute_tasks.append((s, m))

    # (scene, model, {metric:[(spp, value),...],...})
    results = process_map(compute_errors, compute_tasks, max_workers=16)

    scenes = set(t[0] for t in results)
    models = args.models

    # Compute average error of each model
    average_error = defaultdict(lambda: defaultdict(lambda: 0.))
    for _, model, errors in results:
        for metric, values in errors.items():
            average_error[metric][model] += sum(list(zip(*values))[1]) / float(len(values) * len(scenes))

    # Compute average error of each model per scene
    spps = set()
    average_error_per_scene = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.)))
    average_error_per_scene_per_spp = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.))))
    for scene, model, errors in results:
        for metric, values in errors.items():
            average_error_per_scene[metric][scene][model] = sum(list(zip(*values))[1]) / float(len(values))
            for spp, value in values:
                average_error_per_scene_per_spp[metric][scene][model][spp] = value
                spps.add(spp)
    spps = sorted(list(spps))

    # Print out pretty table
    table = []
    for model in models:
        input_features = 'HDR'
        for a, b in [('alb', 'ALB'), ('nrm', 'NRM'), ('var', 'VAR')]:
            if a in model:
                input_features += f', {b}'
        if model == 'input':
            input_features = ''
        table.append([input_features, model] + [f'{average_error[metric][model]:0.04f}' for metric in args.metrics])
    headers = ['inputs', 'model'] + [m.upper() for m in args.metrics]
    table = tabulate(table, headers=headers)
    print(table) ; print()

    # Print out pretty table of per-scene per-spp results
    for metric in args.metrics:
        for s in sorted(list(scenes)):
            table = []
            for model in models:
                table.append([model] + [f'{average_error_per_scene_per_spp[metric][s][model][spp]:0.04f}' for spp in spps])
            headers = [f'{s} -- {metric}\nmodel'] + [f'{spp}' for spp in spps]
            table = tabulate(table, headers=headers)
            print(table) ; print()

