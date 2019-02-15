"""Code for printing/plotting results for the HPatch evaluation protocols.

Usage:
  hpatches_results.py (-h | --help)
  hpatches_results.py --version
  hpatches_results.py --descr-name=<>... --task=<>... [--results-dir=<>] [--split=<>] [--pcapl=<>] [--more_info]

Options:
  -h --help         Show this screen.
  --version         Show version.
  --descr-name=<>   Descriptor name e.g. --descr=sift.
  --results-dir=<>  Results root folder. [default: results]
  --task=<>         Task name. Valid tasks are {verification,matching,retrieval}.
  --split=<>        Split name. Valid are {a,b,c,full,illum,view}. [default: a]
  --pcapl=<>        Show results for pca-power law descr. [default: no]
  --more_info       Outputs partial scores, not only final mAP
For more visit: https://github.com/hpatches/
"""
from utils.tasks import tskdir
from utils.results import *
import os.path
import time
import json
from utils.docopt import docopt

if __name__ == '__main__':
    opts = docopt(__doc__, version='HPatches 1.0')
    descrs = opts['--descr-name']
    with open(os.path.join(tskdir, "splits", "splits.json")) as f:
        splits = json.load(f)
    splt = splits[opts['--split']]
    for t in opts['--task']:
        print("%s task results:" % (green(t.capitalize())))
        for desc in descrs:
            results_methods[t](desc,splt,opts['--more_info'])
            if opts['--pcapl']!='no':
                results_methods[t](desc+'_pcapl',splt,opts['--more_info'])
            print
        print
