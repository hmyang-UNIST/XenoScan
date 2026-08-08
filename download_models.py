#!/usr/bin/env python
"""Fetch the trained XenoScan model weights from Zenodo into the stage directories.

The weights are 229 MB and are not stored in this repository.  They are archived on Zenodo as
a single tar archive and unpacked on demand:

    python download_models.py              # fetch and install anything missing
    python download_models.py --force      # re-download and overwrite
    python download_models.py --check      # report what is present, download nothing

The archive is verified against the MD5 that Zenodo publishes for it before anything is
unpacked.  Requires only `requests`.
"""
import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile

try:
    import requests
except ImportError:
    sys.exit('This script needs `requests`:  pip install requests')

# Zenodo record holding the image dataset and the trained weights.
# https://doi.org/10.5281/zenodo.21845686
ZENODO_RECORD = '21845686'
ARCHIVE = 'XenoScan-models.tar'

HERE = os.path.dirname(os.path.abspath(__file__))

# name inside the archive's models/ directory -> destination in this repository
TARGETS = {
    'UNET_XENOPUS_832x512.h5': '2_segmentation/models/UNET_XENOPUS_832x512.h5',
    'UNET_XENOPUS_208x128.h5': '2_segmentation/models/UNET_XENOPUS_208x128.h5',
    'predict_hpf.h5':          '4_hpf_prediction/models/predict_hpf.h5',
    'resnet18_TB-01.h5':       '5_classification/models/resnet18_TB-01.h5',
    'resnet18_TB-02.h5':       '5_classification/models/resnet18_TB-02.h5',
    'resnet18_TB-03.h5':       '5_classification/models/resnet18_TB-03.h5',
    'resnet18_TB-04.h5':       '5_classification/models/resnet18_TB-04.h5',
}


def md5_file(path, bufsize=8 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(bufsize), b''):
            h.update(chunk)
    return h.hexdigest()


def archive_info():
    r = requests.get(f'https://zenodo.org/api/records/{ZENODO_RECORD}', timeout=60)
    r.raise_for_status()
    for f in r.json().get('files', []):
        if (f.get('key') or f.get('filename')) == ARCHIVE:
            links = f.get('links') or {}
            return {'link': links.get('self') or links.get('download'),
                    'md5': (f.get('checksum') or '').replace('md5:', ''),
                    'size': f.get('size') or f.get('filesize')}
    sys.exit(f'{ARCHIVE} not found in Zenodo record {ZENODO_RECORD}')


def download(url, dest, expect_md5):
    with requests.get(url, stream=True, timeout=(30, 900)) as r:
        r.raise_for_status()
        total = int(r.headers.get('Content-Length', 0))
        done = 0
        with open(dest, 'wb') as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f'\r    {done/1e6:7.1f}/{total/1e6:.1f} MB  '
                          f'{100*done/total:5.1f}%', end='', flush=True)
    print()
    if expect_md5:
        got = md5_file(dest)
        if got != expect_md5:
            raise SystemExit(f'checksum mismatch: {got} != {expect_md5}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true', help='re-download and overwrite')
    ap.add_argument('--check', action='store_true', help='report status only')
    args = ap.parse_args()

    if args.check:
        for name, rel in sorted(TARGETS.items()):
            p = os.path.join(HERE, rel)
            print(f'  present  {rel}  ({os.path.getsize(p)/1e6:.1f} MB)' if os.path.exists(p)
                  else f'  MISSING  {rel}')
        return

    missing = [n for n, rel in TARGETS.items() if not os.path.exists(os.path.join(HERE, rel))]
    if not missing and not args.force:
        print('all weights already present. Nothing to do '
              '(use --force to re-download, --check to verify).')
        return

    info = archive_info()
    need = len(TARGETS) if args.force else len(missing)
    print(f'{ARCHIVE}  ({info["size"]/1e6:.1f} MB)  -> {need} file(s) needed')

    with tempfile.TemporaryDirectory() as tmp:
        tarpath = os.path.join(tmp, ARCHIVE)
        download(info['link'], tarpath, info['md5'])
        with tarfile.open(tarpath) as tf:
            for name, rel in sorted(TARGETS.items()):
                dest = os.path.join(HERE, rel)
                if os.path.exists(dest) and not args.force:
                    print(f'{rel}: already present, skipping')
                    continue
                member = tf.extractfile(f'models/{name}')
                if member is None:
                    sys.exit(f'{name} not found inside {ARCHIVE}')
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, 'wb') as out:
                    shutil.copyfileobj(member, out)
                print(f'{rel}: installed ({os.path.getsize(dest)/1e6:.1f} MB)')

    print('\nall weights present. Verify any time with:  python download_models.py --check')


if __name__ == '__main__':
    main()
