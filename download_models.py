#!/usr/bin/env python
"""Fetch the trained XenoScan model weights from Zenodo into the stage directories.

The weights are 219.5 MB and are not stored in this repository.  They are archived on
Zenodo and downloaded on demand:

    python download_models.py              # fetch everything that is missing
    python download_models.py --force      # re-download even if present
    python download_models.py --check      # verify what is present, download nothing

Each file is verified against the MD5 that Zenodo publishes for it.
Requires only `requests`.
"""
import argparse
import hashlib
import os
import sys

try:
    import requests
except ImportError:
    sys.exit('This script needs `requests`:  pip install requests')

# Zenodo record holding the trained weights (same record as the image dataset).
# https://doi.org/10.5281/zenodo.21845686
ZENODO_RECORD = '21845686'

HERE = os.path.dirname(os.path.abspath(__file__))

# Zenodo filename -> destination inside this repository
TARGETS = {
    'UNET_XENOPUS_832x512.h5': '2_segmentation/models/UNET_XENOPUS_832x512.h5',
    'UNET_XENOPUS_208x128.h5': '2_segmentation/models/UNET_XENOPUS_208x128.h5',
    '__predict_HPF_MAE.h5':    '4_hpf_prediction/models/__predict_HPF_MAE.h5',
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


def record_files():
    url = f'https://zenodo.org/api/records/{ZENODO_RECORD}'
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out = {}
    for f in r.json().get('files', []):
        name = f.get('key') or f.get('filename')
        link = (f.get('links') or {}).get('self') or (f.get('links') or {}).get('download')
        out[name] = {'link': link,
                     'md5': (f.get('checksum') or '').replace('md5:', ''),
                     'size': f.get('size') or f.get('filesize')}
    return out


def download(url, dest, expect_md5=None):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + '.part'
    with requests.get(url, stream=True, timeout=(30, 600)) as r:
        r.raise_for_status()
        total = int(r.headers.get('Content-Length', 0))
        done = 0
        with open(tmp, 'wb') as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f'\r    {done/1e6:7.1f}/{total/1e6:.1f} MB  {pct:5.1f}%',
                          end='', flush=True)
    print()
    if expect_md5:
        got = md5_file(tmp)
        if got != expect_md5:
            os.remove(tmp)
            raise SystemExit(f'checksum mismatch for {dest}: {got} != {expect_md5}')
    os.replace(tmp, dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true', help='re-download files already present')
    ap.add_argument('--check', action='store_true', help='report status only')
    args = ap.parse_args()

    if args.check:
        for name, rel in sorted(TARGETS.items()):
            p = os.path.join(HERE, rel)
            if os.path.exists(p):
                print(f'  present  {rel}  ({os.path.getsize(p)/1e6:.1f} MB)')
            else:
                print(f'  MISSING  {rel}')
        return

    remote = record_files()
    missing = [n for n in TARGETS if n not in remote]
    if missing:
        sys.exit(f'not found in Zenodo record {ZENODO_RECORD}: {", ".join(sorted(missing))}')

    for name, rel in sorted(TARGETS.items()):
        dest = os.path.join(HERE, rel)
        if os.path.exists(dest) and not args.force:
            print(f'{rel}: already present, skipping')
            continue
        info = remote[name]
        print(f'{rel}  ({info["size"]/1e6:.1f} MB)')
        download(info['link'], dest, info['md5'])

    print('\nall weights present. Verify any time with:  python download_models.py --check')


if __name__ == '__main__':
    main()
