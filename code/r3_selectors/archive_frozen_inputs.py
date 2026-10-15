"""Archive original NPZ bytes referenced by frozen manifests, with hash dedup."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


TRACKS = ['scale300', 'hard_confusable']
SPLITS = ['train', 'val', 'test']


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write_member(archive, name, data):
    info = zipfile.ZipInfo(name, date_time=(2026, 10, 2, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, data)


def build(root):
    dest = root / '06_submission'
    dest.mkdir(parents=True, exist_ok=True)
    archive_path = dest / 'frozen_inputs.zip'
    index_path = dest / 'frozen_inputs_index.json'
    if archive_path.exists() or index_path.exists():
        raise FileExistsError('Frozen input archive/index already exists; refuse overwrite')
    records = []
    manifests = []
    objects = {}
    counts = {}
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for track in TRACKS:
            counts[track] = {}
            for split in SPLITS:
                path = root / '02_protocol' / track / f'{split}.jsonl'
                raw_manifest = path.read_bytes()
                manifest_member = f'original_manifests/{track}/{split}.jsonl'
                write_member(archive, manifest_member, raw_manifest)
                samples = [json.loads(line) for line in raw_manifest.decode('utf-8-sig').splitlines() if line]
                counts[track][split] = len(samples)
                manifests.append({'track': track, 'split': split,
                                  'original_absolute': str(path.resolve()),
                                  'original_relative_r3': path.relative_to(root).as_posix(),
                                  'zip_member': manifest_member, 'sha256': sha_bytes(raw_manifest),
                                  'tasks': len(samples)})
                for sample in samples:
                    source = Path(sample['npz_path'])
                    raw = source.read_bytes()
                    digest = sha_bytes(raw)
                    if digest != sample['source_npz_sha256']:
                        raise ValueError('Source NPZ differs from frozen manifest: ' + str(source))
                    logical_member = f'frozen_inputs/{track}/{split}/{source.name}'
                    if digest not in objects:
                        objects[digest] = {'zip_member': logical_member, 'bytes': len(raw)}
                        write_member(archive, logical_member, raw)
                    member = objects[digest]['zip_member']
                    records.append({'sample_id': sample['sample_id'], 'track': track, 'split': split,
                                    'scene_id': sample['scene_id'], 'original_split': sample['original_split'],
                                    'original_absolute': sample['npz_path'],
                                    'original_relative_workspace': sample['npz_path_relative_workspace'],
                                    'logical_archive_path': logical_member, 'zip_member': member,
                                    'source_sha256': digest, 'bytes': len(raw),
                                    'deduplicated_alias': member != logical_member,
                                    'group_id': sample['group_id'],
                                    'canonical_graph_sha256': sample['canonical_graph_sha256'],
                                    'target_index': sample['target_index']})
        protocol = root / '02_protocol' / 'BASELINE_PROTOCOL.md'
        protocol_bytes = protocol.read_bytes()
        protocol_member = 'original_registration/BASELINE_PROTOCOL.md'
        write_member(archive, protocol_member, protocol_bytes)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC check failed')
        for digest, obj in objects.items():
            if sha_bytes(archive.read(obj['zip_member'])) != digest:
                raise ValueError('Archived byte SHA differs: ' + obj['zip_member'])
        for manifest in manifests:
            if sha_bytes(archive.read(manifest['zip_member'])) != manifest['sha256']:
                raise ValueError('Archived manifest SHA differs')
    index = {'format_version': 1,
             'scope': 'Reproduction input data attachment; separate from manuscript source ZIP. Original NPZ bytes only; no regenerated arrays.',
             'archive_filename': archive_path.name, 'archive_sha256': sha_bytes(archive_path.read_bytes()),
             'archive_bytes': archive_path.stat().st_size, 'counts': counts,
             'task_records': len(records), 'unique_npz_byte_objects': len(objects),
             'deduplication': 'Exact full NPZ byte SHA256, not graph hash; index retains every task/group/path mapping.',
             'original_registration': {'zip_member': protocol_member, 'sha256': sha_bytes(protocol_bytes)},
             'manifests': manifests, 'objects_by_sha256': objects, 'records': records,
             'verified': {'all_source_npz_hashes': True, 'zip_crc': True, 'all_archived_npz_hashes': True,
                          'archived_original_manifest_hashes': True},
             'relocation': 'Relink only into a new root; preserve original manifests and original run configurations. Relocated manifests receive new hashes and a distinct rerun protocol.'}
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    sums = dest / 'frozen_inputs_SHA256SUMS.txt'
    sums.write_text(index['archive_sha256'] + '  frozen_inputs.zip\n' + sha_bytes(index_path.read_bytes()) +
                    '  frozen_inputs_index.json\n', encoding='utf-8')
    print(json.dumps({'archive': str(archive_path), 'bytes': index['archive_bytes'],
                      'records': len(records), 'unique_npz': len(objects),
                      'sha256': index['archive_sha256'], 'verified': index['verified']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    build(args.root.resolve())
