"""Exercise the real relinker in an owned temporary R3 subdirectory."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from .pipeline import ManifestDataset


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(root):
    root = root.resolve()
    probe = root / '07_audit' / 'archive_relocation_probe'
    if probe.exists():
        raise FileExistsError('Probe exists; retain prior evidence rather than overwrite')
    source = root / '06_submission' / 'relink_frozen_inputs.py'
    spec = importlib.util.spec_from_file_location('r3_relinker', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = root / '06_submission' / 'frozen_inputs.zip'
    index_path = root / '06_submission' / 'frozen_inputs_index.json'
    index = json.loads(index_path.read_text(encoding='utf-8'))
    before = {m['original_absolute']: digest(m['original_absolute']) for m in index['manifests']}
    module.relocate(archive, index_path, probe)
    relocation = json.loads((probe / 'RELOCATION_REPORT.json').read_text())
    checks = []
    for m in index['manifests']:
        original_path = Path(m['original_absolute'])
        relocated_path = probe / '02_protocol' / m['track'] / f"{m['split']}.jsonl"
        dataset = ManifestDataset(relocated_path)
        original = [json.loads(l) for l in original_path.read_text(encoding='utf-8-sig').splitlines()]
        changed_fields = {'npz_path', 'npz_path_absolute', 'npz_path_relative_workspace'}
        for old, new in zip(original, dataset.records):
            for field, value in old.items():
                if field not in changed_fields and new[field] != value:
                    raise ValueError('Semantic manifest field changed: ' + field)
        if len(original) != len(dataset) or dataset.hash == m['sha256']:
            raise ValueError('Relocated count/hash contract violated')
        # List-path aliases must reference the same newly extracted input files.
        listed = (probe / '02_protocol' / m['track'] / 'lists' / f"{m['split']}.txt").read_text().splitlines()
        if listed != [r['npz_path'] for r in dataset.records]:
            raise ValueError('Relocated list/manifest paths differ')
        checks.append({'track': m['track'], 'split': m['split'], 'tasks': len(dataset),
                       'original_manifest_sha256': m['sha256'], 'relocated_manifest_sha256': dataset.hash,
                       'npz_source_hashes_checked_by_training_loader': True,
                       'all_nonpath_original_fields_preserved': True})
    if {p: digest(p) for p in before} != before:
        raise ValueError('Original manifest modified')
    report = {'passed': True, 'archive_sha256': digest(archive), 'index_sha256': digest(index_path),
              'tasks_checked': sum(c['tasks'] for c in checks), 'checks': checks,
              'new_protocol_sha256': relocation['new_rerun_protocol_sha256'],
              'original_protocol_sha256': relocation['original_protocol_sha256'],
              'original_manifests_unchanged': True,
              'probe_extraction_is_not_a_formal_training_run': True,
              'temporary_npz_copies_removed_after_successful_verification': True}
    destination = root / '07_audit' / 'archive_relocation_verification.json'
    destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    # Delete only the subdirectory that this invocation created, after verifying
    # its exact resolved location and its generated ownership/provenance report.
    if probe.resolve().parent != (root / '07_audit').resolve() or not (probe / 'RELOCATION_REPORT.json').is_file():
        raise ValueError('Refuse cleanup outside owned probe directory')
    shutil.rmtree(probe)
    print(json.dumps({'passed': True, 'tasks_checked': report['tasks_checked'], 'artifact': str(destination)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    main(args.root)
