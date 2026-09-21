"""Entry point embedded in the additive zipapp. Run in multimodel, never overwrite."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile


def install(archive, root):
    root = Path(root).resolve()
    if root.name != 'multimodel' or not (root / 'run_v10.sh').is_file():
        raise ValueError('Run from the existing multimodel project with V10 installed')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or len(names) > 30 or sum(i.file_size for i in z.infolist()) > 2000000:
            raise ValueError('Invalid archive size or duplicate entries')
        manifest = json.loads(z.read('MANIFEST.json'))
        if set(names) != set(manifest) | {'__main__.py', 'MANIFEST.json'}:
            raise ValueError('Unexpected archive members')
        payload = []
        for name, expected in manifest.items():
            p = PurePosixPath(name)
            if (not name.startswith('v10_extension/common_audit/') or p.is_absolute()
                    or '..' in p.parts or '\\' in name or ':' in name or str(p) != name):
                raise ValueError('Unsafe member: ' + name)
            target = root.joinpath(*p.parts)
            if any(x.is_symlink() for x in [target] + list(target.parents)):
                raise ValueError('Symlink target refused')
            data = z.read(name)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError('Checksum mismatch: ' + name)
            if target.exists() and (not target.is_file() or target.read_bytes() != data):
                raise ValueError('Existing different file; no files written: ' + name)
            payload.append((target, data))
        # All collisions checked before first write. Identical reinstall is harmless.
        added = 0
        for target, data in payload:
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as f:
                    f.write(data)
                added += 1
        return added


if __name__ == '__main__':
    print('Installed new files:', install(sys.argv[0], Path.cwd()))
