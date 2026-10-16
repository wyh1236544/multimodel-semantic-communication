"""Optional LPIPS with explicit offline AlexNet/linear weights; NEVER download."""
from pathlib import Path
from docci_baseline_v14.core import read, sha


def load(spec, device):
    if spec is None:
        return None, dict(status='NOT_MEASURED', reason='No approved offline LPIPS asset specification')
    import torch
    import lpips
    from importlib.metadata import version
    c = read(spec)
    if set(c) != {'alexnet', 'alexnet_sha256', 'linear', 'linear_sha256'}:
        raise ValueError('LPIPS assets need explicit filenames AND independent known SHA256')
    files = {}
    for key in ('alexnet', 'linear'):
        p = Path(c[key])
        if not p.is_absolute():
            p = Path(spec).resolve().parent/p
        expected = c[key+'_sha256']
        if not isinstance(expected, str) or len(expected) != 64 or sha(p) != expected:
            raise ValueError('Offline LPIPS weight mismatch: '+key)
        files[key] = p
    # pretrained=False avoids LPIPS linear-weight loading; pnet_rand=True avoids
    # torchvision's automatic AlexNet download. Both are loaded explicitly below.
    metric = lpips.LPIPS(net='alex', pretrained=False, pnet_rand=True, verbose=False)
    alex = torch.load(str(files['alexnet']), map_location='cpu')
    linear = torch.load(str(files['linear']), map_location='cpu')
    feature = metric.net.state_dict()
    for name in feature:
        key = 'features.'+name.split('.', 1)[1]
        if key not in alex or alex[key].shape != feature[name].shape:
            raise ValueError('Expected complete torchvision AlexNet feature weights: '+key)
        feature[name] = alex[key]
    metric.net.load_state_dict(feature, strict=True)
    current = metric.state_dict()
    expected = {n for n in current if n.startswith(tuple('lin'+str(i)+'.' for i in range(5)))}
    if set(linear) != expected:
        raise ValueError('LPIPS learned linear calibration weights differ')
    current.update(linear)
    # LPIPS registers the same linear heads under lin0..lin4 AND lins[0..4]
    # in some releases; load both aliases, never leave a random alias to win.
    for name, value in linear.items():
        alias = 'lins.'+name[3]+name[4:]
        if alias in current:
            current[alias] = value
    metric.load_state_dict(current, strict=True)
    metric = metric.to(device).requires_grad_(False).eval()
    return metric, dict(status='MEASURED_OFFLINE', package_version=version('lpips'),
                        assets_sha256={k: c[k+'_sha256'] for k in files}, net='alex', version='0.1')
