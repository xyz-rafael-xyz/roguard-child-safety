"""Rebase an approved portable inventory; never approve observed local bytes.

No models or artifact code are imported. The caller authenticates the approved
inventory digest, then instantiates the existing installed Closure with the
returned path/digest and checks its identity against receipt_sha256.
"""
from collections import namedtuple
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import unicodedata

FORMAT = 'roguard-portable-runtime-closure-1'
PACKAGES = {'roguard', 'mlx_lm', 'mlx', 'numpy', 'transformers',
            'tokenizers', 'huggingface_hub', 'safetensors'}
NATIVE_SUFFIXES = {'.so', '.dylib', '.pyd', '.metallib'}
BASE = {'README.md', 'chat_template.jinja', 'config.json',
        'generation_config.json', 'model.safetensors',
        'model.safetensors.index.json', 'processor_config.json',
        'tokenizer.json', 'tokenizer_config.json'}
KINDS = {
    'native_response': ('roguard-native-response-installed-closure-1',
                        'roguard-native-response-installed-receipt-1'),
    'native_tail': ('roguard-native-tail-installed-closure-1',
                    'roguard-native-tail-closure-receipt-1'),
    'structured': ('roguard-structured-installed-closure-2',
                   'roguard-verified-structured-closure-receipt-2'),
    'response': ('roguard-response-offset-installed-closure-1',
                 'roguard-response-offset-closure-receipt-1'),
}
HELPERS = {'runtime_source': 'response_offset/runtime.py',
           'closure_source': 'response_offset/closure.py',
           'lexical_source': 'response_offset/lexical.py'}
Binding = namedtuple('Binding', 'path sha256 receipt_sha256 kind')


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _unique(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, 'Duplicate inventory key')
        value[key] = item
    return value


def _sha(value):
    require(type(value) is str and re.fullmatch('[a-f0-9]{64}', value),
            'Expected lowercase SHA256')
    return value


def _relative(value):
    require(type(value) is str and value and unicodedata.normalize('NFC', value) == value
            and '\\' not in value and ':' not in value
            and not value.startswith('/')
            and all(part not in ('', '.', '..') for part in value.split('/')),
            'Expected normalized relative inventory path')
    return PurePosixPath(value)


def _record(value):
    require(type(value) is dict and set(value) == {'bytes', 'sha256'}
            and type(value['bytes']) is int and value['bytes'] >= 0,
            'Invalid approved file record')
    _sha(value['sha256'])


def _asset_ids(kind):
    base = {'base:' + name for name in BASE}
    if kind == 'native_response':
        return base | {'manifest', 'native_spec', 'native_adapter', 'reference_spec', 'reference_weights'}
    if kind == 'native_tail':
        return base | {'selector'} | {lang + '_' + suffix
            for lang in ('ro', 'uk') for suffix in
            ('manifest', 'native_spec', 'native_adapter', 'spec9', 'adapter9')}
    if kind == 'structured':
        return base | {'selector'} | {lang + '_' + suffix
            for lang in ('ro', 'uk') for suffix in
            ('spec75', 'spec9', 'weights75', 'adapter9',
             'residual_manifest', 'residual_tensors')}
    return base | set(HELPERS) | {'response59_spec', 'response59_training_manifest',
        'response59_weights', 'offset_manifest', 'offset_theta', 'offset_lexicon',
        'reference_spec', 'reference_weights'}


def _validate(value):
    require(type(value) is dict and set(value) == {'format', 'kind', 'packages',
        'assets', 'native_dependencies', 'asset_edges',
        'declared_supported_runtime_contract', 'expected_receipt_sha256'}
        and value['format'] == FORMAT and value['kind'] in KINDS,
        'Unsupported portable closure schema')
    _sha(value['expected_receipt_sha256'])
    require(type(value['packages']) is dict and set(value['packages']) == PACKAGES,
            'Exact supported package inventory required')
    for package in value['packages'].values():
        require(type(package) is dict and set(package) == {'distribution', 'version', 'files'}
                and type(package['distribution']) is str and package['distribution']
                and type(package['version']) is str and package['version']
                and type(package['files']) is dict and package['files'],
                'Invalid approved package')
        for relative, record in package['files'].items():
            require(_relative(relative).suffix == '.py', 'Python inventory entry required')
            _record(record)
    require(type(value['assets']) is dict
            and set(value['assets']) == _asset_ids(value['kind']),
            'Exact runtime assets required; no research targets')
    for name, entry in value['assets'].items():
        require(type(entry) is dict and set(entry) == {'location', 'bytes', 'sha256'},
                'Invalid runtime asset')
        _record({key: entry[key] for key in ('bytes', 'sha256')})
        loc = entry['location']
        require(type(loc) is dict and set(loc) == {'root', 'relative_path'},
                'Invalid relative asset location')
        _relative(loc['relative_path'])
        if value['kind'] == 'response' and name in HELPERS:
            require(loc == {'root': 'package:roguard', 'relative_path': HELPERS[name]},
                    'Response helpers must come from installed package')
            require({key: entry[key] for key in ('bytes', 'sha256')} ==
                    value['packages']['roguard']['files'].get(HELPERS[name]),
                    'Helper differs from approved package source')
        else:
            require(loc['root'] == 'bundle', 'Runtime asset must be bundle-relative')
    require(type(value['native_dependencies']) is dict and value['native_dependencies'],
            'Explicit native dependency inventory required')
    native_locations = set()
    for entry in value['native_dependencies'].values():
        require(type(entry) is dict and set(entry) == {'package', 'relative_path', 'bytes', 'sha256'}
                and entry['package'] in PACKAGES, 'Invalid native dependency location')
        require(_relative(entry['relative_path']).suffix in NATIVE_SUFFIXES,
                'Native dependency must be a declared native library')
        location = (entry['package'], entry['relative_path'])
        require(location not in native_locations, 'Duplicate native dependency location')
        native_locations.add(location)
        _record({key: entry[key] for key in ('bytes', 'sha256')})
    require(type(value['asset_edges']) is list, 'Explicit asset edges required')
    for edge in value['asset_edges']:
        require(type(edge) is dict and set(edge) == {'from', 'keys', 'to'}
                and edge['from'] in value['assets'] and edge['to'] in value['assets']
                and type(edge['keys']) is list and edge['keys']
                and all(type(key) is str and key for key in edge['keys']),
                'Invalid finite asset digest edge')
    require(type(value['declared_supported_runtime_contract']) is dict,
            'Explicit supported runtime contract required')
    canonical(value)  # Reject non-JSON/NaN metadata before target access.


def receipt(registration, kind):
    """Exact existing Closure._receipt shape; no host paths or new identity rules."""
    return dict(format=KINDS[kind][1], scope='installed_candidate',
        research_provenance_verification='declared_digests_only_external_verification_required',
        packages={name: {key: value[key] for key in ('files', 'distribution', 'version')}
                  for name, value in registration['packages'].items()},
        assets={name: {key: value[key] for key in ('bytes', 'sha256')}
                for name, value in registration['assets'].items()},
        asset_edges=registration['asset_edges'],
        native_dependencies={name: {key: value[key] for key in ('bytes', 'sha256')}
                             for name, value in registration['native_dependencies'].items()},
        declared_supported_runtime_contract=registration['declared_supported_runtime_contract'])


def _contained(root, relative, *, symlinks=False):
    path = root.joinpath(*_relative(relative).parts)
    if not symlinks:
        current = root
        for part in _relative(relative).parts:
            current = current / part
            require(not current.is_symlink(), 'Package/native symlink is not admitted')
    resolved = path.resolve(strict=True)
    require(resolved.is_relative_to(root) and resolved.is_file(),
            'File escapes its declared root or is not regular')
    return resolved


def _check_file(path, expected):
    require(path.stat().st_size == expected['bytes'], 'Approved file size differs')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    require(h.hexdigest() == expected['sha256'], 'Approved file digest differs')


def _roots(approved):
    roots = {}
    for name, package in approved['packages'].items():
        found = importlib.util.find_spec(name)  # Only fixed top-level package names.
        require(found is not None and found.submodule_search_locations is not None
                and len(found.submodule_search_locations) == 1,
                'Expected one installed package root')
        raw = Path(next(iter(found.submodule_search_locations)))
        require(not raw.is_symlink(), 'Linked package root is not admitted')
        root = raw.resolve(strict=True)
        require(raw == root, 'Package root must be an absolute resolved path')
        distribution = importlib.metadata.distribution(package['distribution'])
        require(distribution.version == package['version']
                and root.is_relative_to(Path(distribution.locate_file('')).resolve()),
                'Installed distribution identity/root differs')
        roots[name] = root
    require(not any(a.is_relative_to(b) for x, a in roots.items()
                    for y, b in roots.items() if x != y),
            'Installed package roots overlap')
    return roots


def _verify(registration, roots, bundle):
    observed_native = set()
    for name, package in registration['packages'].items():
        root = roots[name]
        require(not root.is_symlink() and root.resolve(strict=True) == root
                and root.is_dir(), 'Installed package root changed before traversal')
        observed = set()
        for path in root.rglob('*'):
            require(not path.is_symlink(), 'Package symlink is not admitted')
            if path.is_file() and path.suffix == '.py':
                observed.add(path.relative_to(root).as_posix())
            if path.is_file() and path.suffix in NATIVE_SUFFIXES:
                # Physical package ownership matters: mlx-metal installs its
                # dylibs/Metal library under mlx, outside mlx's own RECORD.
                observed_native.add(path)
        require(observed == set(package['files']), 'Installed Python inventory differs')
        for relative, expected in package['files'].items():
            _check_file(_contained(root, relative), expected)
        for modname, module in list(sys.modules.items()):
            if modname == name or modname.startswith(name + '.'):
                origin = getattr(module, '__file__', None)
                if origin is None:
                    require({Path(p).resolve() for p in getattr(module, '__path__', [])} == {root},
                            'Loaded namespace origin differs')
                else:
                    path = Path(origin).resolve()
                    require(path.is_relative_to(root), 'Loaded module outside installed root')
                    require(path.relative_to(root).as_posix() in package['files'] or any(
                        Path(v['path']) == path for v in registration['native_dependencies'].values()),
                        'Uninventoried loaded module')
    declared_native = [Path(entry['path']) for entry in
                       registration['native_dependencies'].values()]
    require(len(declared_native) == len(set(declared_native))
            and observed_native == set(declared_native),
            'Installed native inventory differs')
    for group in ('assets', 'native_dependencies'):
        for entry in registration[group].values():
            path = Path(entry['path'])
            allowed = list(roots.values()) + ([bundle] if group == 'assets' else [])
            owners = [root for root in allowed if path.is_relative_to(root)]
            require(owners, 'Registered file has no declared root')
            # Recheck the resolved binding at every barrier. A post-preflight
            # symlink replacement must not cause outside bytes to be opened.
            stable = _contained(max(owners, key=lambda p: len(p.parts)),
                                path.relative_to(max(owners, key=lambda p: len(p.parts))).as_posix())
            require(stable == path, 'Registered resolved file location changed')
            _check_file(stable, entry)
    for edge in registration['asset_edges']:
        path = Path(registration['assets'][edge['from']]['path'])
        require(path.stat().st_size <= 4_000_000, 'Digest-edge JSON exceeds bound')
        value = json.loads(path.read_bytes(), object_pairs_hook=_unique)
        for key in edge['keys']:
            value = value[key]
        require(type(value) is str and value == registration['assets'][edge['to']]['sha256'],
                'Transitive runtime digest differs')


def rebase_closure(approved_path, *, expected_sha256, bundle_root, output_path):
    """Write one exclusive local registration from a trusted portable inventory.

    ``expected_sha256`` must come from the approved bundle/profile, never from a
    freshly inventoried local environment. No model is loaded by this function.
    The loader must instantiate the existing kind-specific Closure and compare
    its identity with Binding.receipt_sha256 before constructing a model.
    """
    _sha(expected_sha256)
    source = Path(approved_path)
    require(source.stat().st_size <= 16_000_000, 'Approved inventory exceeds bound')
    raw = source.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, 'Approved inventory digest differs')
    approved = json.loads(raw, object_pairs_hook=_unique)
    _validate(approved)  # All declarations checked before resolving any target.
    roots = _roots(approved)
    bundle = Path(bundle_root).resolve(strict=True)
    require(bundle.is_dir(), 'Bundle root must be a directory')
    registration = dict(format=KINDS[approved['kind']][0], qualified=False,
        scope='installed_candidate', registered=True,
        packages={name: dict(value, root=str(roots[name])) for name, value in approved['packages'].items()},
        assets={}, native_dependencies={}, asset_edges=approved['asset_edges'],
        required_asset_ids=sorted(approved['assets']), required_runtime_assets=sorted(approved['assets']),
        declared_supported_runtime_contract=approved['declared_supported_runtime_contract'])
    if approved['kind'] in ('native_tail', 'native_response'):
        registration['bundle_root'] = str(bundle)
    for name, value in approved['assets'].items():
        loc = value['location']; root = bundle if loc['root'] == 'bundle' else roots['roguard']
        path = _contained(root, loc['relative_path'], symlinks=loc['root'] == 'bundle')
        registration['assets'][name] = dict(path=str(path), **{k: value[k] for k in ('bytes', 'sha256')})
    for name, value in approved['native_dependencies'].items():
        path = _contained(roots[value['package']], value['relative_path'])
        registration['native_dependencies'][name] = dict(path=str(path), **{k: value[k] for k in ('bytes', 'sha256')})
    require(digest(receipt(registration, approved['kind'])) == approved['expected_receipt_sha256'],
            'Approved path-independent closure identity differs')
    _verify(registration, roots, bundle)
    requested = Path(output_path)
    require(requested.parent.is_dir(), 'Closure output parent must exist')
    target = requested.parent.resolve(strict=True) / requested.name
    require(not target.exists() and not target.is_symlink(), 'Closure output cannot overwrite')
    require(not any(target.absolute().is_relative_to(root) for root in roots.values()),
            'Local registration cannot modify installed package')
    encoded = (canonical(registration) + '\n').encode()
    with target.open('xb') as stream:
        stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
    require(target.read_bytes() == encoded, 'Local closure readback differs')
    require(source.read_bytes() == raw, 'Approved inventory changed during rebasing')
    require(_roots(approved) == roots, 'Installed roots changed during rebasing')
    _verify(registration, roots, bundle)
    require(target.read_bytes() == encoded, 'Local closure changed during closing verification')
    return Binding(str(target), hashlib.sha256(encoded).hexdigest(),
                   approved['expected_receipt_sha256'], approved['kind'])


class NativeTailClosure:
    """Runtime verification of externally approved native-tail bundle inventories.

    The expected registration digest comes from rebase_closure's approved input.
    This class observes no files for approval and authenticates no release proof.
    """
    KIND = 'native_tail'

    def __init__(self, registration_path, expected_sha256):
        self._path = Path(registration_path)
        self._sha = _sha(expected_sha256)
        require(self._path.stat().st_size <= 32 << 20, 'Closure exceeds bound')
        raw = self._path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == self._sha, 'Closure registration differs')
        self._registration = json.loads(raw, object_pairs_hook=_unique)
        self._state = digest(self._registration)
        r = self._registration
        required = {'format', 'qualified', 'scope', 'registered', 'packages', 'assets',
                    'native_dependencies', 'asset_edges', 'required_asset_ids',
                    'required_runtime_assets', 'declared_supported_runtime_contract', 'bundle_root'}
        require(type(r) is dict and set(r) == required and
                r['format'] == KINDS[self.KIND][0] and r['qualified'] is False and
                r['scope'] == 'installed_candidate' and r['registered'] is True,
                'Explicit approved native-tail registration required')
        require(type(r['packages']) is dict and set(r['packages']) == PACKAGES,
                'Exact installed package inventory required')
        self._roots = {}
        for name, p in r['packages'].items():
            require(type(p) is dict and set(p) == {'root', 'distribution', 'version', 'files'},
                    'Exact installed package binding')
            root = Path(p['root'])
            require(type(p['root']) is str and root.is_absolute() and '..' not in root.parts,
                    'Absolute registered package root required')
            self._roots[name] = root
        self._bundle = Path(r['bundle_root'])
        require(type(r['bundle_root']) is str and self._bundle.is_absolute() and
                '..' not in self._bundle.parts, 'Absolute bundle root required')
        assets = {}
        for name, value in r['assets'].items():
            require(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'},
                    'Exact registered asset binding')
            path = Path(value['path'])
            require(path.is_relative_to(self._bundle), 'Runtime asset outside bundle root')
            assets[name] = dict(location={'root': 'bundle',
                'relative_path': path.relative_to(self._bundle).as_posix()},
                **{k: value[k] for k in ('bytes', 'sha256')})
        natives = {}
        for name, value in r['native_dependencies'].items():
            require(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'},
                    'Exact registered native binding')
            path = Path(value['path'])
            owners = [key for key, root in self._roots.items() if path.is_relative_to(root)]
            require(len(owners) == 1, 'Native file needs one registered package owner')
            key = owners[0]
            natives[name] = dict(package=key, relative_path=path.relative_to(self._roots[key]).as_posix(),
                                **{k: value[k] for k in ('bytes', 'sha256')})
        expected = _asset_ids(self.KIND)
        require(type(r['required_asset_ids']) is list and
                len(r['required_asset_ids']) == len(expected) and set(r['required_asset_ids']) == expected and
                type(r['required_runtime_assets']) is list and
                len(r['required_runtime_assets']) == len(expected) and set(r['required_runtime_assets']) == expected,
                'Exact native runtime asset inventory')
        self.receipt = receipt(r, self.KIND)
        self.identity = digest(self.receipt)
        self._approved = dict(format=FORMAT, kind=self.KIND,
            packages={name: {k: p[k] for k in ('distribution', 'version', 'files')}
                      for name, p in r['packages'].items()}, assets=assets, native_dependencies=natives,
            asset_edges=r['asset_edges'], declared_supported_runtime_contract=r['declared_supported_runtime_contract'],
            expected_receipt_sha256=self.identity)
        _validate(self._approved)  # Validate every finite declaration before target traversal.
        self._approved_state = digest(self._approved)
        self.verify()

    def asset_path(self, name):
        value = self._registration['assets'][name]
        path = Path(value['path'])
        return _contained(self._bundle, path.relative_to(self._bundle).as_posix())

    def asset_sha(self, name):
        return self._registration['assets'][name]['sha256']

    def cheap_verify(self):
        require(digest(self._approved) == self._approved_state and
                str(self._bundle) == self._registration['bundle_root'] and
                {k: str(v) for k, v in self._roots.items()} ==
                {k: v['root'] for k, v in self._registration['packages'].items()}, 'Closure derived state changed')
        require(digest(self._registration) == self._state and digest(self.receipt) == self.identity and
                digest(receipt(self._registration, self.KIND)) == self.identity, 'Closure state changed')
        require(_roots(self._approved) == self._roots, 'Installed package roots changed')
        require(not self._bundle.is_symlink() and self._bundle.resolve(strict=True) == self._bundle and
                self._bundle.is_dir(), 'Bundle root changed')
        for name, root in self._roots.items():
            require(not root.is_symlink() and root.resolve(strict=True) == root and root.is_dir(),
                    'Installed package root changed')
            for modname, module in list(sys.modules.items()):
                if modname == name or modname.startswith(name + '.'):
                    origin = getattr(module, '__file__', None)
                    if origin is None:
                        require({Path(v).resolve() for v in getattr(module, '__path__', [])} == {root},
                                'Loaded namespace origin changed')
                    else:
                        path = Path(origin).resolve()
                        require(path.is_relative_to(root) and (path.relative_to(root).as_posix() in
                            self._registration['packages'][name]['files'] or any(
                            Path(v['path']) == path for v in self._registration['native_dependencies'].values())),
                            'Loaded module outside approved inventory')
        return True

    def verify(self):
        self.cheap_verify()
        require(hashlib.sha256(self._path.read_bytes()).hexdigest() == self._sha,
                'Closure registration changed')
        _verify(self._registration, self._roots, self._bundle)
        from roguard.gemma4_contract import FILES, VERSIONS
        for name, expected in FILES.items():
            require({k: self._registration['assets']['base:' + name][k] for k in ('bytes', 'sha256')} == expected,
                    'Original native base asset differs')
        versions = {p['distribution']: p['version'] for p in self._registration['packages'].values()}
        require(all(versions.get(name) == version for name, version in VERSIONS.items()),
                'Native runtime dependency versions differ')
        return True

    def language_view(self, language):
        require(language in ('ro', 'uk'), 'Language closure required')
        self.verify()
        return _NativeLanguageClosure(self, language)


class NativeResponseClosure(NativeTailClosure):
    """Finite native74 assets under the same externally approved byte closure."""
    KIND = 'native_response'

    def language_view(self, language):
        raise ValueError('Response closure is one bilingual model, not a language view')

    def base_path(self):
        paths = [self.asset_path('base:' + name) for name in BASE]
        root = paths[0].parent
        require(all(path == root / name for path, name in zip(paths, BASE)),
                'Original base files must share one directory')
        return root


class _NativeLanguageClosure:
    """Path-free language view of one approved bilingual closure; no new approval."""
    def __init__(self, parent, language):
        self.parent, self.language = parent, language
        self.identity = digest({'format': 'native-tail-language-closure-1',
                                'parent': parent.identity, 'language': language})

    def verify(self):
        self.parent.verify()
        require(self.identity == digest({'format': 'native-tail-language-closure-1',
            'parent': self.parent.identity, 'language': self.language}), 'Language closure identity changed')
        return True

    def asset_path(self, name):
        require(name in {'manifest', 'native_spec', 'native_adapter', 'spec9', 'adapter9'}, 'Native runtime asset role')
        return self.parent.asset_path(self.language + '_' + name)

    def asset_sha(self, name):
        require(name in {'manifest', 'native_spec', 'native_adapter', 'spec9', 'adapter9'}, 'Native runtime asset role')
        return self.parent.asset_sha(self.language + '_' + name)

    def base_path(self):
        paths = [self.parent.asset_path('base:' + name) for name in sorted(BASE)]
        base = paths[0].parent
        require(all(path == base / name for path, name in zip(paths, sorted(BASE))),
                'Native base files must share one directory')
        return base
