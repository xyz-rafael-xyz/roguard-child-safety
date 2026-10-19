"""Read independently verified cache corpora without rewriting their identities."""
from training.diagnose_contextual_complementarity_v9 import read_rows
from training.extract_gemma4_qualified_scope_v9 import digest
from training.train_shared_tail_v9 import Reader
import json


class CompositeReader:
    def __init__(self, studies):
        self.entries, self.locations, self.readers = [], [], []
        seen = set()
        self.parents = {}
        for study in studies:
            for path, expected in json.loads((study/'source-freeze.json').read_text()).items():
                if digest(path) != expected:
                    raise ValueError('Frozen corpus differs: '+path)
            entries = read_rows(study/'inputs.jsonl.gz')
            for entry in entries:
                if entry['token_sha256'] in seen:
                    raise ValueError('Duplicate input across cache corpora')
                seen.add(entry['token_sha256'])
            reader = Reader(study, entries)
            self.locations.extend((len(self.readers), i) for i in range(len(entries)))
            self.entries.extend(entries)
            self.readers.append(reader)
            report_path = study/'actual/report.json'
            self.parents[str(report_path)] = digest(report_path)
            for chunk in json.loads(report_path.read_text())['chunks']:
                self.parents[str(study/'actual'/f"chunk-{chunk['start']:06d}.json")] = chunk['sha256']

    def read_many(self, indices):
        grouped = {}
        for position, index in enumerate(indices):
            if type(index) is not int or not 0 <= index < len(self.locations):
                raise ValueError('Invalid composite cache index')
            reader, local = self.locations[index]
            grouped.setdefault(reader, []).append((position, local))
        result = [None]*len(indices)
        for reader, requests in grouped.items():
            states = self.readers[reader].read_many([local for _, local in requests])
            for (position, _), state in zip(requests, states):
                result[position] = state
        return result

    def development_batches(self, language, size):
        # Preserve the exact original per-corpus batch shape: GEMM rounding
        # changes when appending rows to the last partial development batch.
        offset = 0
        for reader in self.readers:
            indices = [offset+i for i, entry in enumerate(reader.entries)
                if entry['partition'] == 'dev' and entry['row']['language'] == language]
            for start in range(0, len(indices), size):
                yield indices[start:start+size]
            offset += len(reader.entries)
