import json
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer


class EventNormalizer:
    def __init__(self, method="min_max_per_vector"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.method = method
        self.stats = {}

    def normalize_text_field(self, events, field_name):
        """Normalize text field embeddings to [0,1] range using per-vector method."""
        print(f"Normalizing {field_name} with method: {self.method}")

        texts = [event.get(field_name, "none") for event in events]

        embeddings = self.model.encode(texts, normalize_embeddings=False)

        normalized_embeddings = []
        for emb in embeddings:
            emb_min, emb_max = emb.min(), emb.max()
            if emb_max != emb_min:
                norm_emb = (emb - emb_min) / (emb_max - emb_min)
            else:
                norm_emb = np.full_like(emb, 0.5)
            normalized_embeddings.append(norm_emb)

        normalized_embeddings = np.array(normalized_embeddings)

        for i, event in enumerate(events):
            event[field_name] = normalized_embeddings[i].tolist()

        self.stats[field_name] = {
            "method": self.method,
            "original_range": [float(embeddings.min()), float(embeddings.max())],
            "normalized_range": [float(normalized_embeddings.min()), float(normalized_embeddings.max())],
            "mean": float(normalized_embeddings.mean()),
            "std": float(normalized_embeddings.std()),
        }

        print(f"  Normalized range: [{self.stats[field_name]['normalized_range'][0]:.6f}, "
              f"{self.stats[field_name]['normalized_range'][1]:.6f}]")

    def _is_empty_value(self, val):
        if val is None:
            return True
        if isinstance(val, str) and not val.strip():
            return True
        if isinstance(val, list) and (not val or all(not str(v).strip() for v in val)):
            return True
        return False

    def normalize_time_field(self, events, key="time", date_format="%B %d, %Y", reuse=False):
        timestamps = []
        null_indices = []

        for i, e in enumerate(events):
            time_val = e.get(key)

            if self._is_empty_value(time_val):
                timestamps.append(0)
                null_indices.append(i)
                continue

            try:
                if isinstance(time_val, list):
                    valid_times = [t for t in time_val if not self._is_empty_value(t)]
                    if not valid_times:
                        timestamps.append(0)
                        null_indices.append(i)
                        continue
                    ts_list = [datetime.strptime(t, date_format).timestamp() for t in valid_times]
                    ts = sum(ts_list) / len(ts_list)
                else:
                    ts = datetime.strptime(time_val, date_format).timestamp()
                timestamps.append(ts)
            except (ValueError, TypeError):
                print(f"WARNING: Invalid time format for event {i}: {time_val}, setting to null")
                timestamps.append(0)
                null_indices.append(i)

        valid_timestamps = [ts for i, ts in enumerate(timestamps) if i not in null_indices]

        if not valid_timestamps:
            print("WARNING: No valid timestamps found, all time values will be null")
            for e in events:
                e[key] = 0
            return

        if reuse and "time" in self.stats:
            min_ts = self.stats["time"]["min"]
            max_ts = self.stats["time"]["max"]
            print(f"Reusing time normalization: min={min_ts}, max={max_ts}")
        else:
            min_ts = min(valid_timestamps)
            max_ts = max(valid_timestamps)
            self.stats["time"] = {"min": min_ts, "max": max_ts, "null_count": len(null_indices)}
            print(f"Time normalization: min={min_ts}, max={max_ts}")

        denom = max_ts - min_ts if max_ts != min_ts else 1

        for i, e in enumerate(events):
            if i in null_indices:
                e[key] = 0
            else:
                e[key] = (timestamps[i] - min_ts) / denom

        print(f"Normalized time field for {len(events)} events ({len(null_indices)} set to null)")

    def save_stats(self, filepath="normalization_stats.json"):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)
        print(f"Saved normalization stats to {filepath}")

    def load_stats(self, filepath="normalization_stats.json"):
        with open(filepath, "r", encoding="utf-8") as f:
            self.stats = json.load(f)
        print(f"Loaded normalization stats from {filepath}")

    def get_stats(self):
        return self.stats
