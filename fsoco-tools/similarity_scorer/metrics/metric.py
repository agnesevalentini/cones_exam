from collections import defaultdict
from pathlib import Path

import numpy as np


class Metric:
    def __init__(self):
        self.feature_vectors = None
        self.index = None
        self.name = ""

    def load_feature_vectors(self, feature_vectors: [(Path, np.array)]):
        num_rows = len(feature_vectors)
        num_columns = feature_vectors[0][1].shape[0]
        img_vec = np.zeros(shape=(num_rows, num_columns))

        self.index = defaultdict(dict)

        i = 0
        for image_file_path, vec in feature_vectors:
            if vec is not None:
                if len(image_file_path.parts) == 1:
                    folder = "."
                else:
                    folder = "/".join(image_file_path.parts[:-1])

                self.index[folder][image_file_path] = i
                img_vec[i, :] = vec
                i += 1

        self.feature_vectors = img_vec[: i + 1, :].astype(np.float32)

    def can_be_applied_per_folder(self):
        if len(self.index.keys()) > 1:
            return True
        else:
            return False

    def get_index_per_folder(self):
        for folder in self.index.keys():
            yield folder, self.index[folder]

    def get_index_for_all_images(self):
        flat_index = {}
        for _, index in self.get_index_per_folder():
            flat_index.update(index)

        return flat_index

    def get_feature_vectors_for_index(self, index: dict):
        new_index = {}
        num_rows = len(index.keys())
        num_columns = self.feature_vectors.shape[1]

        img_vecs = np.zeros(shape=(num_rows, num_columns))

        i = 0
        for key, j in index.items():
            img_vecs[i] = self.feature_vectors[j]
            new_index[key] = i
            i += 1

        return new_index, img_vecs

    def get_metric(self, per_folder: bool = True) -> [(Path, float)]:
        raise NotImplementedError()
