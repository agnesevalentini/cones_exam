import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import random
from pathlib import Path
from screeninfo import get_monitors
from collections import defaultdict

from .logger import Logger

# visualize similarity
CV_FONT = cv2.FONT_HERSHEY_SIMPLEX
CV_LINE_TYPE = 2
HEIGHT_MARGIN = 130
MAX_FILENAME_LENGTH = 45
WINDOW_NAME = "Similar"


class SimilarityViewer:
    def __init__(self, sample_percent: int = 5, grid_size: int = 1024):
        self.sample_percent = sample_percent
        self.grid_size = grid_size

        self.similarity_matrix = None
        self.image_name_for_index = None
        self.similarity_index = {}
        self.cells = []
        self.grid = None

        self.screen_size = [-1, -1]
        self._get_biggest_screen_size()

    def _get_biggest_screen_size(self):
        for m in get_monitors():
            if m.width > self.screen_size[0] and m.height > self.screen_size[1]:
                self.screen_size[0] = m.width
                self.screen_size[1] = m.height

    def active(self):
        return self.sample_percent > 0

    def load_images(self, feature_vectors):
        vectors = np.zeros((len(feature_vectors), feature_vectors[0][1].shape[0]))
        for i, pair in enumerate(feature_vectors):
            file_name, feature_vector = pair
            vectors[i, :] = feature_vector
            self.similarity_index[str(file_name)] = i

        self.similarity_matrix = cosine_similarity(vectors).astype(np.float32)

        self.image_name_for_index = {v: k for k, v in self.similarity_index.items()}

    def _load_cell_overlay(self, image_file: Path, value: str = "", base: bool = False):
        img = cv2.imread(str(image_file))
        img = cv2.resize(img, (self.grid_size, self.grid_size))

        overlay = img.copy()
        x, y, w, h = 0, 0, self.grid_size, 80
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
        alpha = 0.8
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        shortened_file_name = (
            f"{'...' if len(str(image_file)) > MAX_FILENAME_LENGTH else ''}"
            f"{str(image_file)[-MAX_FILENAME_LENGTH:]}"
        )

        cv2.putText(
            img,
            f"{shortened_file_name}",
            (25, 30),
            CV_FONT,
            1,
            (255, 255, 255),
            CV_LINE_TYPE,
        )

        if base:
            cv2.putText(img, "base", (25, 65), CV_FONT, 1, (0, 255, 0), CV_LINE_TYPE)
        else:
            cv2.putText(
                img, f"{value}", (25, 65), CV_FONT, 1, (0, 0, 255), CV_LINE_TYPE
            )

        self.cells.append(img)

    def _build_grid_view(self):
        self.grid = np.vstack([np.hstack(self.cells[:3]), np.hstack(self.cells[3:])])
        self.cells = []

        cv2.putText(
            self.grid,
            "press 'n' for next | 'q' for quit",
            (int(2 * self.grid_size) + 10, int(2 * self.grid_size) - 20),
            CV_FONT,
            1.0,
            (0, 0, 255),
            CV_LINE_TYPE,
        )

        width_factor = self.grid.shape[1] / self.screen_size[0]
        height_factor = self.grid.shape[0] / (self.screen_size[1] - HEIGHT_MARGIN)

        # rescale to fit screen
        rescale_factor = max(width_factor, height_factor)
        new_width = int(self.grid.shape[1] / rescale_factor)
        new_height = int(self.grid.shape[0] / rescale_factor)

        self.grid = cv2.resize(
            self.grid, (new_width, new_height), interpolation=cv2.INTER_AREA
        )

    def _show_view(self):
        cv2.imshow(WINDOW_NAME, self.grid)
        while True:
            wait_key = cv2.waitKey(1)
            if wait_key == ord("n"):
                return False
            elif wait_key == ord("q"):
                return True
            elif cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return True

    def _get_random_samples(self):
        folder_keys = defaultdict(list)

        for key in self.similarity_index.keys():
            folder = Path(key).parents[0]
            folder_keys[folder].append(key)

        for folder, keys in folder_keys.items():
            random.shuffle(keys)
            i = 0
            while i <= int(len(keys) * (self.sample_percent / 100)):
                key = keys[i]
                yield key, self.similarity_index[key]
                i += 1

    def show_samples(self):
        for key, index in self._get_random_samples():
            similarities = self.similarity_matrix[index, :]
            similarities[index] = 0  # set self similarity to 0

            num_images = 5
            top_most_similar = np.argpartition(similarities, -num_images)[-num_images:]

            # the top one are not sorted, so we have to sort them
            top_similarity_values = similarities[top_most_similar]
            top = zip(top_most_similar, top_similarity_values)
            top = sorted(top, key=lambda tup: tup[1], reverse=True)

            image_file = Path(key)
            self._load_cell_overlay(image_file, base=True)

            for i in range(num_images):
                image_file = Path(self.image_name_for_index[top[i][0]])
                self._load_cell_overlay(image_file, value=f"{top[i][1]:.3f}")

            Logger.log_info(
                "Top most similar image for {} is {}".format(
                    key, self.image_name_for_index[top[0][0]]
                )
            )
            self._build_grid_view()
            quite = self._show_view()
            if quite:
                return
