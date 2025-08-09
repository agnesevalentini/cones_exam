from pathlib import Path
import shutil
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import time
from tqdm import tqdm

from .logger import Logger

# Clusters
CLUSTERS_FOLDER_NAME = "clusters"
CLUSTER_FOLDER_PREFIX = "cluster_"
NO_CLUSTER_FOLDER_NAME = "_no_cluster_"
AUTO_SELECTION_FOLDER = "_auto_selection_"


class SimilarityClustering:
    def __init__(self, clustering_threshold: float, auto_select: bool = False):
        self.clustering_threshold = clustering_threshold
        self.auto_select = auto_select

        self.similarity_matrix = None
        self.image_name_for_index = None
        self.similarity_index = {}
        self.filenames_in_folder = defaultdict(list)
        self.ids_in_folder = defaultdict(list)

        self.graph = None
        self.clusters = []

    def active(self):
        return self.clustering_threshold > 0

    @staticmethod
    def _create_output_folders(folder: str):
        src_folder = Path(folder)
        review_folder = Path(f"{folder}/{CLUSTERS_FOLDER_NAME}")

        shutil.rmtree(review_folder, ignore_errors=True)
        Path.mkdir(review_folder)

        return src_folder, review_folder

    def load_images(self, feature_vectors):
        vectors = np.zeros((len(feature_vectors), feature_vectors[0][1].shape[0]))
        for i, pair in enumerate(feature_vectors):
            file_name, feature_vector = pair
            vectors[i, :] = feature_vector
            self.similarity_index[str(file_name)] = i

            folder = file_name.parents[0]
            self.filenames_in_folder[folder].append(file_name)
            self.ids_in_folder[folder].append(i)

        self.similarity_matrix = cosine_similarity(vectors).astype(np.float32)
        self.image_name_for_index = {v: k for k, v in self.similarity_index.items()}

    def _find_clusters(self):
        high_similarity = self.similarity_matrix > self.clustering_threshold
        self.graph = nx.from_numpy_matrix(high_similarity, create_using=nx.Graph)
        self.clusters = [
            cluster
            for cluster in nx.connected_components(self.graph)
            if len(cluster) > 1
        ]
        self.clusters.sort(key=len)
        self.clusters.reverse()

    def _get_clusters_for_ids(self, ids_in_folder: set):
        clusters_in_folder = []

        for cluster in self.clusters:
            cluster_in_folder = cluster & ids_in_folder
            if len(cluster_in_folder) > 1:
                clusters_in_folder.append(cluster_in_folder)
                ids_in_folder = ids_in_folder - cluster_in_folder

        return clusters_in_folder, ids_in_folder

    def _get_filenames_for_ids(self, ids):
        for i in ids:
            yield self.image_name_for_index[i]

    def _get_auto_selection(self):
        Logger.log_info("Start auto selection process.")

        finished = False
        num_removed_nodes = 0
        iterations = 0

        while not finished:
            start_time = time.time()
            num_nodes_in_clusters = 0

            clusters = [
                cluster
                for cluster in nx.connected_components(self.graph)
                if len(cluster) > 1
            ]

            if len(clusters) == 0:
                finished = True

                num_nodes_final = len(self.graph.nodes)
                num_nodes_total = num_nodes_final + num_removed_nodes

                Logger.log_info(f"Needed {iterations} iterations to break up clusters.")
                Logger.log_info(
                    f"Picked {num_nodes_final} out of {num_nodes_total} images.",
                    bold=True,
                )

                break
            else:
                iterations += 1

            # TODO performance bottleneck, find parallel solution

            for cluster in clusters:
                max_edges = 0
                node_with_max_edges = -1
                cluster_size = len(cluster)

                for node in cluster:
                    # only count nodes that are still in a cluster after this iteration
                    num_nodes_in_clusters += 1 if cluster_size > 2 else 0

                    edges = self.graph.edges(node)
                    if len(edges) > max_edges:
                        node_with_max_edges = node
                        max_edges = len(edges)

                self.graph.remove_node(node_with_max_edges)
                num_removed_nodes += 1

            round_duration = time.time() - start_time
            nodes_still_in_cluster = num_nodes_in_clusters - len(clusters)
            print(
                f"> Removal round {iterations}: time needed {round_duration:4.2f}s; {nodes_still_in_cluster} images still in a cluster ...",
                end="\r",
            )

        return list(self.graph.nodes)

    def run(self):
        self._find_clusters()
        selection_ids = self._get_auto_selection() if self.auto_select else []

        number_images_to_copy = 0
        for folder, ids_in_folder in self.ids_in_folder.items():
            ids_in_folder = set(ids_in_folder)
            _, in_no_cluster = self._get_clusters_for_ids(ids_in_folder)

            cluster_ratio = (len(ids_in_folder) - len(in_no_cluster)) / len(
                ids_in_folder
            )
            Logger.log_info(
                f"{folder} -> {cluster_ratio * 100:.2f}% of the images are in dense clusters!"
            )

            number_images_to_copy += len(ids_in_folder)
            if self.auto_select:
                selected_ids_in_folder = set(ids_in_folder) & set(selection_ids)
                number_images_to_copy += len(selected_ids_in_folder)

        Logger.log_info("Start copying images into cluster folders ...")

        with tqdm(total=number_images_to_copy) as pbar:
            for folder, ids_in_folder in self.ids_in_folder.items():
                ids_in_folder = set(ids_in_folder)
                _, review_folder = self._create_output_folders(folder)
                clusters_in_folder, in_no_cluster = self._get_clusters_for_ids(
                    ids_in_folder
                )

                for i, cluster in enumerate(clusters_in_folder):
                    cluster_folder = Path(
                        review_folder / f"{CLUSTER_FOLDER_PREFIX}{i:04d}"
                    )
                    Path.mkdir(cluster_folder)

                    for file in self._get_filenames_for_ids(cluster):
                        src = Path(file)
                        dst = cluster_folder / src.name
                        shutil.copy2(src, dst)

                no_cluster_folder = Path(review_folder / NO_CLUSTER_FOLDER_NAME)
                Path.mkdir(no_cluster_folder)

                for file in self._get_filenames_for_ids(in_no_cluster):
                    src = Path(file)
                    dst = no_cluster_folder / src.name
                    shutil.copy2(src, dst)
                    pbar.update(1)

                if self.auto_select:
                    selection_folder = Path(review_folder / "_auto_selection_")
                    Path.mkdir(selection_folder)

                    selected_ids_in_folder = set(ids_in_folder) & set(selection_ids)

                    for file in self._get_filenames_for_ids(selected_ids_in_folder):
                        src = Path(file)
                        dst = selection_folder / src.name
                        shutil.copy2(src, dst)
                        pbar.update(1)
