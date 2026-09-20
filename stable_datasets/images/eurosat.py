from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


class EuroSAT(BaseDatasetBuilder):
    """
    EuroSAT Dataset

    STATUS: work in progress, claimed for CSCI 2952X course project.
    _split_generators / _generate_examples are not implemented yet.

    Abstract
    A land use / land cover classification dataset built from Sentinel-2
    satellite imagery, covering 10 classes with 27,000 labeled 64x64 RGB
    images in total. There is no official train/test split; a fixed random
    split (or the commonly used 80/20 split from follow-up benchmarks) needs
    to be chosen here.
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://github.com/phelber/EuroSAT",
        citation=r"""@article{helber2019eurosat,
                         title={EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification},
                         author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
                         journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
                         year={2019}}""",
        assets={
            # RGB-only version (there is also a 13-band multispectral .npy release).
            "images": DownloadInfo(url="https://madm.dfki.de/files/sentinel/EuroSAT.zip"),
        },
    )

    def _info(self):
        return DatasetInfo(
            description="EuroSAT dataset with 10 land use / land cover classes.",
            features=Features(
                {
                    "image": ImageFeature(encode_format="JPEG"),
                    "label": ClassLabel(names=self._labels()),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self):
        source = self._source()

        local_paths = bulk_download([source["assets"]["images"]], dest_folder=self._raw_download_dir)
        path_map = {"images": local_paths[0]}

        # TODO: no official split ships with the archive (it's one folder per
        # class). Decide + document a fixed train/val/test split here (e.g.
        # a seeded random split) so results are reproducible across re-runs.
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"path_map": path_map, "split": "train"},
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"path_map": path_map, "split": "test"},
            ),
        ]

    def _generate_examples(self, path_map, split):
        # TODO: unzip EuroSAT.zip, walk the per-class subfolders (one per
        # entry in `_labels()`), decode JPEGs, and apply the chosen split.
        raise NotImplementedError("EuroSAT._generate_examples is not implemented yet.")

    @staticmethod
    def _labels():
        """10 land use / land cover class names, matching the dataset's folder names."""
        return [
            "AnnualCrop",
            "Forest",
            "HerbaceousVegetation",
            "Highway",
            "Industrial",
            "Pasture",
            "PermanentCrop",
            "Residential",
            "River",
            "SeaLake",
        ]
