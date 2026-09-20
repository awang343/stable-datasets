from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import Split, SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


class OxfordIIITPet(BaseDatasetBuilder):
    """
    Oxford-IIIT Pet Dataset

    STATUS: work in progress, claimed for CSCI 2952X course project.
    _split_generators / _generate_examples are not implemented yet.

    Abstract
    A 37 category pet dataset with roughly 200 images per class. Images have
    large variations in scale, pose, and lighting. Each image is annotated
    with a breed label (species + breed) and a trimap segmentation.
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://www.robots.ox.ac.uk/~vgg/data/pets/",
        citation=r"""@inproceedings{parkhi12a,
                         title={Cats and Dogs},
                         author={Parkhi, O. M. and Vedaldi, A. and Zisserman, A. and Jawahar, C. V.},
                         booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
                         year={2012}}""",
        assets={
            "images": DownloadInfo(url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"),
            "annotations": DownloadInfo(url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"),
        },
    )

    def _info(self):
        return DatasetInfo(
            description="Oxford-IIIT Pet dataset with 37 breed categories.",
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

        asset_keys = ["images", "annotations"]
        local_paths = bulk_download([source["assets"][key] for key in asset_keys], dest_folder=self._raw_download_dir)
        path_map = dict(zip(asset_keys, local_paths))

        # TODO: annotations.tar.gz contains annotations/trainval.txt and
        # annotations/test.txt (space-separated "<image_id> <class_id> ..."),
        # which define the train/test split. Wire those up here.
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"path_map": path_map, "split": "trainval"},
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"path_map": path_map, "split": "test"},
            ),
        ]

    def _generate_examples(self, path_map, split):
        # TODO: extract annotations/{trainval,test}.txt to get the image ->
        # label mapping for `split`, then decode the matching JPEGs out of
        # images.tar.gz (mirrors stable_datasets/images/flowers102.py).
        raise NotImplementedError("OxfordIIITPet._generate_examples is not implemented yet.")

    @staticmethod
    def _labels():
        """37 breed names, in the order used by the dataset's own annotation files."""
        return [
            "Abyssinian",
            "Bengal",
            "Birman",
            "Bombay",
            "British_Shorthair",
            "Egyptian_Mau",
            "Maine_Coon",
            "Persian",
            "Ragdoll",
            "Russian_Blue",
            "Siamese",
            "Sphynx",
            "american_bulldog",
            "american_pit_bull_terrier",
            "basset_hound",
            "beagle",
            "boxer",
            "chihuahua",
            "english_cocker_spaniel",
            "english_setter",
            "german_shorthaired",
            "great_pyrenees",
            "havanese",
            "japanese_chin",
            "keeshond",
            "leonberger",
            "miniature_pinscher",
            "newfoundland",
            "pomeranian",
            "pug",
            "saint_bernard",
            "samoyed",
            "scottish_terrier",
            "shiba_inu",
            "staffordshire_bull_terrier",
            "wheaten_terrier",
            "yorkshire_terrier",
        ]
