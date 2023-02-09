# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Openrice review dataset."""


import os
import csv
import datasets
from datasets.tasks import TextClassification


_DESCRIPTION = """\
Openrice review dataset\
"""

_CITATION = """\
None
"""

# Openrice

_URLs = {
    "Openrice": r"/dataset"
    # "Openrice": r"../Dataset"
}


class OpenriceReviewsConfig(datasets.BuilderConfig):
    """BuilderConfig for Openrice."""

    def __init__(self, **kwargs):
        """BuilderConfig for Openrice.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(OpenriceReviewsConfig, self).__init__(**kwargs)

class Openrice(datasets.GeneratorBasedBuilder):
    """Openrice dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        OpenriceReviewsConfig(
            name="Openrice", version=VERSION, description="Openrice dataset"
        ),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "labels": datasets.features.ClassLabel(
                    names=["1", "2", "3", "4", "5"]
                ),
                "text": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=None,
            citation=_CITATION,
            # task_templates=[TextClassification(text_column="text", label_column="label")],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # my_urls = _URLs[self.config.name]
        # data_dir = dl_manager.download_and_extract(my_urls)
        data_dir = _URLs[self.config.name]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "data.csv"),
                    "split": "train",
                }
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.TEST,
            #     gen_kwargs={
            #         "filepath": os.path.join(data_dir, "validate.tsv"),
            #         "split": "test",
            #     }
            # ),
        ]

    def _generate_examples(self, filepath, split):
        """Generate Openrice examples."""
        # For labeled examples, extract the label from the path.
        with open(filepath, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter = '\t')
            for id_, row in enumerate(reader):
                if id_ == 0:
                    continue
                yield id_, {
                    "text": row[0],
                    "labels": int(row[1]) - 1,
                }

