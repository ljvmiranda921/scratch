from enum import Enum
from typing import Dict

from .base import DatasetReader
from .hellaswag import HellaSwag
from .lambada import Lambada
from .piqa import PIQA
from .winogrande import Winogrande


class Dataset(str, Enum):
    piqa = "piqa"
    hellaswag = "hellaswag"
    winogrande = "winogrande"
    lambada = "lambada"


def get_dataset_reader(name: Dataset) -> DatasetReader:
    reader_map: Dict[str, DatasetReader] = {
        Dataset.piqa.value: PIQA,
        Dataset.hellaswag.value: HellaSwag,
        Dataset.winogrande.value: Winogrande,
        Dataset.lambada.value: Lambada,
    }

    reader_instance = reader_map[name.value]()
    return reader_instance
