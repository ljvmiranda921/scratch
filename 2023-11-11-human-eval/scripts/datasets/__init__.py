from enum import Enum

from .hellaswag import HellaSwagDataset
from .piqa import PIQADataset
from .winogrande import WinograndeDataset


class Dataset(str, Enum):
    piqa = "piqa"
    hellaswag = "hellaswag"
    winogrande = "winogrande"


DATASETS = {
    Dataset.piqa.value: PIQADataset,
    Dataset.hellaswag.value: HellaSwagDataset,
    Dataset.winogrande.value: WinograndeDataset,
}
