from enum import Enum

from .hellaswag import HellaSwagDataset
from .piqa import PIQADataset
from .winogrande import WinograndeDataset
from .lambada import LAMBADADataset


class Dataset(str, Enum):
    piqa = "piqa"
    hellaswag = "hellaswag"
    winogrande = "winogrande"
    lambada = "lambada"


DATASETS = {
    Dataset.piqa.value: PIQADataset,
    Dataset.hellaswag.value: HellaSwagDataset,
    Dataset.winogrande.value: WinograndeDataset,
    Dataset.lambada.value: LAMBADADataset,
}
