import logging
from typing import Dict, Any

from data.data_instance_processor.data_instance_processor import (
    DataInstanceProcessor,
    IdentityDataInstanceProcessor,
)

logger = logging.getLogger("app")


@DataInstanceProcessor.register("pcfg_bos", exist_ok=True)
class S2SPCFGBOSDataInstanceProcessor(IdentityDataInstanceProcessor):
    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        example = super().__call__(example)
        example.update({self.source_seq_key: f"Solve: {example[self.source_seq_key]}"})
        return example


if __name__ == "__main__":
    data_instance = {
        "source": "echo append copy Q10 E4 S15 P7 , shift remove_second K15 D8 M9 W5 , H12 Q5 G10 J6 J10",
        "target": "Q10 E4 S15 P7 D8 M9 W5 K15 K15",
        "category": 5,
    }
    processor = S2SPCFGBOSDataInstanceProcessor(input_output_sep_token="[SEP] ")

    o = processor(data_instance)

    print(o)
