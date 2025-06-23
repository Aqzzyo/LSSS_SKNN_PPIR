from yacs.config import CfgNode
from .registry import AGGREGATORS, SPLITTERS, EXTRACTORS
from .aggregator import AggregatorBase
from ..utils import simple_build
from typing import List

def build_aggregators(cfg: CfgNode) -> List[AggregatorBase]:
    """
    Instantiate a list of aggregator classes.

    Args:
        cfg (CfgNode): the configuration tree.

    Returns:
        aggregators (list): a list of instances of aggregator class.
    """
    names = cfg["names"]
    aggregators = list()
    for name in names:
        aggregators.append(simple_build(name, cfg, AGGREGATORS))
    return aggregators