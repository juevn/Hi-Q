from .registry import get_dataset_cls
import src.dataset


def load_dataset(cfg):
    name = cfg.benchmark.name
    ds_cls = get_dataset_cls(name)
    return ds_cls(cfg)
