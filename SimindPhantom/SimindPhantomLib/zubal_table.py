"""Default Zubal index table and TotalSegmentator -> Zubal mapping (Slicer-independent)."""
import os

from . import core

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "Resources", "Data")
DEFAULT_MAPPING_CSV = os.path.join(DATA_DIR, "ts_to_zubal.csv")
DEFAULT_ZUBAL_CSV = os.path.join(DATA_DIR, "zubal_indices.csv")
DEFAULT_CURVE_CSV = os.path.join(DATA_DIR, "generic_120kVp_curve.csv")

UNKNOWN_DEFAULT_INDEX = 9   # unknown segment names -> skeletal muscle / soft tissue


def default_rules():
    return core.load_mapping_csv(DEFAULT_MAPPING_CSV)


def default_zubal_table():
    return core.load_zubal_csv(DEFAULT_ZUBAL_CSV)


def zubal_name(table, index):
    e = table.get(int(index))
    return e.name if e else f"(index {index} not in table)"


def zubal_density(table, index, fallback=1000):
    e = table.get(int(index))
    return e.density if e else fallback
