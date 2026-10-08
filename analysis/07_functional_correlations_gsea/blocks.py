"""The 27 T-cell marker genes and their functional blocks, copied verbatim from
`cart_region/step76_27genes_ndrdr.py` (EFF_core / EFF_cyt / EFF_mem / EFF_act / EXH)."""
import os
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', '')
LEGACY_DATA_ROOT = os.environ.get('LEGACY_DATA_ROOT', '')
HOME_ROOT = os.environ.get('HOME_ROOT', '')
BLOCKS = {'EFF_core': ['Gzma', 'Gzmb', 'Gzmk', 'Prf1', 'Ifng', 'Nkg7'], 'EFF_cyt': ['Il12a', 'Il12b', 'Il21', 'Tnf'], 'EFF_mem': ['Ccr7', 'Tcf7', 'Sell', 'Lef1', 'Il7r'], 'EFF_act': ['Cd69', 'Il2ra', 'Il2rb', 'Mki67'], 'EXH': ['Pdcd1', 'Ctla4', 'Lag3', 'Havcr2', 'Tigit', 'Tox', 'Entpd1', 'Bach2']}
G2B = {g: b for b, v in BLOCKS.items() for g in v}
