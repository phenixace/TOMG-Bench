# python query.py --task MolCustom --subtask AtomNum --json_check --port 8000
python query.py --task MolCustom --subtask BondNum --json_check --port 8000
# python query.py --task MolCustom --subtask FunctionalGroup --json_check --port 8000

python query.py --task MolEdit --subtask AddComponent --json_check --port 8001
python query.py --task MolEdit --subtask DelComponent --json_check --port 8000
python query.py --task MolEdit --subtask SubComponent --json_check --port 8001

python query.py --task MolOpt --subtask LogP --json_check --port 8000
python query.py --task MolOpt --subtask MR --json_check --port 8001
python query.py --task MolOpt --subtask QED --json_check --port 8000