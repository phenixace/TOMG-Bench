python query_openai.py --task MolEdit --subtask AddComponent --json_check --port 8002 --name GPT-4o --model gpt-4o
python query_openai.py --task MolEdit --subtask DelComponent --json_check --port 8002 --name GPT-4o --model gpt-4o
python query_openai.py --task MolEdit --subtask SubComponent --json_check --port 8002 --name GPT-4o --model gpt-4o

python query_openai.py --task MolCustom --subtask AtomNum --json_check --port 8002 --name GPT-4o --model gpt-4o
python query_openai.py --task MolCustom --subtask BondNum --json_check --port 8002 --name GPT-4o --model gpt-4o
python query_openai.py --task MolCustom --subtask FunctionalGroup --json_check --port 8002 --name GPT-4o --model gpt-4o

python query_openai.py --task MolOpt --subtask LogP --json_check --port 8002 --name GPT-4o --model gpt-4o
python query_openai.py --task MolOpt --subtask MR --json_check --port 8002 --name GPT-4o --model gpt-4o
python query_openai.py --task MolOpt --subtask QED --json_check --port 8002 --name GPT-4o --model gpt-4o
