'''
From facts for key question, draw findings on the hypothesis.
Improvement:
1. Have the prompts cite the exact facts with URL
'''

from prompts import get_initial_inference_prompt, get_inference_verifiers_prompt, get_inference_improved_prompt
from llm import chat_openai
from util import sanitize_filename
import json
import os

search_query = "How efficiently do the ECR enzymes work, especially in Kitsatospor setae bacteria?"
folder_path = f'autoscious_logs/{sanitize_filename(search_query)}/inferences'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

def inference_with_verifiers(search_query, decomposition, facts_text, key_question_idx):
    MAX_TOKENS = 10000
    model = "gpt-3.5-turbo"
    search_query_file_safe = sanitize_filename(search_query)
    # Save decomposition and verifier feedback to file
    if not os.path.exists(f'autoscious_logs/{search_query_file_safe}/inferences/kq{key_question_idx}'):
        os.makedirs(f'autoscious_logs/{search_query_file_safe}/inferences/kq{key_question_idx}')

    # Get initial inferences
    save_path = f'autoscious_logs/{search_query_file_safe}/inferences/kq{key_question_idx}/initial_inference.json'
    if not os.path.exists(save_path):
        initial_inference = chat_openai(get_initial_inference_prompt(decomposition, facts_text, MAX_TOKENS), model=model)[0]
        with open(save_path, 'w') as f:
            json.dump(json.loads(initial_inference), f, indent=2)

    # Get verifier feedback on initial inference
    save_path = f'autoscious_logs/{search_query_file_safe}/inferences/kq{key_question_idx}/initial_verifier_feedback.txt'
    if not os.path.exists(save_path):
        inference_verifiers_res = chat_openai(get_inference_verifiers_prompt(initial_inference), model=model)[0]
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(inference_verifiers_res)

    # Improve original decomposition with verifier feedback
    save_path = f'autoscious_logs/{search_query_file_safe}/inferences/kq{key_question_idx}/improved_inference.json'
    if not os.path.exists(save_path):
        improved_inference_res = chat_openai(get_inference_improved_prompt(initial_inference, inference_verifiers_res), model=model)[0]
        with open(save_path, 'w') as f:
            json.dump(json.loads(improved_inference_res), f, indent=2)

    # Get verifier feedback on final decomposition
    save_path = f'autoscious_logs/{search_query_file_safe}/inferences/kq{key_question_idx}/improved_verifier_feedback.txt'
    if not os.path.exists(save_path):
        improved_inference_verifiers_res = chat_openai(get_inference_verifiers_prompt(improved_inference_res), model=model)[0]
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(improved_inference_verifiers_res)

    # Return because all content should be in files
    return

def get_hypothesis_inference_findings_from_facts(search_query):
    # Load in decomposition
    with open(f'autoscious_logs/{sanitize_filename(search_query)}/decompositions/improved_decomposition.json', 'r') as f:
        decomposition = json.load(f)

    # Create a decomposition for each key question only
    key_question_decomposition_list = []
    for driver_key, driver_value in decomposition['key_drivers'].items():
        for hypothesis_key, hypothesis_value in driver_value['hypotheses'].items():
            for question_key, question_value in hypothesis_value['key_questions'].items():
                new_decomposition = decomposition.copy()
                new_decomposition['key_drivers'] = {
                    driver_key: {
                        'driver': driver_value['driver'],
                        'hypotheses': {
                            hypothesis_key: {
                                'hypothesis': hypothesis_value['hypothesis'],
                                'key_questions': {
                                    question_key: question_value
                                }
                            }
                        }
                    }
                }
                key_question_decomposition_list.append(new_decomposition)
    print("Key questions decomposition list: ", key_question_decomposition_list)

    # Go through each key question and make inferences on how its findings on the hypothesis.
    for i, kq_decomposition in enumerate(key_question_decomposition_list):
        with open(f'autoscious_logs/{sanitize_filename(search_query)}/facts/kq{i}/facts.txt', 'r', encoding='utf-8') as f:
            kq_facts = f.read()
        inference_with_verifiers(search_query, kq_decomposition, kq_facts, i)

get_hypothesis_inference_findings_from_facts(search_query)