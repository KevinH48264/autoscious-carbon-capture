from prompts import get_initial_decomposition_prompt, get_decomposition_verifiers_prompt, get_decomposition_improved_prompt
from llm import chat_openai # copied over to folder version for now
import re
import json

def sanitize_filename(filename):
    return re.sub(r'[^a-zA-Z0-9]', '_', filename)

search_query = "How efficiently do the ECR enzymes work, especially in Kitsatospor setae bacteria?"

# Total is around 4K tokens = $0.24
# Might be overkill for now, but for verification, it's pretty important to be as thorough as posible
# Need GPT-4 with higher capabilities, because GPT3.5-turbo currently does not respond to feedback
def decompose_with_verifiers(search_query):
    model = "gpt-3.5-turbo"
    search_query_file_safe = sanitize_filename(search_query)

    # Get initial decomposition
    initial_decomposition_res = chat_openai(get_initial_decomposition_prompt(search_query), model=model)[0]

    # Get verifier feedback on initial decomposition
    decomposition_verifiers_res = chat_openai(get_decomposition_verifiers_prompt(initial_decomposition_res), model=model)[0]

    # Improve original decomposition with verifier feedback
    improved_decomposition_res = chat_openai(get_decomposition_improved_prompt(search_query, initial_decomposition_res, decomposition_verifiers_res), model=model)[0]

    # Get verifier feedback on final decomposition
    improved_decomposition_verifiers_res = chat_openai(get_decomposition_verifiers_prompt(improved_decomposition_res), model=model)[0]

    improved_decomposition_res_json = json.loads(improved_decomposition_res)

    # Save decomposition and verifier feedback to file
    with open(f'autoscious_logs/decompositions/{search_query_file_safe}.json', 'w') as f:
        json.dump(improved_decomposition_res_json, f, indent=2)
    with open(f'autoscious_logs/decompositions/{search_query_file_safe}_verifier_feedback.txt', 'w', encoding='utf-8') as f:
        f.write(improved_decomposition_verifiers_res)

    # Return final decomposition, along with verifier feedback
    return improved_decomposition_res_json, improved_decomposition_verifiers_res

decomposition, decomposition_verifier_feedback = decompose_with_verifiers(search_query)

print("\n Decomposition: \n", decomposition)
print("\n Decomposition verifier feedback: \n", decomposition_verifier_feedback)
