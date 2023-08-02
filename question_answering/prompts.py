def get_initial_decomposition_prompt(search_query):
    return f'''
Research project question: {search_query}

Task: Decompose this research question into 1) project objective: what should be the outcome of the project, 2) key drivers: what are the key problem drivers to achieve the project objective, 3) hypotheses: what can already be hypothesized about the key drivers, 4) key questions: what analyses need to be conducted to verify, falsify, or change the hypotheses?

Rules: Be as mutually exclusive, completely exhaustive (MECE) as possible. The output should be in nested JSON format of 
```json
{{
  "project_objective": "",
  "key_drivers": {{
    "1": {{
      "driver": "",
      "hypotheses": {{
          "1": {{
            "hypothesis": "",
            "key_questions": {{
                "1": "", 
                "2": "",
                etc.
            }},
          }},
          etc.
      }},
    "2" : {{}},
    etc.
    }}"",
  }},
}}
```
Respond only with the output, with no explanation or conversation.
'''

def get_decomposition_verifiers_prompt(response_1):
    return f'''
{response_1}

Task:
You are well trained in logical reasoning and verification and understand JSON format. Based on the sub-questions and sub-tasks, use the following logical verification checks to ensure that the decomposition is accurate, relevant, and comprehensive.

1. Comprehensiveness: Do the sub-questions or sub-tasks cover all aspects of the main question or task? This check ensures you haven't left out any key component that needs to be addressed.
2. Relevance: Are all sub-questions or sub-tasks directly related to the main question or task? This helps to ensure that you haven't included unnecessary elements that may distract from the main issue.
3. Consistency: Are the sub-questions or sub-tasks logically consistent with each other and the main question or task? They should be interconnected and not contradict each other.
4. Depth: Have you dissected the question or task enough? Some complex issues might need several layers of breakdown. Ensure that the depth of your analysis corresponds to the complexity of the main question or task.
5. Clarity: Are the sub-questions or sub-tasks clear and straightforward? They should not introduce further complexity and should be easily understandable.
6. Synthesis Capability: Can the answers to the sub-questions or results of the sub-tasks be combined to solve the main problem or answer the main question? The breakdown should facilitate a clear path to the solution.
7. Decomposition feedback: Provide feedback on the decomposition content itself.

Your response should be in this format:
1. Comprehensiveness: 
2. Relevance: 
3. Consistency: 
4. Depth: 
5. Clarity: 
6. Synthesis Capability:
7. Decomposition feedback: 

Be as critical, thorough, and specific in your responses as possible.
'''

def get_decomposition_improved_prompt(search_query, response_1, response_2):
    return f'''
Research project question: {search_query}

Previous Task: Decompose this research question into 1) project objective: what should be the outcome of the project, 2) key drivers: what are the key problem drivers to achieve the project objective, 3) hypotheses: what can already be hypothesized about the key drivers, 4) key questions: what analyses need to be conducted to verify, falsify, or change the hypotheses?

Decomposition:
{response_1}

Feedback:
{response_2}

Current Task:
You are well trained in logical reasoning and verification and understand JSON format. Based on the decomposition and feedback provided, improve the robustness of the decomposition.

Rules: Be as mutually exclusive, completely exhaustive (MECE) as possible. The output should be in nested JSON format of 
```json
{{
  "project_question:", "",
  "project_objective": "",
  "key_drivers": {{
    "1": {{
      "driver": "",
      "hypotheses": {{
          "1": {{
            "hypothesis": "",
            "key_questions": {{
                "1": "", 
                "2": "",
                etc.
            }},
          }},
          etc.
      }},
    "2" : {{}},
    etc.
    }}"",
  }},
}}
```
Respond only with the output, with no explanation or conversation.
'''

def get_initial_inference_prompt(decomposition, facts_text, max_tokens):
    return f'''
Decomposed research question:
{decomposition}

Facts: {facts_text[:max_tokens]}

Task: 
Based on the facts, report your findings on the current answer to the key question and how this verifies, falsifies, or changes the hypothesis. Be sure to use accurate direct quotes and their urls from the the facts used in each reasoning step. Think step by step.

The output should be of the JSON format: 
```json
{{
    "relevant_facts": "<insert relevant facts with urls>",
    "key_question_answer": "<insert answer to key question>",
    "hypothesis_finding": "<insert hypothesis>",
}}
```
'''

def get_inference_verifiers_prompt(inference):
    return f'''
Inference
{inference}

Task:
You are well trained in logical reasoning and verification and understand JSON format. Based on the facts and assuming all the facts are true, use the following logical verification checks to ensure that the inference is valid.

1. Deductive consistency: Based on the premises and assuming all the premises are true, is the conclusion also true, partially true, or false? Does the conclusion contradict any of the premises or does it logically follow from them? And what percentage degree of support do the premises provide for the truth of the conclusion?
2. Coherence with background knowledge: True, partially true, or false, and to what percentage degree: does the conclusion align with what is generally accepted or known about the subject? Generate background knowledge about the conclusion.
3. Biases or fallacies: True or Fale: Are there any biases or logical fallacies present that might undermine the argument? Common ones to look for include hasty generalization, or appeal to ignorance.
4. Inference feedback: Provide feedback on the inference content itself.

Your response should be in this format:
1. Deductive consistency: 
Reasoning: 
Label: 
Score: 
2. Coherence with background knowledge: 
Background knowledge: 
Reasoning: 
Label: 
Score: 
3. Consistency: 
Reasoning: 
Label:
4. Inference feedback: 
Feedback: 

Be as critical, thorough, and specific in your responses as possible.
'''

def get_inference_improved_prompt(initial_inference, inference_verifiers_res):
    return f'''
Inference
{initial_inference}

Feedback
{inference_verifiers_res}

Current Task: 
You are well trained in logical reasoning and verification and understand JSON format.  Based on the inference and feedback provieded, improve the soundness of the inference.

The output should be of the JSON format: 
```json
{{
    "relevant_facts": "",
    "key_question_answer": "",
    "hypothesis_finding": "",
}}
```
'''