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

Decomposition
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