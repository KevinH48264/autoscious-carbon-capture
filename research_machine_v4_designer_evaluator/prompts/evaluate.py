def get_evaluate_prompt(problem, approach):
    return f'''
Task:
Show that this approach can't work to solve the research problem.

Research problem: 
{problem}

Approach:
{approach}
'''

def extract_failure_modes_prompt(problem, approach):
    return f'''
Task:    
What are the fundamental principles explaining why this approach cannot work? I only want fundamental principles that completely rule out this approach, otherwise, give me none.

Research problem: 
{problem}

Approach:
{approach}
'''

def extract_useful_info_prompt(problem, approach):
    return f'''
Task:    
What are the fundamental principles explaining which parts of the approach are most useful? I only want useful information that could help in future designs, otherwise, give me none.

Research problem: 
{problem}

Approach:
{approach}
'''