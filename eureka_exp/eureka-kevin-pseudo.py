def gen_eureka(
        curriculum_agent, # curriculum agent for proposing the next task
        action_agent, # action agent for code or natural language generation
        execution_agent, # execution agent that uses code or natural language generation?
        critic_agent, # critic agent for self-verification
        skill_manager, # skill manager for adding new skills and skill retrieval
):
    while True:
        exploration_progress = (
            curriculum_agent.get_exploration_progress(
                curriculum_agent.get_completed_tasks(),
                curriculum_agent.get_failed_tasks(),
            ) # this should contain inventory of where we're at now and what files we have / memory stream
        )
        task = curriculum_agent.propose_next_task(
            skill_manager.get_most_relevant_skills(), skill_manager.get_info_blocks(), exploration_progress
        ) # Assuming that the curriculum agent will come up with a task to look for more info. OR, if it can't propose a reasonable new task and needs more info, it can NOT set a new task and instead focus on using existing skills to gather more information (thinking, writing, reading, searching). I guess the outcome can become summarized as a new skill as learnings if significant, and then the content is the output. Trust that there will be enough info and the recency score will help add some randomness to the propose_new_task function.
        methods_prompt = None
        execution_feedback = None
        execution_errors = None
        critique = None
        success = False

        # try at most 4 rounds before moving on to the next task
        for i in range (4) :
            skills = skill_manager.retrieve_skills(
                task, execution_feedback
            ) # skills are information blocks (Q&A, and reading the file will reveal the reasoning)
            methods_prompt = action_agent.generate_function_callable_prompt(
                task,
                methods_prompt,
                execution_feedback,
                execution_errors,
                critique,
                skills
            ) # if the agent thinks that it can't do it with the existing actions, then it should generate subtasks (task decomposition)
            execution_feedback, execution_errors = execution_agent.function_call(methods_prompt) # TODO: handle execution errors to route back
            success, critique = critic_agent.check_task_success(
                task, execution_feedback
            ) # The critic can check 1) is the method sound and 2) does the execution feedback actually answer the task / question
            if success:
                break
        if success:
            skill_manager.add_skill(methods_prompt) # LLM can make a function name based on task/question, final answer (extracted from ExecutionAgent) can go in description, and method prompt can be returned if the LLM wants to figure out more information on how something was figured out
            curriculum_agent.add_completed_task(task)
        else:
            curriculum_agent.add_failed_task(task)

# Template pseudocode from Voyager: https://arxiv.org/pdf/2305.16291.pdf, System aligned with Eureka

# Problems:
# P: Where is there time for the agent to gather more information? Because the search part should actually add whatever info it found into its skills (if it knows more). Specifically in the format of discovery-name: methods & reasoning.
# A: Oh interesting, there's actually just an "exploreUntil" so that you can do paper = exploreUntil(paper) and you'll get a paper from search. If the curriculum agent can't propose a next task, then it can information gather.

# P: skill_manager.get_most_relevant_skills() cannot get all skills. 
# A: get most relevant skills by cosine similarity

# Assumptions
# Skill representation
# Ex. skill (should be executable via function call): 
# From task: Figure out what are the problems that lead to oxygen instability in electroswing.
# Skillname: Reasons for oxygen instability in electroswing.
# Content: Methods that are executable via function call, OR natural language answer.

# Skillname: Conduct a literature review
# Content: methods executable via function call

# Pre-programmed skills -- all runnable via function calling:
# think-aloud(prompt)
# write(file, content)
# read(file)
# search(query)