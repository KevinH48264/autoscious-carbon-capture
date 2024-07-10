# Autoscious Carbon Capture
Research project to use AI to generate experiments for researchers to better tackle carbon capture

## Codebase Structure
This codebase is modeled after: https://www.moderndescartes.com/essays/research_code/

### 0_core
- Core. Libraries for reusable components like cloud data storage, notebook tooling, neural network libraries, model serialization/deserialization, statistics tests, visualization, testing libraries, hyperparameter optimization frameworks, wrappers and convenience functions built on top of third-party libraries. Engineers typically work here.
- Code is reviewed to engineering standards. Code is tested, covered by continuous integration, and should never be broken. Very low tolerance for tech debt.
- Breaking changes to core code should be accompanied by fixes to affected project code. The project owner should assist in identifying potential breakage. No need to fix experimental code.

### 1_projects
- Projects. A new top-level folder for each major effort (rough criteria: a project represents 1-6 months of work). Engineers and researchers work here.
Code is reviewed for correctness. Testing is recommended but optional, as is continuous integration.
- No cross-project dependencies. If you need code from a different project, either go through the effort of polishing the code into core, or clone the code.

### 2_experimental
- Experimental. Anything goes. Typically used by researchers. I suggest namespacing by time (e.g. a new directory every month).
- Rubber-stamp approvals. Code review is optional and comments may be ignored without justification. Do not plug this into continuous integration.
- The goal of this directory is to create a safe space for researchers so that they do not need to hide their work. By passively observing research code “in the wild”, engineers can understand research pain points.
- Any research result that is shared outside the immediate research group may not be derived from experimental code.