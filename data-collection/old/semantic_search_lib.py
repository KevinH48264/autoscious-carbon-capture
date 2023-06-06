from semanticscholar import SemanticScholar
import json

sch = SemanticScholar()
paper = sch.get_paper('26327e3f321410baab2c55ad5df509710f5bb296')
print(paper.abstract)

# # JSON variable
# data = paper

# # Open the file in append mode
# with open('output.txt', 'a') as file:
#     # Convert JSON to a string
#     json_string = json.dumps(data)
#     # Write the string to the file
#     file.write(json_string + '\n')

# # Regular text variable
# text = "Hello, world!"

# # Open the file in append mode
# with open('output.txt', 'a') as file:
#     # Write the text to the file
#     file.write(text + '\n')