import wordnet_module
from bert_module import similarity

word = 'paper'

meanings = wordnet_module.get_meanings(word)
print(f"Meanings of '{word}':")
for meaning in meanings:
    print(f"- {meaning}")

parents = wordnet_module.get_parent(word)
print(f"\nParents of '{word}':")
for parent in parents:
    print(f"- {', '.join(parent)}")

text1 = "software"
text2 = "program"

similarity_score = similarity(text1, text2)
print(f"The similarity between '{text1}' and '{text2}' is: {similarity_score}")