"""Makes use of an online instance of Parsey McParseface, Google's
SyntaxNet pretrained dependency parser."""
import os
from pprint import pprint
import Algorithmia

input = {
    "src":
    "Algorithmia is a marketplace for algorithms. The Technological Singularity will transform Society.",
    "format": "conll",
    "language": "english"
}
api_key = os.getenv('ALGORITHMIA_API_KEY')
client = Algorithmia.client(api_key)
result = client.algo('deeplearning/Parsey/1.1.1').pipe(input).result['output']
pprint(result)

sentence_proper_nouns = []
for sentence_data in result['sentences']:
    for word in sentence_data['words']:
        if word['universal_pos'] in ["PROPN", "NOUN"]:
            sentence_proper_nouns.append(word['form'])

print(sentence_proper_nouns)
