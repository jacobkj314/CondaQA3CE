'''
    This script takes the json from a CONDAQA dataset partition and groups it by ORIGINAL PASSAGE, then QUESTION, then EDIT
    The output is a list of( (each original passage's) list of( question and list of (edited passages and editID and label)) )

	!!!I still need some work to turn it into a DataLoader (or maybe just print out to another json file)
'''

import re, json

def json2data(url):
	with open(url, "r") as file:
		text = file.read()
	jsonArray = "[" + re.sub("\n", ",", text[:-1]) + "]" #format as json array
	jsonData = json.loads(jsonArray)
	
	passageIDs = set(d["PassageID"] for d in jsonData)
	
	def gather(pId):
		return [{"editID":d["PassageEditID"], "passage":d['sentence1'], "question":d['sentence2'], "label":d['label']} for d in jsonData if d["PassageID"] == pId]
	def sortByQuestion(gathered):
		questions = set(d["question"] for d in gathered)
		return [{"question":q, "contexts":[{"passage":e["passage"], "editID":e["editID"], "label":e["label"]} for e in gathered if e["question"] == q]} for q in questions]
	
	return [sortByQuestion(gather(i)) for i in passageIDs]

def json2bundles(url):
	data = json2data(url)
	bundles = []

	for passage in data:
		for bundle in passage:
			bundles.append(
				{
					"input": [(bundle['question'] + '\n' + context['passage']) for context in bundle['contexts']],
					"answer": [(context['label']) for context in bundle['contexts']]
				}
			)

	return bundles

def bundle(source, destination):
	bundles = json2bundles(source)

	with open(destination, "w") as out:
		out.write('')
	with open(destination, "a") as out:
		for bundle in bundles:
			out.write(json.dumps(bundle) + '\n')

bundle("data/condaqa_train.json", "data/unifiedqa_formatted_data/condaqa_train_unifiedqa.json")
