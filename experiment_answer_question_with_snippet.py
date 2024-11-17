import ollama # requires you to install & start the ollama service https://ollama.com/ 
import json
from typing import TypedDict

model = 'llama3.1:8b' # alternatively 'llama3.1:70b'

class GoldenDataSnippet(TypedDict):
    offsetInBeginSection: int
    offsetInEndSection: int
    text: str
    beginSection: str
    document: str
    endSection: str

class GoldenData(TypedDict):
    question_text: str
    documents: list[str]
    ideal_answer: list[str]
    snippets: list[GoldenDataSnippet]

class Answer(TypedDict):
    question: str
    result: str
    target: str

def answerQuestionWithContext(question: str, relevant_paper_contents: list[str]) -> str:
    prompt = "Given the following context:\n" + "\n".join(relevant_paper_contents) + "\n\nPlease provide a brief answer to the following question: " + question
    result = ollama.generate(model=model, prompt=prompt)
    return result['response']

def loadGoldenData() -> list[GoldenData]:
    with open('./dataset/12B1_golden.json', 'r') as file:
        return [
            ({
                'question_text': question['body'],
                'documents': [doc.split('/')[-1] for doc in question['documents']],
                'ideal_answer': question['ideal_answer'],
                'snippets': question['snippets']
            })
            for question in json.load(file)['questions']
        ]

def main():
    train = loadGoldenData()
    answers: list[Answer] = []
    
    for idx, question in enumerate(train):
        relevant_paper_contents = [snippet['text'] for snippet in question['snippets']]
        answer = answerQuestionWithContext(question['question_text'], relevant_paper_contents)
        answers.append({
            'question': question['question_text'],
            'result': answer,
            'target': question['ideal_answer'][0]
        })
        print(f"{idx}/{len(train)}\n\tinput=\"{question['question_text']}\"\n\tanswer=\"{answer}\"\n\texpected=\"{question['ideal_answer'][0]}\"")

    with open('results.json', 'w') as outfile:
        json.dump(answers, outfile, indent=4)

if __name__ == '__main__':
    main()
