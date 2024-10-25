import ollama # requires you to install & start the ollama service https://ollama.com/ 
import json
from typing import TypedDict
import sqlite3

model = 'llama3.1:8b' # alternatively 'llama3.1:70b'


class GoldenData(TypedDict):
    question_text: str
    documents: list[str]
    ideal_answer: list[str]

def answerQuestionWithContext(question: str, relevant_paper_contents: list[str]) -> str:
    prompt = "Given the following context:\n" + "\n".join(relevant_paper_contents) + "\n\nPlease provide a brief answer to the following question: " + question
    result = ollama.generate(model=model, prompt=prompt)    
    return result['response']

def loadDocuments(conn: sqlite3.Connection, document_pmids: list[str]) -> list[str]:
    placeholders = ','.join('?' for _ in document_pmids)
    query = f"SELECT content FROM articles WHERE pmid IN ({placeholders})"
    cursor = conn.execute(query, document_pmids)
    rows = cursor.fetchall()
    
    # disabled for now
    # if len(rows) != len(document_pmids):
    #     print("Error: The number of retrieved documents does not match the number of requested PMIDs.")

    return [row[0] for row in rows]

def loadGoldenData() -> list[GoldenData]:
    with open('./dataset/12B1_golden.json', 'r') as file:
        return [
            ({
                'question_text': question['body'],
                'documents': [doc.split('/')[-1] for doc in question['documents']],
                'ideal_answer': question['ideal_answer']
            })
            for question in json.load(file)['questions']
        ]

def main():
    train = loadGoldenData()
    conn = sqlite3.connect("./dataset/db.sqlite")
    
    for question in train:
        if (train.index(question) != 80): # debug
            continue
        
        relevant_paper_contents = loadDocuments(conn, question['documents'])
        answer = answerQuestionWithContext(question['question_text'], relevant_paper_contents)
        print(f"Question:\n{question['question_text']}\n\nAnswer:\n{answer}\n\nIdeal Answer[0]:\n{question['ideal_answer'][0]}")    

if __name__ == '__main__':
    main()
