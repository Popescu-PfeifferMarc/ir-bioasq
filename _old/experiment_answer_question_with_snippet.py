import ollama  # requires you to install & start the ollama service https://ollama.com/
import json
from typing import TypedDict

model = "llama3.1:8b"  # alternatively 'llama3.1:70b'


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
    prompt = (
        "Given the following context:\n"
        + "\n".join(relevant_paper_contents)
        + "\n\nPlease provide a brief answer to the following question: "
        + question
    )
    result = ollama.generate(model=model, prompt=prompt)
    return result["response"]


def loadGoldenData() -> list[GoldenData]:
    with open("./dataset/12B1_golden.json", "r") as file:
        return [
            (
                {
                    "question_text": question["body"],
                    "documents": [doc.split("/")[-1] for doc in question["documents"]],
                    "ideal_answer": question["ideal_answer"],
                    "snippets": question["snippets"],
                }
            )
            for question in json.load(file)["questions"]
        ]


def answerAllQuestions():
    train = loadGoldenData()
    answers: list[Answer] = []

    for idx, question in enumerate(train):
        relevant_paper_contents = [snippet["text"] for snippet in question["snippets"]]
        answer = answerQuestionWithContext(
            question["question_text"], relevant_paper_contents
        )
        answers.append(
            {
                "question": question["question_text"],
                "result": answer,
                "target": question["ideal_answer"][0],
            }
        )
        print(
            f"{idx}/{len(train)}\n\tinput=\"{question['question_text']}\"\n\tanswer=\"{answer}\"\n\texpected=\"{question['ideal_answer'][0]}\""
        )

    with open("results.json", "w") as outfile:
        json.dump(answers, outfile, indent=4)


def checkAnswers():
    with open("results.json", "r") as file:
        answers = json.load(file)
        out = []
        for idx, answer in enumerate(answers):
            prompt = (
                'You are evaluating an automated system answering questions. Given the question "'
                + answer["question"]
                + "\", check if the generated response matches the expected response. It's okay if the actual response contains a different emphasis or focus or there is some extra information. Ensure only that there is no incorrect data. The generated response is:\n"
                + answer["result"]
                + "\n\nThe expected response is:\n"
                + answer["target"]
                + "\n\nDo they match? Briefly explain your reasoning. ALWAYS start your response with either YES or NO."
            )
            result = ollama.generate(model=model, prompt=prompt)
            print(
                f"{idx}/{len(answers)}\n\tinput=\"{answer['question']}\"\n\tanswer=\"{answer['result']}\"\n\texpected=\"{answer['target']}\"\n\tresult=\"{result['response']}\""
            )
            out.append(
                {
                    "question": answer["question"],
                    "result": result["response"],
                    "target": answer["target"],
                    "answer": answer["result"],
                }
            )
        with open("results_evaluated.json", "w") as outfile:
            json.dump(out, outfile, indent=4)


def main():
    # answerAllQuestions()
    checkAnswers()


if __name__ == "__main__":
    main()
