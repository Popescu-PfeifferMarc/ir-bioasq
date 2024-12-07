# evaluation for tfidf & bm25 
# maybe can be notebook

"""
input_file # to evaluate
input_gold_file = './dataset/taskB_golden.json' # to compare against


# input_file will be this:
{
  "questions": [
    {
      "query": "Concizumab is used for which diseases?",
      "documents": [
        {
          "pmid": "http://www.ncbi.nlm.nih.gov/pubmed/37341887",
          "score": 0.9
        }
      ],
      "snippets": [
        {
          "text": "Concizumab is being developed by Novo Nordisk for the treatment of hemophilia A and B with and without inhibitors. In March 2023, concizumab was approved in Canada for the treatment of adolescent and adult patients (12 years of age or older) with hemophilia B who have FIX inhibitors and require routine prophylaxis to prevent or reduce the frequency of bleeding episodes. ",
          "pmid": "37341887",
          "score": 0.9
        }
      ]
    },
    {
        // more here
    }
  ]
}
"""
