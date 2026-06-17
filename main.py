pdf  = "data/policies/policy1.pdf"
questions="What is a hospital."
ques1 ="Is Air Ambulance covered under Domestic paitient"
from ingestion.ingest_pipeline import ingest_document
from query.query_pipeline import ask_question

# print("INgesting docs.. \n")
# ingest_document(pdf)
#query the quesrtion and run the chain
print("Querying the question")
response = ask_question(ques1)

print(response)
