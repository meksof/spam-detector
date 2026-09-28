- V1: TypeSafe-only classifier
The classifier has an accuracy of about 90% on the test set. However for borderline cases (threshold=2), some emails are misclassified.
So we will use an llm model to classify borderline case

- V2: TypeSafe + Ollama LLM for borderline cases
Giving the local model the responsability to provide a much more accurate classification of borderline cases. 
It seems that llm models are better at chat than at classification. And they are not able to provide a good explanation of their classification.
So we will think about a different approach to classify borderline cases in next iterations.

- V3: Replace the LLM borderline classifier with a fine-tuned BERT/MiniLM encoder (the spam-classifier project)
