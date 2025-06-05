class AgenticRAGController:
    def __init__(self, embedder, db, llm, max_iterations=3, top_k=5, score_threshold=0.1):
        self.embedder = embedder
        self.db = db
        self.llm = llm
        self.max_iterations = max_iterations
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(self, query_embedding):
        results = self.db.query(vector=query_embedding, top_k=self.top_k)
        matches = results.get("matches", [])
        return [m for m in matches if m.get("score", 0) >= self.score_threshold]

    def analyze_sufficiency(self, context, question):
        # Ask the LLM if the context is sufficient to answer the question
        prompt = (
            f"Given the following context:\n{context}\n"
            f"Is this information sufficient to answer the question: '{question}'? "
            "Answer 'Yes' or 'No' with a brief explanation."
        )
        response = self.llm.ask(prompt=prompt, context=context, temperature=0, max_tokens=50)
        return "yes" in response.lower()

    def refine_query(self, previous_query, context):
        # Use LLM to generate a refined query based on previous query and retrieved context
        prompt = (
            f"The previous query was: '{previous_query}'.\n"
            f"The retrieved context was:\n{context}\n"
            "Refine the query to get more relevant information."
        )
        refined_query = self.llm.ask(prompt=prompt, context=context, temperature=0.7, max_tokens=50)
        return refined_query.strip()

    def run(self, question):
        query = question
        aggregated_context = ""
        for i in range(self.max_iterations):
            query_embedding = self.embedder.generate_embeddings([query])[0].tolist()
            matches = self.retrieve(query_embedding)
            if not matches:
                break
            context = "\n\n".join([m["metadata"]["text"] for m in matches])
            aggregated_context += context + "\n\n"

            if self.analyze_sufficiency(aggregated_context, question):
                # If sufficient, generate final answer
                answer = self.llm.ask(prompt=question, context=aggregated_context)
                return answer
            else:
                # Refine query for next iteration
                query = self.refine_query(query, aggregated_context)

        # Fallback: generate answer with whatever context we have
        return self.llm.ask(prompt=question, context=aggregated_context)
