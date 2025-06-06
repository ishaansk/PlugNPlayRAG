This project implements an **Agentic Retrieval-Augmented Generation (Agentic RAG)** system that combines advanced retrieval techniques with large language models (LLMs) to deliver accurate, context-aware, and reliable AI-generated responses. Unlike traditional RAG systems, this project incorporates autonomous agents that iteratively retrieve, evaluate, and refine information to handle complex, multi-step reasoning tasks effectively.

There are TWO approaches included in this project, through Streamlit and Flask, apart from just the frontend implementation, these two approaches also have different workings. The Flask App has Agentic iterative query refinement loops to ensure the best output through calculating context sufficiency. The Streamlit app uses static agent modes, namely Auto, Reasoning, Summarise and Number Focused, so as to provide the option of picking how you want the chatbot to approach a query.

How to run this?

1, Clone the repository
>git clone https://github.com/ishaansk/PlugNPlayRAG

2, Install the requirements 
>pip install -r requirements.txt

3, Make a .env file in your same repository and include all the necessary APIs and Endpoints. To see which ones are needed, an env.txt is included, feel free to use it as a template. (Note : The Flask secret key can be any 48-character hex string)

4, To run the Agentic RAG Chatbot (available only through Flask),
>python flaskapp.py

After which open http://127.0.0.1:5000/ in your browser to locally host and utilise the chatbot.

5, To run the Static Agent Chatbot (available onlly through Streamlit), 
>streamlit run app.py

The app will open in a browser.
(Note : There is also a sample.txt included just for the sake of testing)



Key Features :
- Retrieval-Augmented Generation (RAG): Integrates document retrieval with language generation to enhance response relevance and accuracy.
- Agentic Autonomous Control: Uses AI agents to dynamically plan and refine queries, ensuring sufficient and high-quality context is gathered before generation.
- Embedding Models & Vector Databases: Employs embedding models to convert text into vectors, stored and searched efficiently using vector databases.
- Large Language Models (LLMs): Utilizes state-of-the-art LLMs (e.g., GPT-4) for natural language understanding and generation.
- Semantic Search & Query Optimization: Implements semantic similarity search and iterative query refinement to improve retrieval results.
- Tokenization & Usage Management: Tracks token usage for cost optimization and managing input/output limits effectively.