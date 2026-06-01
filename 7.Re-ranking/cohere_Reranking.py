from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereRerank
from dotenv import load_dotenv
from langchain_classic.retrievers import ContextualCompressionRetriever

load_dotenv()

# Define 30 source documents spanning ML, Generative AI, and Cloud
docs = [

    # ── ML (1–10) ──────────────────────────────────────────────────────────────

    Document(page_content="Transformers rely on self-attention mechanisms to weigh the importance of different tokens in a sequence. Unlike RNNs, transformers process all tokens in parallel, making them highly efficient on modern GPU hardware. This architecture underpins nearly every state-of-the-art NLP model today."),

    Document(page_content="Gradient descent is the backbone of training neural networks. By computing the gradient of the loss function with respect to each weight, the optimizer nudges parameters in the direction that reduces error. Variants like Adam and RMSProp adapt the learning rate per parameter for faster convergence."),

    Document(page_content="Overfitting occurs when a model learns the training data too well, including its noise, and fails to generalize to unseen examples. Regularization techniques such as L2 weight decay, dropout, and early stopping are commonly used to combat overfitting and improve validation performance."),

    Document(page_content="Convolutional Neural Networks (CNNs) are designed for grid-structured data like images. Convolutional layers apply learnable filters that detect local features such as edges and textures. Pooling layers then downsample feature maps, reducing spatial dimensions while preserving the most important activations."),

    Document(page_content="Support Vector Machines (SVMs) find the optimal hyperplane that maximally separates classes in feature space. The kernel trick allows SVMs to operate in high-dimensional spaces without explicitly computing the transformation, making them effective for non-linearly separable datasets."),

    Document(page_content="Random forests are an ensemble method that builds many decision trees on random subsets of the data and features. Predictions are aggregated by majority vote for classification or averaging for regression. This bagging strategy reduces variance and makes the model robust to noisy features."),

    Document(page_content="Batch normalization normalizes the activations of each layer across a mini-batch during training. This stabilizes the learning process, allows higher learning rates, and acts as a mild regularizer. It has become a standard component in deep architectures such as ResNet and EfficientNet."),

    Document(page_content="Recurrent Neural Networks (RNNs) process sequential data by maintaining a hidden state that is updated at each time step. Long Short-Term Memory (LSTM) cells address the vanishing gradient problem by using gating mechanisms to selectively remember or forget information over long sequences."),

    Document(page_content="Transfer learning allows a model pre-trained on a large dataset to be fine-tuned on a smaller, task-specific dataset. This dramatically reduces the data and compute required for downstream tasks. Techniques like full fine-tuning, layer freezing, and LoRA offer different trade-offs between cost and performance."),

    Document(page_content="Dimensionality reduction techniques such as PCA and t-SNE compress high-dimensional data into lower-dimensional representations. PCA finds linear directions of maximum variance, while t-SNE preserves local neighborhood structure for visualization. Both are widely used in exploratory data analysis and feature engineering."),

    # ── Generative AI (11–20) ──────────────────────────────────────────────────

    Document(page_content="Large Language Models (LLMs) are trained on massive text corpora using a next-token prediction objective. By scaling model size, data, and compute, these models develop emergent capabilities such as reasoning, summarization, and code generation that were not explicitly trained for."),

    Document(page_content="Retrieval-Augmented Generation (RAG) combines a retrieval system with a generative model. Relevant documents are fetched from a vector store at inference time and injected into the prompt as context. This grounds the model's response in factual sources and reduces hallucination without retraining."),

    Document(page_content="Diffusion models generate images by learning to reverse a gradual noising process. During training, Gaussian noise is progressively added to an image. The model learns to denoise step by step. At inference, starting from pure noise, the model iteratively refines the output into a coherent image."),

    Document(page_content="Prompt engineering is the practice of crafting input prompts to elicit desired outputs from a language model. Techniques include zero-shot prompting, few-shot examples, chain-of-thought reasoning, and role assignment. Well-designed prompts can significantly improve accuracy without any model fine-tuning."),

    Document(page_content="Reinforcement Learning from Human Feedback (RLHF) aligns language models with human preferences. A reward model is trained on human comparisons of model outputs, and the LLM is then fine-tuned using PPO to maximize this reward. RLHF is central to models like InstructGPT and Claude."),

    Document(page_content="Vector embeddings map text, images, or other data into dense numerical vectors where semantic similarity corresponds to geometric proximity. Embedding models like text-embedding-ada-002 encode meaning into hundreds of dimensions. These vectors power semantic search, clustering, and recommendation systems."),

    Document(page_content="Generative Adversarial Networks (GANs) consist of a generator and a discriminator trained in opposition. The generator produces synthetic samples while the discriminator tries to distinguish them from real data. This adversarial dynamic pushes the generator toward producing increasingly realistic outputs."),

    Document(page_content="Hallucination in LLMs refers to confident generation of factually incorrect or fabricated information. It arises because LLMs optimize for fluent, plausible text rather than strict factual accuracy. Mitigation strategies include RAG, self-consistency checks, and grounding outputs with citations."),

    Document(page_content="Attention mechanisms allow models to dynamically focus on different parts of the input when producing each output token. In multi-head attention, multiple attention heads run in parallel, each learning different relationships. This enables transformers to capture both local syntactic and long-range semantic dependencies."),

    Document(page_content="Fine-tuning adapts a pre-trained LLM to a specific domain or task by continuing training on a curated dataset. Parameter-efficient methods like LoRA inject small trainable matrices into attention layers, reducing GPU memory requirements dramatically. Fine-tuned models typically outperform prompting alone on narrow tasks."),

    # ── Cloud (21–30) ──────────────────────────────────────────────────────────

    Document(page_content="Kubernetes is an open-source container orchestration platform that automates the deployment, scaling, and management of containerized applications. It groups containers into pods, manages their lifecycle across a cluster of nodes, and provides built-in load balancing, self-healing, and rolling update capabilities."),

    Document(page_content="Serverless computing allows developers to run code without provisioning or managing servers. Cloud providers like AWS Lambda, Google Cloud Functions, and Azure Functions automatically allocate resources on demand and charge only for execution time. This model is ideal for event-driven workloads with unpredictable traffic."),

    Document(page_content="Infrastructure as Code (IaC) treats cloud resources as versioned, declarative configuration files. Tools like Terraform and AWS CloudFormation allow teams to provision and manage infrastructure reproducibly. IaC enables consistent environments, peer review of changes, and automated rollback on failure."),

    Document(page_content="Cloud object storage services such as Amazon S3, Google Cloud Storage, and Azure Blob Storage provide highly durable and scalable storage for unstructured data. Data is organized into buckets and objects, replicated across multiple availability zones, and accessible via standard HTTP APIs with fine-grained access controls."),

    Document(page_content="A Content Delivery Network (CDN) distributes static and dynamic content across geographically distributed edge servers. When a user requests a resource, it is served from the nearest edge location, reducing latency and offloading traffic from the origin server. CDNs are critical for global web application performance."),

    Document(page_content="Auto-scaling dynamically adjusts the number of compute instances based on real-time demand metrics such as CPU utilization or request rate. Horizontal scaling adds or removes instances while vertical scaling changes instance size. Auto-scaling ensures cost efficiency during low traffic and reliability during peak load."),

    Document(page_content="A Virtual Private Cloud (VPC) is a logically isolated network within a public cloud provider's infrastructure. Users define subnets, route tables, and security groups to control traffic flow. VPCs enable organizations to deploy cloud resources with the same network segmentation and security controls as on-premises environments."),

    Document(page_content="Managed database services such as Amazon RDS, Google Cloud SQL, and Azure Database for PostgreSQL handle provisioning, patching, backups, and failover automatically. They support multi-AZ replication for high availability and read replicas to horizontally scale read-heavy workloads without operational overhead."),

    Document(page_content="DevOps CI/CD pipelines automate the process of building, testing, and deploying application code. Tools like GitHub Actions, GitLab CI, and Jenkins trigger pipelines on code commits, run unit and integration tests, build container images, and deploy to staging or production environments with minimal manual intervention."),

    Document(page_content="Cloud cost optimization involves right-sizing instances, using spot or preemptible VMs for fault-tolerant workloads, and purchasing reserved capacity for predictable usage. FinOps practices bring together engineering and finance teams to create shared accountability for cloud spend and align infrastructure decisions with business value."),
]

# Split documents into smaller chunks for better retrieval performance
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

#initialise the embedding model and vector store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# buil in memory vector store from documents chunks
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding_model,
    collection_name="cohere_reranking_example",
)
print(f"Total documents in vector store: {vectorstore._collection.count()}\n")

#create base retriever from vector store top 5 results
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# query to test the retriever
# query1 = "What are the key differences between RNNs and transformers in NLP?"
query1 = "What are the best practices for scaling compute infrastructure during traffic spikes?"
print(f"Query: {query1}\n")
base_results1 = retriever.invoke(query1)
for i, doc in enumerate(base_results1):
    print(f"BASE RESULTS {i+1}:\n{doc.page_content}\n")

print("-" * 80)

# create cohere reranker and rerank the retrieved results
compressor = CohereRerank(model="rerank-english-v3.0" ) # top_n=3
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=retriever
    )    
re_ranked_results1 = compression_retriever.invoke(query1)
for i, doc in enumerate(re_ranked_results1):
    print(f"Re-ranked RESULTS {i+1}:\n{doc.page_content}\n")
