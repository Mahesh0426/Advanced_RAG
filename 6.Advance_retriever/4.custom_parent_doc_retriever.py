from dotenv import load_dotenv
load_dotenv()

import uuid
from typing import Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.storage import InMemoryStore


embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

docs = [
    Document(
        page_content=(
            'Deep learning is a subfield of machine learning that uses neural networks with many layers '
            'to learn representations of data with multiple levels of abstraction. The field was '
            'revolutionized in 2012 when AlexNet won the ImageNet competition by a wide margin, '
            'demonstrating the power of convolutional neural networks trained on GPUs. Deep learning '
            'models learn hierarchical representations: lower layers capture simple features like edges '
            'and colors, while deeper layers capture complex abstractions like faces or objects.\n\n'
            'The fundamental building block is the artificial neuron, which computes a weighted sum of '
            'its inputs, adds a bias term, and applies a non-linear activation function such as ReLU, '
            'sigmoid, or tanh. Layers of neurons are stacked to form deep networks capable of '
            'approximating any continuous function, as guaranteed by the universal approximation theorem.\n\n'
            'Convolutional Neural Networks (CNNs) are specialized architectures for grid-like data such '
            'as images. They use convolutional filters that slide across the input, sharing parameters '
            'across spatial locations to efficiently learn spatial hierarchies of features. Pooling '
            'layers reduce spatial dimensions while preserving important features. Modern CNN '
            'architectures like ResNet, VGG, and EfficientNet have achieved human-level performance on '
            'image classification tasks.\n\n'
            'Recurrent Neural Networks (RNNs) are designed for sequential data. They maintain a hidden '
            'state that captures information from previous time steps, enabling them to model temporal '
            'dependencies. Vanilla RNNs suffer from vanishing gradients over long sequences. Long '
            'Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs) address this by using '
            'gating mechanisms that selectively remember and forget information across many time steps.\n\n'
            'The Transformer architecture, introduced in Attention Is All You Need (2017), revolutionized '
            'deep learning by replacing recurrence with self-attention mechanisms. Self-attention computes '
            'relationships between all pairs of positions in a sequence in parallel, capturing long-range '
            'dependencies more effectively than RNNs. Transformers form the backbone of large language '
            'models. The architecture consists of encoder and decoder stacks with multi-head attention, '
            'feed-forward networks, residual connections, and layer normalization.\n\n'
            'Training deep networks requires careful optimization. Backpropagation computes gradients '
            'through the network using the chain rule. Optimizers like Adam and SGD with momentum adapt '
            'learning rates to speed convergence. Regularization techniques such as dropout, batch '
            'normalization, weight decay, and early stopping prevent overfitting. Transfer learning '
            'allows pre-trained models to be fine-tuned on downstream tasks with limited data, '
            'dramatically reducing training time and data requirements across domains.'
        ),
        metadata={'topic': 'deep_learning'},
    ),
    Document(
        page_content=(
            'Machine learning is the science of getting computers to learn from data without being '
            'explicitly programmed. The field spans supervised learning, unsupervised learning, '
            'semi-supervised learning, and self-supervised learning, each addressing different problem '
            'settings and data availability constraints.\n\n'
            'Supervised learning trains models on labeled input-output pairs to learn a mapping function. '
            'Classification predicts discrete labels while regression predicts continuous values. Key '
            'algorithms include linear and logistic regression, which model linear relationships and are '
            'highly interpretable. Support Vector Machines find the maximum-margin hyperplane separating '
            'classes and use the kernel trick to handle non-linear boundaries by implicitly mapping data '
            'into high-dimensional feature spaces.\n\n'
            'Decision trees recursively partition the feature space using binary splits that maximize '
            'information gain or minimize Gini impurity. While interpretable, individual trees overfit '
            'easily. Ensemble methods combat this: Bagging trains multiple trees on random data subsets '
            'and averages predictions, as in Random Forest. Boosting trains models sequentially, with '
            'each model correcting predecessors — Gradient Boosting and XGBoost are among the most '
            'powerful algorithms for structured tabular data competitions.\n\n'
            'The bias-variance tradeoff is central to machine learning. Bias refers to error from '
            'incorrect assumptions in the model, leading to underfitting. Variance refers to sensitivity '
            'to fluctuations in training data, leading to overfitting. Regularization techniques such '
            'as L1 (Lasso), L2 (Ridge), and Elastic Net add penalty terms to the loss function to '
            'control model complexity. Cross-validation estimates generalization performance by rotating '
            'the held-out validation set across multiple folds.\n\n'
            'Unsupervised learning discovers hidden structure in unlabeled data. Clustering algorithms '
            'like K-Means, DBSCAN, and hierarchical clustering group similar data points. Dimensionality '
            'reduction methods like PCA and t-SNE transform high-dimensional data into lower-dimensional '
            'representations while preserving important structure. Generative models like Variational '
            'Autoencoders (VAEs) and Generative Adversarial Networks (GANs) learn to generate new data '
            'samples that resemble the training distribution.\n\n'
            'Feature engineering is often the most impactful step in building machine learning systems. '
            'It involves transforming raw data into representations that algorithms can exploit. '
            'Techniques include normalization, one-hot encoding, polynomial features, and domain-specific '
            'feature creation. Modern deep learning reduces the need for manual feature engineering by '
            'automatically learning features from raw inputs, but structured tabular data still benefits '
            'greatly from expert-driven feature construction and selection strategies.'
        ),
        metadata={'topic': 'machine_learning'},
    ),
    Document(
        page_content=(
            'Natural Language Processing (NLP) is the branch of artificial intelligence concerned with '
            'enabling computers to understand, interpret, and generate human language. Language presents '
            'unique challenges: it is ambiguous, context-dependent, hierarchically structured, and '
            'governed by implicit social conventions. Early NLP relied on hand-crafted rules and '
            'grammars, but modern NLP is dominated by statistical and neural approaches trained on '
            'large text corpora.\n\n'
            'Tokenization is the fundamental preprocessing step that breaks text into tokens: words, '
            'subwords, or characters. Modern subword tokenization schemes like Byte-Pair Encoding (BPE), '
            'WordPiece, and SentencePiece balance vocabulary size with coverage of rare words. Word '
            'embeddings like Word2Vec and GloVe represent words as dense vectors where semantically '
            'similar words are geometrically close. Word2Vec captures analogical relationships '
            'like king minus man plus woman equals queen from geometric structure alone.\n\n'
            'The attention mechanism was a pivotal innovation allowing models to focus on relevant parts '
            'of the input when producing each output token. The self-attention mechanism in Transformers '
            'computes query, key, and value projections from the same sequence, enabling each token to '
            'attend to every other token in parallel. Multi-head attention runs several attention '
            'operations in parallel, capturing different types of syntactic and semantic relationships '
            'simultaneously.\n\n'
            'BERT (Bidirectional Encoder Representations from Transformers) pre-trains a Transformer '
            'encoder on masked language modeling and next sentence prediction, learning deep bidirectional '
            'representations. Fine-tuning BERT on downstream tasks like question answering, sentiment '
            'analysis, and named entity recognition achieved state-of-the-art results across many '
            'NLP benchmarks. GPT-style models use causal self-attention and are pre-trained on '
            'next-token prediction, then prompted in zero-shot and few-shot settings.\n\n'
            'Modern large language models such as GPT-4, LLaMA, and Gemini are scaled-up Transformer '
            'decoders trained on trillions of tokens from diverse internet sources. Reinforcement '
            'Learning from Human Feedback (RLHF) aligns these models to follow instructions and avoid '
            'harmful outputs. Retrieval-Augmented Generation (RAG) combines LLMs with external '
            'knowledge retrieval to reduce hallucination and enable knowledge updates without '
            'retraining the base model parameters.'
        ),
        metadata={'topic': 'natural_language_processing'},
    ),
    Document(
        page_content=(
            'Computer vision is the field of artificial intelligence that enables machines to interpret '
            'and understand visual information from the world. Early approaches used hand-crafted feature '
            'descriptors like SIFT, HOG, and SURF, feeding them into classical classifiers such as SVMs. '
            'The deep learning revolution, particularly CNNs, transformed computer vision into one of '
            "AI's greatest success stories.\n\n"
            'Image classification assigns a label to an entire image. The ImageNet challenge benchmarked '
            'progress on classifying images into 1000 categories. AlexNet (2012), VGGNet (2014), '
            'GoogLeNet (2014), ResNet (2015), and EfficientNet (2019) represent successive generations '
            'achieving progressively better accuracy. ResNet introduced residual (skip) connections, '
            'allowing gradients to flow through very deep networks of over 100 layers without vanishing '
            'during backpropagation.\n\n'
            'Object detection localizes and classifies multiple objects within an image. Two-stage '
            'detectors like Faster R-CNN first propose candidate regions, then classify them. '
            'Single-stage detectors like YOLO predict bounding boxes and classes in one forward pass, '
            'enabling real-time detection. Modern architectures like DETR use Transformers for end-to-end '
            'object detection without hand-designed anchor boxes, simplifying the detection pipeline '
            'considerably.\n\n'
            'Semantic segmentation classifies every pixel into a category. Fully Convolutional Networks '
            'replaced dense layers with convolutional layers to produce dense pixel-wise predictions. '
            'The U-Net architecture uses an encoder-decoder structure with skip connections to preserve '
            'fine-grained spatial details. Instance segmentation additionally distinguishes between '
            'individual objects of the same class, which Mask R-CNN addresses by adding a mask '
            'prediction branch to the Faster R-CNN framework.\n\n'
            'Vision Transformers (ViTs) apply the Transformer architecture to images by splitting them '
            'into fixed-size patches and treating each patch as a token. ViTs match CNN performance '
            'when trained on large datasets. Self-supervised pre-training methods like SimCLR, MoCo, '
            'and DINO learn visual representations from unlabeled images by maximizing agreement between '
            'different augmented views of the same image through contrastive learning. Data augmentation '
            'strategies — random cropping, flipping, color jitter, Mixup, CutMix — are critical for '
            'training robust vision models with limited labeled data.'
        ),
        metadata={'topic': 'computer_vision'},
    ),
    Document(
        page_content=(
            'Reinforcement learning (RL) is the computational framework for learning through interaction '
            'with an environment. An agent takes actions, receives scalar reward signals, and learns a '
            'policy that maximizes cumulative long-term reward. Unlike supervised learning, the agent '
            'receives no ground-truth labels — only reward signals that may be delayed, sparse, and '
            'noisy. This makes RL well-suited for sequential decision-making problems.\n\n'
            'The mathematical foundation of RL is the Markov Decision Process (MDP), defined by states, '
            'actions, a transition function, a reward function, and a discount factor. The discount '
            'factor balances immediate and future rewards. The goal is to find a policy — a mapping '
            'from states to actions — that maximizes expected discounted cumulative reward. The '
            'value function V(s) represents the expected return from state s; the action-value '
            'function Q(s,a) represents the expected return after taking action a in state s.\n\n'
            'Q-learning is an off-policy temporal difference method that learns Q-values by bootstrapping '
            'from estimates of future returns. Deep Q-Networks (DQN) combined Q-learning with deep '
            'neural networks and experience replay to achieve superhuman performance on Atari games. '
            'Double DQN, Dueling DQN, and Prioritized Experience Replay addressed known instabilities '
            'and improved sample efficiency across many game environments.\n\n'
            'Policy gradient methods directly optimize the policy by computing gradients of the expected '
            'return with respect to policy parameters. Actor-Critic methods combine value function '
            'estimation (the critic) with direct policy optimization (the actor), reducing variance '
            'while maintaining unbiasedness. Proximal Policy Optimization (PPO) and Trust Region Policy '
            'Optimization (TRPO) constrain policy updates to prevent catastrophically large steps, '
            'stabilizing training on complex continuous control tasks.\n\n'
            'AlphaGo and AlphaZero demonstrated superhuman game-playing through self-play and Monte Carlo '
            'Tree Search, discovering strategies that astonished professional players. Offline RL learns '
            'from fixed datasets without further environment interaction, enabling applications where '
            'online data collection is dangerous. Model-based RL accelerates learning by using a learned '
            'world model for planning. Real-world RL applications span robot manipulation, autonomous '
            'driving, drug discovery, chip design, and energy grid optimization.'
        ),
        metadata={'topic': 'reinforcement_learning'},
    ),
]

print(f'Created {len(docs)} documents')

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
child_splitter  = RecursiveCharacterTextSplitter(chunk_size=400,  chunk_overlap=50)

#
class CustomParentDocumentRetriever(BaseRetriever):
    """Retriever that indexes small child chunks for search but returns larger parent chunks."""

    vectorstore: Any    # Chroma — stores child chunks for similarity search
    docstore: Any       # InMemoryStore — stores parent chunks keyed by UUID
    child_splitter: Any
    parent_splitter: Any
    id_key: str = 'parent_doc_id'

    def add_documents(self, docs: list[Document]) -> None:
        # Step 1 & 2: split raw documents into parent chunks
        parent_chunks = self.parent_splitter.split_documents(docs)
        for parent in parent_chunks:
            # Step 2: assign a unique ID to each parent chunk
            parent_id = str(uuid.uuid4())
            # Step 3: split each parent into smaller child chunks
            children = self.child_splitter.split_documents([parent])
            # tag every child with its parent's ID so we can look it up later
            for child in children:
                child.metadata[self.id_key] = parent_id
            # Step 3: store children in ChromaDB for semantic similarity search
            self.vectorstore.add_documents(children)
            # Step 4: store the parent in InMemoryStore keyed by its UUID
            self.docstore.mset([(parent_id, parent)]) # mset --> dict1[parent_id] = parent

    def _get_relevant_documents(self, query: str) -> list[Document]:
        # Step 5: similarity search retrieves the most relevant child chunks
        child_docs = self.vectorstore.similarity_search(query, k=3)
        # Step 6: collect unique parent IDs from child metadata (dict.fromkeys preserves order)
        parent_ids = list(dict.fromkeys(
            doc.metadata[self.id_key] for doc in child_docs
            if self.id_key in doc.metadata
        ))
        # Step 6: fetch and return the corresponding parent documents
        return [doc for doc in self.docstore.mget(parent_ids) if doc is not None] # mget --> dict1[parent_id]


vectorstore = Chroma(collection_name='custom_children', embedding_function=embeddings)

docstore = InMemoryStore()

retriever = CustomParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(docs)

parent_count = len(list(docstore.yield_keys()))
child_count  = len(vectorstore.get()['ids'])
print(f'Parent chunks: {parent_count}  |  Child chunks: {child_count}\n')

query = 'How do transformer architectures work in deep learning?'

results = retriever.invoke(query)
print(f'Retrieved {len(results)} parent chunk(s):\n')
for i, doc in enumerate(results):
    print(f'--- Result {i+1} [{doc.metadata.get("topic")}] ---')
    print(doc.page_content)
    print()