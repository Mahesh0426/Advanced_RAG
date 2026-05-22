from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

#STEP 1 - python code 
python_code = """
import numpy as np
from typing import List, Optional

def calculate_mean(numbers: List[float]) -> float:
    '''Calculate the arithmetic mean of a list of numbers.
    
    Args:
        numbers: List of numerical values
        
    Returns:
        The mean value
    '''
    return sum(numbers) / len(numbers)

def calculate_median(numbers: List[float]) -> float:
    '''Calculate the median of a list of numbers.'''
    sorted_nums = sorted(numbers)
    n = len(sorted_nums)
    mid = n // 2
    
    if n % 2 == 0:
        return (sorted_nums[mid - 1] + sorted_nums[mid]) / 2
    return sorted_nums[mid]

class StatisticalAnalyzer:
    '''A class for performing statistical analysis on datasets.'''
    
    def __init__(self, data: List[float]):
        self.data = data
        self.mean = None
        self.median = None
    
    def analyze(self) -> dict:
        '''Perform complete statistical analysis.'''
        self.mean = calculate_mean(self.data)
        self.median = calculate_median(self.data)
        
        return {
            'mean': self.mean,
            'median': self.median,
            'count': len(self.data)
        }
    
    def get_summary(self) -> str:
        '''Return a formatted summary of the analysis.'''
        if self.mean is None:
            self.analyze()
        
        return f"Mean: {self.mean:.2f}, Median: {self.median:.2f}"

def main():
    '''Main execution function.'''
    data = [1.5, 2.3, 3.7, 4.2, 5.1]
    analyzer = StatisticalAnalyzer(data)
    results = analyzer.analyze()
    print(analyzer.get_summary())

if __name__ == "__main__":
    main()
"""

# print(python_code)

# create the splitter
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=700,
    chunk_overlap=100
)

#  split the code \ most cases you will use split_documents() 
code_chunks = python_splitter.split_text(python_code)
# print(code_chunks)

# proper display  the above  code_chunks
from termcolor import colored, COLORS
from random import choice

def display_chunks(chunks):
    colors_list = list(COLORS.keys())[2:8]
    print(f"Total Number of Chunks: {len(chunks)}")
    for num, chunk in enumerate(chunks, 1):
        print(f"Chunk {num}: Length {len(chunk)} chars")
        print(colored(text=chunk, color=choice(colors_list)), end="\n\n")

# display_chunks(code_chunks)

#get_separators_for_language() is a utility method that simply shows you what separators LangChain uses internally for a given language.
print(python_splitter.get_separators_for_language(Language.PYTHON))
print(python_splitter.get_separators_for_language(Language.MARKDOWN))

# ----------------------------------------------------------------------------------

# STEP 2 - JSON data
JSON_DATA = {
    "company": "AI Research Corp",
    "departments": [
        {
            "name": "Machine Learning",
            "team_size": 25,
            "projects": [
                {
                    "id": "ML001",
                    "title": "Computer Vision System",
                    "description": "Developing advanced image recognition using CNNs",
                    "status": "active",
                    "team_members": ["Alice", "Bob", "Charlie"]
                },
                {
                    "id": "ML002",
                    "title": "NLP Platform",
                    "description": "Building transformer-based language models",
                    "status": "active",
                    "team_members": ["David", "Eve"]
                }
            ]
        },
        {
            "name": "Data Engineering",
            "team_size": 15,
            "projects": [
                {
                    "id": "DE001",
                    "title": "Data Pipeline",
                    "description": "ETL pipeline for real-time data processing",
                    "status": "active"
                }
            ]
        }
    ],
    "technologies": {
        "frameworks": ["TensorFlow", "PyTorch", "scikit-learn"],
        "languages": ["Python", "R", "Julia"],
        "cloud": ["AWS", "Google Cloud", "Azure"]
    },
    "metadata": {
        "founded": 2020,
        "headquarters": "San Francisco",
        "employees": 150
    }
}

# RecursiveJsonSplitter - Splits JSON data into smaller, structured chunks while preserving hierarchy.
from langchain_text_splitters import RecursiveJsonSplitter

# create the json splitter
json_splitter = RecursiveJsonSplitter(
    max_chunk_size=200    #default is 2000
    #  min_chunk_size: int | None = None
)

# return dictionaries
chunks_dict = json_splitter.split_json(json_data=JSON_DATA)
# print(chunks_dict)

# return json text in to string
chunks = json_splitter.split_text(JSON_DATA)
# print(chunks)

#dispay in proper way
# display_chunks(chunks)

# -----------------------------------------------------------------------
# STEP 3 - MARKDOWN_TEXT data
MARKDOWN_TEXT = """# Artificial Intelligence Overview

Artificial intelligence is transforming technology and shaping the future of computing.

## Machine Learning

Machine learning is a subset of AI that focuses on pattern recognition.

### Supervised Learning

Supervised learning algorithms learn from labeled training data.
They make predictions based on input-output pairs.

Common algorithms include:
- Linear regression
- Decision trees
- Support vector machines

### Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data.
It's useful for clustering and dimensionality reduction.

Common techniques:
- K-means clustering
- Principal component analysis
- Hierarchical clustering

## Deep Learning

Deep learning uses neural networks with multiple layers.

### Neural Networks

Neural networks are inspired by biological neurons.
They consist of interconnected nodes organized in layers.

### Convolutional Neural Networks

CNNs excel at image recognition tasks.
They use convolutional layers to detect features hierarchically.

## Applications

AI has applications across multiple domains:

### Healthcare

- Disease diagnosis
- Drug discovery
- Medical imaging analysis

### Finance

- Fraud detection
- Algorithmic trading
- Risk assessment
"""

from langchain_text_splitters import MarkdownHeaderTextSplitter
headers_to_split_on = [
    ("#", "Header_1"),
    ("##", "Header_2"),
    ("###", "Header_3")
]

# create the markdown splitter
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False
    
 # headers_to_split_on: list[tuple[str, str]],
# return_each_line: bool = False,
# strip_headers: bool = True,
# custom_header_patterns: dict[str, int] | None = None
)

# split the text
markdown_chunks = markdown_splitter.split_text(MARKDOWN_TEXT)
# print(markdown_chunks)
for doc in markdown_chunks:
    print(doc.page_content, end="\n\n")