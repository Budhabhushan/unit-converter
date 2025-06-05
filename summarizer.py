
# Import the Hugging Face transformers pipeline utility
from transformers import pipeline

# Initialize the summarization pipeline using a pre-trained model from Hugging Face
# In this case, we are using "facebook/bart-large-cnn", which is fine-tuned for summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Define a function to summarize input text
def summarize_text(text, max_length=130, min_length=30, do_sample=False):
    """
    Summarizes the input text using a transformer-based summarization model.

    Parameters:
    - text (str): The input text to .
    - max_length (int): The maximum length of the summary.
    - min_length (int): The minimum length of the summary.
    - do_sample (bool): Whether or not to use sampling; if False, greedy decoding is used.

    Returns:
    - str: The summarized text.
    """
    # Check if the input text is empty or consists only of whitespace
    if not text.strip():
        return "Please enter valid text."
       
    # Generate the summary using the pre-loaded summarizer pipeline
    # The summarizer returns a list of dictionaries, each with a 'summary_text' key
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=do_sample)

    # Return only the summarized text from the first result
    return summary[0]['summary_text']
