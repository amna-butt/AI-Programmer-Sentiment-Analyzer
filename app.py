!pip install -q transformers torch gradio
%pip install -q transformers torch
from transformers import pipeline

model_id = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model=model_id
)

print("AI Programmer Sentiment Analyzer is ready!")\
text = "You are an excellent programmer!"

result = sentiment_analyzer(text)[0]

print("Input:", text)
print("Sentiment:", result["label"])
print("Confidence:", round(result["score"] * 100, 2), "%")

if result["label"] == "POSITIVE":
    print("Message: You are a great programmer! Keep it up!")
else:
    print("Message: You are still learning. Never give up!")
  import gradio as gr

def analyze_programmer_text(text):

    if not text or not text.strip():
        return "Please enter some text!", ""

    result = sentiment_analyzer(text)[0]

    label = result["label"]
    confidence = result["score"] * 100

    if label == "POSITIVE":
        message = "🌟 Positive: You are a great programmer! Keep it up!"
    else:
        message = "💻 Negative: You are still learning. Keep practicing!"

    details = (
        f"Sentiment: {label}\n"
        f"Confidence: {confidence:.2f}%"
    )

    return message, details


app = gr.Interface(
    fn=analyze_programmer_text,
    inputs=gr.Textbox(
        lines=3,
        label="Enter Your Text",
        placeholder="Example: I am learning Python and improving my coding skills."
    ),
    outputs=[
        gr.Textbox(label="Your Result"),
        gr.Textbox(label="Analysis Details")
    ],
    title="🤖 AI Programmer Sentiment Analyzer",
    description="Enter English text to find out whether its sentiment is positive or negative.",
    submit_btn="Analyze Sentiment"
)

app.launch(share=True)
