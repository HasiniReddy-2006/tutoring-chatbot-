import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_name = "gpt2"


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
def ask_tutor_by_subject(subject, question):
    prompt = f"""You are a helpful and accurate {subject} teacher.
Answer the following question in 2-3 clear and factual sentences.
Avoid repeating or hallucinating information.

Question: {question}
Answer:"""

    result = pipe(prompt.strip(),
                  max_new_tokens=100,
                  do_sample=False,  # deterministic, more accurate
                  temperature=0.3,
                  pad_token_id=tokenizer.eos_token_id)[0]['generated_text']

    answer = result.replace(prompt.strip(), "").strip()
    if "Question:" in answer:
        answer = answer.split("Question:")[0].strip()
    return answer
import gradio as gr

def chat(subject, user_input, history=[]):
    reply = ask_tutor_by_subject(subject, user_input)
    history.append((user_input, reply))
    return history, history

with gr.Blocks() as demo:
    gr.Markdown("# 📚 AI School Subject Tutor")

    subject = gr.Dropdown(
        choices=["Maths", "Science", "English", "Social"],
        label="Select a Subject"
    )

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask a question:")
    clear = gr.Button("Clear")

    state = gr.State([])

    msg.submit(chat, [subject, msg, state], [chatbot, state])
    clear.click(lambda: ([], []), None, [chatbot, state])

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        prevent_thread_lock=True
    )
