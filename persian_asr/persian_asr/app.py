import gradio as gr

from .pipeline import asr_with_sentiment
from .utils import log_action, setup_logger

setup_logger()

class WebApp:
    """
    Main Gradio web interface for Persian ASR with Sentiment Analysis.
    """

    def __init__(self):
        log_action("WebApp initialized")

    def interface(self):
        """
        Creates Gradio interface for ASR + Sentiment Analysis.
        """
        log_action("Building Gradio interface")

        with gr.Blocks() as demo:
            gr.Markdown("# 🎙️ Persian ASR Demo with Sentiment Analysis")
            gr.Markdown("Upload or record Persian speech, get transcript and sentiment:")

            audio_input = gr.Audio(type="filepath", label="🎤 Record or upload your voice")
            text_output = gr.Textbox(label="📝 Transcribed Text")
            sentiment_output = gr.Textbox(label="💬 Sentiment Analysis")

            gr.Button("Convert").click(
                fn=asr_with_sentiment,
                inputs=[audio_input],
                outputs=[text_output, sentiment_output]
            )

        return demo

    def launch(self):
        """
        Launches the Gradio app with a public shareable link.
        """
        log_action("Launching Gradio app")
        demo = self.interface()
        demo.launch()
