import streamlit as st
import requests
import os
import tempfile
import json
from docx import Document
from sseclient import SSEClient
from PIL import Image
import io
from generate_pic import send_async_generation_request, send_generation_request
from openai import OpenAI

# Load settings from environment variables
DIFY_API_KEY = os.getenv('DIFY_API_KEY')
DIFY_API_URL = os.getenv('DIFY_API_URL')
DIFY_API_TIMEOUT = int(os.getenv('DIFY_API_TIMEOUT', 300))
STABILITY_KEY = os.getenv('STABILITY_KEY')
clientLLM = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def send_message_to_dify(user_message):
    headers = {
        "Authorization": f"Bearer {DIFY_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "inputs": {},
        "query": user_message,
        "response_mode": "streaming",
        "conversation_id": "",
        "user": "user123"
    }
    
    try:
        response = requests.post(DIFY_API_URL, headers=headers, json=data, stream=True, timeout=DIFY_API_TIMEOUT)
        response.raise_for_status()
        client = SSEClient(response)
        return client
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with Dify API: {str(e)}")
        return None

def main():
    st.title("Dify Chat Application")

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Text input field
    user_input = st.text_input("Write your introduction, and your dream!!", 
                               #value="My name is :\nMy dream is:"
                               )

    # File upload (document and image)
    uploaded_doc = st.file_uploader("Upload your resume (optional)", type=['docx'])

    # Read document content
    file_content = ""
    if uploaded_doc is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
            temp_file.write(uploaded_doc.getvalue())
            doc = Document(temp_file.name)
            file_content = "\n".join([para.text for para in doc.paragraphs])
        os.unlink(temp_file.name)
        st.write("File content loaded successfully!")

    # Add image upload functionality
    uploaded_image = st.file_uploader("Upload an image (JPG or PNG)", type=['jpg', 'jpeg', 'png'], key="image_uploader")

    # Control send button enable/disable
    if st.button("Send", disabled=(not user_input and not file_content)):
        # Combine file content and user input
        user_message = f"{file_content}\n{user_input}".strip()
        
        with st.spinner('Getting response from Your career consultant team...'):
            sse_client = send_message_to_dify(user_message)
            if sse_client:
                response_container = st.empty()
                full_response = ""
                for event in sse_client.events():
                    if event.event == "message":
                        try:
                            data = json.loads(event.data)
                            message_chunk = data.get("answer", "")
                            full_response += message_chunk
                            response_container.markdown(f"**Bot:**\n{full_response}")
                        except json.JSONDecodeError:
                            st.error("Error decoding response from Dify")
                    elif event.event == "error":
                        st.error(f"Error from Dify: {event.data}")
                    elif event.event == "done":
                        break

                st.session_state.chat_history.append(("Bot", full_response))

                with st.spinner("Let's take a peek at your future..."):
                    press_release = clientLLM.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", 
                                 "content": '''
                                 You are a very talented journalist. You can write up the interview content as an attractive press release.
                                 '''},
                                {"role": "user", "content": f'''
                                 Please describe the situation when the future vision proposed in the following text becomes a reality, as if it were a press release such as a newspaper article.
                                 text: {full_response}
                                 Please output only the press release.'''
                                 }
                            ]
                        ).choices[0].message.content
                
                
                st.write(press_release)

                # If an image is uploaded, execute image generation
                if uploaded_image is not None:
                    with st.spinner("Acquiring your future vision..."):
                        # Summarize full_response to create a prompt for image generation
                        summary_prompt = clientLLM.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", 
                                "content": '''
                                You are an excellent AI painter. You have a very good ability to describe picturesque scenery from sentences.
                                '''},
                                {"role": "user", "content": f'''
                                Please summarize the following text as a prompt for image generation. 
                                Please try to be as specific as possible when generating images, such as by describing the people or the overall scene.
                                text: {press_release}
                                Please output onlythe prompt for image generation.'''
                                }
                            ]
                        ).choices[0].message.content

                        # Add 'Photorealistic quality' to the end of summary_prompt
                        summary_prompt += " Photorealistic quality."

                        #st.write(summary_prompt)
                        # Save image as a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                            temp_file.write(uploaded_image.getvalue())
                            temp_file_path = temp_file.name

                        # Send image generation request
                        host = "https://api.stability.ai/v2beta/stable-image/control/structure"
                        params = {
                            "control_strength": 0.7,
                            "image": temp_file_path,
                            "seed": 42,
                            "output_format": "png",
                            "prompt": summary_prompt,
                            "negative_prompt": ""
                        }
                        response = send_generation_request(host, params)

                        if response.status_code == 200:
                            # Display generated image
                            generated_image = Image.open(io.BytesIO(response.content))
                            resized_image = generated_image.resize((512, 512))
                            st.image(resized_image, 
                                     #caption="Generated image", 
                                     use_column_width=False)
                        else:
                            st.error("Failed to generate image")

                        # Delete temporary file
                        os.unlink(temp_file_path)



    # Display chat history (Bot responses only)
#    for role, message in st.session_state.chat_history:
#        if role == "Bot":
#            st.markdown(f"**Bot:**\n{message}")
#            st.divider()  # Add a divider between each response

if __name__ == '__main__':
    main()