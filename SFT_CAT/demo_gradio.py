import requests, json
from conversation import conv_llama_2,conv_templates, SeparatorStyle
import gradio as gr
from model.utils import disable_torch_init
from mm_utils import tokenizer_image_token
from model.constants import IMAGE_TOKEN_INDEX,QUESTION_TOKEN_INDEX
from extract_video_feat import Image_feat_extract,audio_feat_extract
import torch
from model.builder import load_pretrained_model
import argparse
import os

model = 'llama2:latest' #You can replace the model name if needed
context = [] 

import gradio as gr

#Call Ollama API
# def generate(prompt, context, top_k, top_p, temp):
#     r = requests.post('http://localhost:11434/api/generate',
#                      json={
#                          'model': model,
#                          'prompt': prompt,
#                          'context': context,
#                          'options':{
#                              'top_k': top_k,
#                              'temperature':top_p,
#                              'top_p': temp
#                          }
#                      },
#                      stream=False)
#     r.raise_for_status()

 
#     response = ""  

#     for line in r.iter_lines():
#         body = json.loads(line)
#         response_part = body.get('response', '')
#         print(response_part)
#         if 'error' in body:
#             raise Exception(body['error'])

#         response += response_part

#         if body.get('done', False):
#             context = body.get('context', [])
#             return response, context

# ========================================
#             Model Initialization
# ========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--clip_path", type=str, default="checkpoints/clip/ViT-L-14.pt")
    parser.add_argument("--model_base", type=str, default="/home/qilang/PythonProjects/Video-LLaMA-main/llama-2-7b-chat-hf/")
    parser.add_argument("--pretrain_CA", type=str, default="/home/qilang/PythonProjects/AVLLM/MLLM_Cat/ckpt-AVQA-adapter/CA/CA.bin")
    parser.add_argument("--pretrain_mm_mlp_adapter_v", type=str, default="/home/qilang/PythonProjects/AVLLM/MLLM_Cat/ckpt-AVQA-adapter/mm_projector_v/mm_projector_v.bin")
    parser.add_argument("--pretrain_mm_mlp_adapter_a", type=str, default="/home/qilang/PythonProjects/AVLLM/MLLM_Cat/ckpt-AVQA-adapter/mm_projector_a/mm_projector_a.bin")
    parser.add_argument("--pretrain_attn_a2v", type=str, default="/home/qilang/PythonProjects/AVLLM/MLLM_Cat/ckpt-AVQA-adapter/attn_a2v/attn_a2v.bin")


    args = parser.parse_args()
    return args

args = parse_args()
disable_torch_init()
tokenizer, model, context_len = load_pretrained_model(args)
model = model.cuda()
model = model.to(torch.bfloat16)

def upload_imgorvideo(gr_video):

    # images = ImageClIP_feat_extract(video_path=gr_video, height=224, width=224, n_frms=60, sampling="uniform").unsqueeze(0).to(torch.float16)
    # audio = torch.randn([60,128]).unsqueeze(0).to(torch.float16)
    images = Image_feat_extract(video_path=gr_video, height=224, width=224, n_frms=60, sampling="uniform").unsqueeze(0).to(torch.bfloat16)
    audio = audio_feat_extract(video_path=gr_video).unsqueeze(0).to(torch.bfloat16)
    images=images.cuda()
    audios=audio.cuda()
    feat_list = []
    feat_list.append(images)
    feat_list.append(audios)

    return feat_list

# def upload_text(feat_list,query,top_k, top_p, temp):
def upload_text(feat_list,query):
    if ':' in query:
        q_index = query.find(':')+2
        temp = list(query)
        temp.insert(q_index,'<Q>')
        query = ''.join(temp)
        query = query+"<Q>\n<video>"
    else:
        query = '<Q>'+query+"<Q>\n<video>"
    conv = conv_templates["llama_2"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, QUESTION_TOKEN_INDEX, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]

    images = feat_list[0]
    audios = feat_list[1]
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images,
            audios=audios,
            # do_sample=True,
            # temperature=temp,
            do_sample=False,
            # top_k = top_k,
            # top_p = top_p,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True)
    
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()

    chat_history = []
    chat_history.append((query, outputs))
    return chat_history, chat_history



def chat(input, chat_history, top_k, top_p, temp):

    chat_history = chat_history or []

    global context
    # output, context = generate(input, context, top_k, top_p, temp)
    output = 'ssss'

    chat_history.append((input, output))

    return chat_history, chat_history
  #the first history in return history, history is meant to update the 
  #chatbot widget, and the second history is meant to update the state 
  #(which is used to maintain conversation history across interactions)


#########################Gradio Code##########################
block = gr.Blocks()


with block:

    gr.Markdown("""<h1><center> Qilang Ye Assistant </center></h1>
    """)

    state = gr.State()
    # with gr.Column():
    with gr.Row():
        with gr.Column():
            video = gr.Video()
            feat_list = gr.State()
            upload_button = gr.Button(value="Upload video")
            clear = gr.Button("Restart")
            with gr.Accordion("You can ask me"):
                gr.Markdown("What is the main sound in the video?")
                gr.Markdown("What are the people doing in the video?")
                gr.Markdown("Please describe the video to help me answer: question")

        # num_beams = gr.Slider(
        #         minimum=1,
        #         maximum=10,
        #         value=1,
        #         step=1,
        #         interactive=True,
        #         label="beam search numbers)",
        #     )
            

        # audio = gr.Checkbox(interactive=True, value=False, label="Audio")
        # top_k = gr.Slider(0.0,100.0, label="top_k", value=40, info="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)")
        # top_p = gr.Slider(0.0,1.0, label="top_p", value=0.9, info=" Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)")
        
        with gr.Row():
            with gr.Column():
                chat_state = gr.State()
                chatbot = gr.Chatbot(label='CAT')
                # with gr.Column():
                message = gr.Textbox(placeholder="Type here")
                submit = gr.Button(value="Send")
                        # text_input = gr.Textbox(label='User', placeholder='Upload your image/video first, or directly click the examples at the bottom of the page.', interactive=False)
        # with gr.Row():
        #     top_k = gr.Slider(0.0,100.0, label="top_k", value=40)
        #     top_p = gr.Slider(0.0,1.0, label="top_p", value=0.9)
        #     temp = gr.Slider(
        #         minimum=0.1,
        #         maximum=2.0,
        #         value=1.0,
        #         step=0.1,
        #         interactive=True,
        #         label="Temperature",
        #     )
    
    upload_button.click(upload_imgorvideo, inputs=[video], outputs=[feat_list])
    # submit.click(upload_text, inputs=[feat_list, message,top_k,top_p,temp], outputs=[chatbot, state])
    submit.click(upload_text, inputs=[feat_list, message], outputs=[chatbot, state])


block.launch(debug=True)