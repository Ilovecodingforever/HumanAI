import torch
import transformers

from transformers import LlamaForQuestionAnswering, LlamaTokenizer



def load_model(model_dir="/zfsauton2/home/mingzhul/Prompt-Tuning-LLM/llama/llama-2-7b-hf"):
    model = LlamaForQuestionAnswering.from_pretrained(model_dir)

    return model


def load_tokenizer(model_dir="/zfsauton2/home/mingzhul/Prompt-Tuning-LLM/llama/llama-2-7b-hf"):

    tokenizer = LlamaTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer



def run_model(input):

    model = load_model('meta-llama/Llama-2-7b-chat-hf')
    tokenizer = load_tokenizer('meta-llama/Llama-2-7b-chat-hf')

    pipeline = transformers.pipeline("text-generation",
                                    model=model,
                                    tokenizer=tokenizer,
                                    torch_dtype=torch.float16,
                                    device_map="auto",)

    sequences = pipeline(input,
                        do_sample=True,
                        top_k=10,
                        num_return_sequences=1,
                        eos_token_id=tokenizer.eos_token_id,
                        max_length=400,)

    return sequences[0]['generated_text']




if __name__ == '__main__':
    # run this in terminal: huggingface-cli login

    model = load_model('meta-llama/Llama-2-7b-chat-hf')
    tokenizer = load_tokenizer('meta-llama/Llama-2-7b-chat-hf')


    # model = load_model()
    # model.push_to_hub('ml233/humanai-llama')




