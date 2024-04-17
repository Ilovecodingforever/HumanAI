







def load_model(model_dir="/zfsauton2/home/mingzhul/Prompt-Tuning-LLM/llama/llama-2-7b-hf"):
    model = LlamaForQuestionAnswering.from_pretrained(model_dir,
                                                           device_map='auto', torch_dtype="auto",)

    return model


def load_tokenizer(model_dir="/zfsauton2/home/mingzhul/Prompt-Tuning-LLM/llama/llama-2-7b-hf"):

    tokenizer = LlamaTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer



def run_model(input):   
    
    model = load_model()
    tokenizer = load_tokenizer()

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


