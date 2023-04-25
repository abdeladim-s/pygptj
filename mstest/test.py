import _pygptj as pp

model_path = "/home/su/Downloads/ggml-gpt4all-j.bin"

def new_text_callback(text: str):
    print(text , end='')

def main():
    # loading the model
    model = pp.gptj_model()
    vocab = pp.gpt_vocab()
    pp.gptj_model_load(model_path, model, vocab)
    # print(vocab.token_to_id)
    # gnerate
    params = pp.gptj_gpt_params()
    params.prompt = "Once upon a time "
    params.n_predict = 55
    # tokens = pp.gpt_tokenize(vocab, params.prompt)
    # # print(tokens)
    pp.gptj_generate(params, model, vocab, new_text_callback)

    pp.gptj_free(model)


if __name__ == '__main__':
    main()
