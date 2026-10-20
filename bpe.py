from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=["input.txt"],
    vocab_size=1000,
    min_frequency=2,
    special_tokens=["<pad>", "<unk>", "<end>"],
)

tokenizer._tokenizer.save("shakespeare_tokenizer.json")