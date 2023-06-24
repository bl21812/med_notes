# NOTE: I might have to change the [D:] and [P:] speaker indicators to 
    # become special tokens for finetuning
# NOTE: this is not batched
def tokenize_qa(tokenizer, x1, x2=None, max_seq_length=2048, doc_stride=128):
    '''
    Tokenize question(s) and context(s) for a QA task
    :param x1: question or output
    :param x2: if passed, context
    :param max_seq_length: Max length of tokenized sequences
    :param doc_stride: Length of overlap between sequences from x

    :return: List of input ids (tokens)
        or list of lists of input ids if len(x) > max_seq_length
    '''

    if x2:
        tokenized = tokenizer(
            x1, 
            x2, 
            max_length=max_seq_length,
            truncation='only_second',
            return_overflowing_tokens=True,
            stride=doc_stride
        )
    else:
        tokenized = tokenizer(
            x1,
            max_length=max_seq_length/4,
            truncation=True,
            stride=0
        )

    # extract token ids
    # TODO: is this how it works for batches ?
    ids = [[token_seq for token_seq in token_seq_list] for token_seq_list in tokenized['input_ids']]

    return ids