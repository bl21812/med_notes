# NOTE: I might have to change the [D:] and [P:] speaker indicators to 
    # become special tokens for finetuning
def tokenize_qa(tokenizer, x1, x2=None, max_seq_length=2048, doc_stride=128):
    '''
    Tokenize question(s) and context(s) for a QA task
    :param x1: batch of [questions] or batch of [outputs]
    :param x2: if passed, batch of [contexts]
    :param max_seq_length: Max length of tokenized sequences
    :param doc_stride: Length of overlap between sequences from x
    :param pad_length: Pad output to this length
        only used when only x1 given (for output)

    :return: List of input ids (tokens)
        or list of lists of input ids if len(x) > max_seq_length
    '''

    if x2:
        tokenized = tokenizer(
            x1, 
            x2, 
            add_special_tokens=True,
            max_length=max_seq_length,
            truncation='only_second',
            return_overflowing_tokens=True,
            stride=doc_stride
        )
    else:
        tokenized = tokenizer(
            x1,
            add_special_tokens=True,
            max_length=max_seq_length,
            truncation=True,
            stride=0
        )

    # extract token ids
    ids = tokenized['input_ids']

    return ids


# TODO: Implement adding SEP tokens (just replace the D: and P: with SEP except the first one?)
# TODO: Implement task separation (diff preprocessing depending on dataset)
def preprocess_text(row, columns, task, add_sep=False):
    '''
    Any preprocessing needed for our read-in text data
    Pulling relevant rows and ...
    e.g. removing newline characters
    e.g. removing speaker indications (D:, P:)

    :param add_sep: Whether to add SEP tokens to separate dialogue or not
    '''
    ret = {}
    for col in columns:
        text = row[col]
        text = repr(text).replace('\\n', ' ')
        text = repr(text).replace('\\r', ' ')
        text = repr(text).replace('\\', '')
        # text = text.replace('D:', '')
        # text = text.replace('P:', '')
        if 'summary' in col:
            col = 'summary'
        ret[col] = text
    return ret

# TRY: 
    # describing soap notes (ask for patient's description of their condition, results of physical exams, doctor's diagnosis, and the doctor's plan) instead of just asking for soap
    # Separately asking for each (as I wrote in prev point)