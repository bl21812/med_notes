# NOTE: I might have to change the [D:] and [P:] speaker indicators to 
    # become special tokens for finetuning
# NOTE: Ignoring overflow rn
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
        IM RETURNING DICT WITH input_ids, attention_mask, and labels keys now
            where labels = input_ids
    '''

    if x2:
        tokenized = tokenizer(
            x1, 
            x2, 
            add_special_tokens=True,
            max_length=max_seq_length,
            truncation=True,
            # truncation='only_second',
            # return_overflowing_tokens=True,
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

    keys = ['input_ids', 'attention_mask']
    result = {key: tokenized[key] for key in keys}

    if tokenized['input_ids'][-1] != tokenizer.eos_token_id:
        result['input_ids'].append(tokenizer.eos_token_id)
        result['attention_mask'].append(1)
    
    result['labels'] = result['input_ids'].copy()

    return result


# IGNORING OVERFLOW FOR NOW
def tokenize_dialogue_summary(tokenizer, inputs, outputs, max_seq_length=2048, doc_stride=128):
    '''
    Let input_ids be a prompt template with dialogue and blank summary section (inputs)
        and labels be the above but with the summary section filled (outputs)
    '''
    input_tokenized = tokenizer(
        inputs, 
        add_special_tokens=True,
        max_length=max_seq_length,
        truncation=True,
        stride=doc_stride
    )

    output_tokenized = tokenizer(
        outputs, 
        add_special_tokens=True,
        max_length=max_seq_length,
        truncation=True,
        stride=doc_stride
    )

    keys = ['input_ids', 'attention_mask']
    result = {key: input_tokenized[key] for key in keys}
    result['labels'] = output_tokenized['input_ids']

    if input_tokenized['input_ids'][-1] != tokenizer.eos_token_id:
        result['input_ids'].append(tokenizer.eos_token_id)
        result['attention_mask'].append(1)
    if output_tokenized['input_ids'][-1] != tokenizer.eos_token_id:
        result['labels'].append(tokenizer.eos_token_id)

    return result


# TODO: Implement adding SEP tokens (just replace the D: and P: with SEP except the first one?)
# TODO: Implement task separation (diff preprocessing depending on dataset)
def preprocess_text(row, columns, task=None, add_sep=False):
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
        text = repr(text).replace('\\n \\n', '\\n ')
        text = repr(text).replace('\\r', ' ')
        # text = repr(text).replace('\\', '')
        text = text.replace('D:', '#Doctor#:')
        text = text.replace('P:', '#Patient#:')
        if 'summary' in col:
            col = 'summary'
        ret[col] = text
    return ret

# TRY: 
    # describing soap notes (ask for patient's description of their condition, results of physical exams, doctor's diagnosis, and the doctor's plan) instead of just asking for soap
    # Separately asking for each (as I wrote in prev point)