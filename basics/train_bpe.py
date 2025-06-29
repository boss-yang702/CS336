import regex as re
from typing import Iterable
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from collections import Counter
import concurrent.futures

from basics.io import GPT2_PRETOKENIZER_PATTERN

def _find_pretokens(text: str):
    """
    Find the pretokens in the text.
    """
    logging.info(f"Pre-tokenizing the text of length {len(text)}")
    return Counter(re.findall(GPT2_PRETOKENIZER_PATTERN, text))

def _read_text_file(input_path: str, num_worker: int, special_tokens: Iterable[str]):
    """
    Read the text file at the given path.
    Return the text as pretoken frequency table.
    """

    # Read the input text file
    with open(input_path, "r") as file:
        text = file.read()

    # Remove special tokens from the text
    for token in special_tokens:
        text = text.replace(token, "")
    
    logging.info("Initializing pretoken frequency table")
    if num_worker == 1:
        pretokens = _find_pretokens(text)
    else:
        chunk_size = len(text) // num_worker
        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_worker) as executor:
            pretokens = executor.map(_find_pretokens, text_chunks)
        pretokens = sum(pretokens, Counter())
    gen_tuple_of_bytes = lambda pretoken: tuple([bytes([b]) for b in pretoken.encode("utf-8")])
    pretoken_freq = {}
    for pretoken, freq in pretokens.items():
        pretoken_freq[gen_tuple_of_bytes(pretoken)] = freq
    
    return pretoken_freq


def _update_byte_tuple(byte_tuple: Iterable[bytes], merge_loc: int):
    """
    Merge the byte tuple at the merge location.
    """
    assert len(byte_tuple) > 1, "Cannot merge a byte tuple with length less than 2."
    prefix = byte_tuple[:merge_loc]
    tomerge = byte_tuple[merge_loc:merge_loc+2]
    suffix = byte_tuple[merge_loc+2:]
    new_byte_tuple = prefix + (b"".join(tomerge),) + suffix
    return new_byte_tuple, prefix, suffix


def train_bpe(input_path: str, vocab_size: int, special_tokens: Iterable[str],
              progress_bar: bool = False, num_workers: int = 1):
    """
    Train a byte pair encoding tokenizer on the input text file.

    Args:
        input_path: Path to the input text file.
        vocab_size: Size of the vocabulary.
        special_tokens: List of special tokens to add to the vocabulary.

    Returns:
        Tuple of the learned vocab and merges.
    """
    # Initialize the vocab with 256 bytes and sepcial tokens
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256+i] = token.encode("utf-8")
    
    pretoken_freq = _read_text_file(input_path, num_workers, special_tokens)

    logging.info("Initializing byte pair frequency table")
    pair_freq = Counter()
    for pretoken_tuple, freq in tqdm(pretoken_freq.items(), disable=not progress_bar):
        for i in range(len(pretoken_tuple) - 1):
            pair = pretoken_tuple[i:i+2]
            if pair not in pair_freq:
                pair_freq[pair] = 0
            pair_freq[pair] += freq

    logging.info("Performing BPE algorithm")
    pre_merge_vocab_size = len(vocab)
    pbar = tqdm(total=vocab_size-pre_merge_vocab_size) if progress_bar else None
    merges = []
    while len(vocab) < vocab_size:
        # Find the most frequent pair
        most_freq_pair = max(pair_freq, key=lambda k: (pair_freq[k], k))

        # Add the pair to the merges list
        merges.append(most_freq_pair)
        
        # Update the vocab
        new_id = max(vocab.keys()) + 1
        vocab[new_id] = b"".join(most_freq_pair)

        # Update the pre-token frequency table and pair frequency table
        new_pretoken_freq = {}
        for pretoken_tuple, freq in pretoken_freq.items():
            i=0
            while i < len(pretoken_tuple):
                pair = pretoken_tuple[i:i+2]
                if pair == most_freq_pair:
                    pretoken_tuple, prefix, suffix = _update_byte_tuple(pretoken_tuple, i)

                    # Update the pair frequency table
                    if prefix:
                        add_pair = (prefix[-1], vocab[new_id])
                        pair_freq[add_pair] = pair_freq.get(add_pair, 0) + freq
                        del_pair = (prefix[-1], most_freq_pair[0])
                        pair_freq[del_pair] -= freq
                    if suffix:
                        add_pair = (vocab[new_id], suffix[0])
                        pair_freq[add_pair] = pair_freq.get(add_pair, 0) + freq
                        del_pair = (most_freq_pair[1], suffix[0])
                        pair_freq[del_pair] -= freq
                    pair_freq[most_freq_pair] -= freq
                i+=1
            # Update the pre-token frequency table
            new_pretoken_freq[pretoken_tuple] = freq
        pretoken_freq = new_pretoken_freq
        pbar.update(len(vocab) - pre_merge_vocab_size - pbar.n) if progress_bar else None
    pbar.close() if progress_bar else None

    return vocab, merges

from basics.io import save_voacb_and_merge
if __name__=="__main__":
    datasetname='tinystory'
    input_path="data/tinystory/TinyStoriesV2-GPT4-train.txt"
    vocab_size=10000
    special_tokens=['<|endoftext|>']
    vocab,merges=train_bpe(input_path,vocab_size,special_tokens)
    vocab_path=f"data/{datasetname}/vocab.json"
    merge_path=f"data/{datasetname}/merge.txt"
    save_voacb_and_merge(vocab,merges,vocab_path=vocab_path,merges_path=merge_path)

