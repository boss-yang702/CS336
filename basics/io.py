import json
from typing import Dict, List, Tuple
import os
from functools import lru_cache
import torch

GPT2_PRETOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

@lru_cache()
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d

def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike
    ):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return vocab, merges

def save_voacb_and_merge(vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]],
                            vocab_path: str, merges_path: str):
    byte_to_unicode = gpt2_bytes_to_unicode()

    # Reverse the mapping from unicode characters to bytes
    unicode_to_byte = {v: k for k, v in byte_to_unicode.items()}
    
    # Convert the byte tokens in the vocab back to string tokens using the unicode mapping
    reversed_vocab = {''.join([byte_to_unicode[b] for b in bytes_token]):k
                      for k, bytes_token in vocab.items()}

    # Convert the byte sequences in merges back to string tokens
    reversed_merges = [' '.join([''.join([byte_to_unicode[b] for b in merge[0]]),
                                 ''.join([byte_to_unicode[b] for b in merge[1]])])
                       for merge in merges]

    # Save the vocab dictionary as a JSON file
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(reversed_vocab, f, ensure_ascii=False)
    
    # Save the merges list to a file
    with open(merges_path, 'w', encoding='utf-8') as f:
        for merge in reversed_merges:
            f.write(merge + '\n')

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }, out)

def load_checkpoint(src: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration

if __name__ == "__main__":
    vocab_path = 'data/out/tinystories_vocab.json'
    merges_path = 'data/out/tinystories_merges.txt'
    vocab, merges = get_tokenizer_from_vocab_merges_path(vocab_path, merges_path)
    output_vocab_path = 'data/out/test_vocab.json'
    output_merge_path = 'data/out/test_merges.txt'
    save_voacb_and_merge(vocab, merges, output_vocab_path, output_merge_path)
    load_vocab, load_merges = get_tokenizer_from_vocab_merges_path(output_vocab_path, output_merge_path)
    assert merges == load_merges
    assert vocab == load_vocab