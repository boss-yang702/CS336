import regex as re
from typing import Dict, Tuple, Iterable, List
from tqdm import tqdm
import numpy as np
import os
from basics.io import get_tokenizer_from_vocab_merges_path, GPT2_PRETOKENIZER_PATTERN

def get_pairs(ids: Iterable[int]) -> Iterable[Tuple[int, int]]:
    """ Return a set of pairs in int ids """
    pairs = set()
    for pair in zip(ids, ids[1:]):
        pairs.add(pair)
    return pairs

def update(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """ Update the ids by merging the pairs """
    new_ids = []
    i = 0
    while i < len(ids):
        curr_pair = tuple(ids[i:i+2])
        if curr_pair == pair:
            new_ids.append(new_id)
            i += 1
        else:
            new_ids.append(ids[i])
        i += 1
    return new_ids

def _fix_vocab(vocab_i_to_b: Dict[int, bytes], vocab_b_to_i: Dict[str, bytes]):
    """ Make sure all bytes are in the vocab """
    for i in range(256):
        byte = bytes([i])
        if byte not in vocab_b_to_i:
            vocab_b_to_i[byte] = len(vocab_b_to_i)
            vocab_i_to_b[len(vocab_i_to_b)] = byte
    return dict(int_to_byte=vocab_i_to_b, byte_to_int=vocab_b_to_i)

class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: Iterable[Tuple[bytes, bytes]], special_tokens: Iterable[str]=None):
        self.vocab = {}
        self.vocab['int_to_byte'] = vocab
        self.vocab['byte_to_int'] = {v: k for k, v in vocab.items()}
        self.vocab = _fix_vocab(self.vocab['int_to_byte'], self.vocab['byte_to_int'])

        # reorganzie merges into pair -> new token id dict
        self.merges = {}
        for a, b in merges:
            id_pair = (self.vocab['byte_to_int'][a], self.vocab['byte_to_int'][b])
            self.merges[id_pair] = self.vocab['byte_to_int'][a+b]
        
        # add special tokens as string to id mapping
        self.special_tokens = {}
        if special_tokens:
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            for token in special_tokens:
                token_byte = token.encode("utf-8")
                if token_byte not in self.vocab['byte_to_int']:
                    self.vocab['byte_to_int'][token_byte] = len(self.vocab['byte_to_int'])
                    self.vocab['int_to_byte'][len(self.vocab['int_to_byte'])] = token_byte
                    self.special_tokens[token] = len(self.vocab['int_to_byte'])
                else:
                    self.special_tokens[token] = self.vocab['byte_to_int'][token_byte]
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None, **kwargs):
        vocab, merges = get_tokenizer_from_vocab_merges_path(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    @property
    def vocab_size(self):
        return len(self.vocab['int_to_byte'])
    
    def _encode_chunk(self, text: str) -> List[int]:
        """
        Encode the text without special tokens.
        """
        if text in self.special_tokens:
            return [self.special_tokens[text]]
        else:
            text_chunks = re.findall(GPT2_PRETOKENIZER_PATTERN, text)
            result = []
            for chunk in text_chunks:
                text_bytes = chunk.encode("utf-8")
                ids = [self.vocab['byte_to_int'][bytes([b])] for b in chunk.encode("utf-8")]
                while len(ids)>=2:
                    pairs = get_pairs(ids)
                    high_priority_pair = min(pairs, key=lambda pair: self.merges.get(pair, float('inf')))
                    if high_priority_pair not in self.merges:
                        break
                    new_id = self.merges[high_priority_pair]
                    ids = update(ids, high_priority_pair, new_id)
                result.extend(ids)
            return result


    def encode(self, text: str, progress_bar: bool=False) -> List[int]:
        """
        Encode the text into a list of token ids.
        """
        if self.special_tokens:
            special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
            special_split_chunk = re.split(special_pattern, text)
        else:
            special_split_chunk = [text]
        ids = []
        for chunk in tqdm(special_split_chunk, disable=not progress_bar,
                          desc=f"Encoding {len(special_split_chunk)} documents"):
            ids += self._encode_chunk(chunk)
        return ids
    
    def encode_iterable(self, texts: Iterable[str]) -> Iterable[List[int]]:
        """
        Encode the texts into a list of token ids.
        """
        for text in texts:
            ids = self.encode(text)
            for id in ids:
                yield id

    def decode(self, ids: List[int]) -> str:
        """
        Decode the token ids into the original text.
        """
        text_bytes = b''.join([self.vocab['int_to_byte'][i] for i in ids])
        return text_bytes.decode("utf-8", errors="replace")
    
# --- 主执行脚本 ---
def main():
    """
    主函数，执行从 TXT 到 BIN 的转换。
    """
    # --- 配置 ---
    # 输入的文本文件路径
    input_txt_path = 'data/tinystory/TinyStoriesV2-GPT4-valid.txt'
    # 输出的二进制文件路径
    output_bin_path = 'data/tinystory/valid.bin' 
    # 词汇表和合并规则的路径（供 get_tokenizer_from_vocab_merges_path 使用）
    vocab_path = 'data/tinystory/vocab.json' # 请替换为您的路径
    merges_path = 'data/tinystory/merge.txt'  # 请替换为您的路径
    special_tokens = ['<|endoftext|>'] # 示例特殊token
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_bin_path), exist_ok=True)

    # 1. 加载分词器
    print("正在加载分词器...")
    # 注意：这里假设您的 from_files 方法可以正确处理路径
    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_path, 
        merges_filepath=merges_path, 
        special_tokens=special_tokens
    )
    print(f"分词器加载完毕，词汇表大小: {tokenizer.vocab_size}")

    # 2. 读取TXT文件
    print(f"正在读取输入文件: {input_txt_path}")
    with open(input_txt_path, 'r', encoding='utf-8') as f:
        text_data = f.read()
    print("文件读取完毕。")

    # 3. 编码文本为Token ID
    print("正在将文本编码为token...")
    token_ids = tokenizer.encode(text_data, progress_bar=True)
    print(f"编码完成，共生成 {len(token_ids)} 个token。")

    # 4. 将Token ID转换为NumPy数组并保存为BIN文件
    print(f"正在将token保存到: {output_bin_path}")
    # 使用 uint16 类型，与您的Dataset类保持一致，适用于词汇表大小 < 65536 的情况
    output_array = np.array(token_ids, dtype=np.uint16)
    
    # 使用 tofile 方法将数组内容直接写入二进制文件
    output_array.tofile(output_bin_path)

    print("转换完成！")

if __name__ == "__main__":
    main()