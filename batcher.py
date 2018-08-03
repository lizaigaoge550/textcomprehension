import data
class Batch():
    def __init__(self, examples, config, vocab):
        self.pad_id = vocab.word2id(data.PAD_TOKEN)
        self._init_article_seq(examples, config.max_article_len)
        self._init_question_seq(examples, config.max_question_len)

    def _init_article_seq(self, examples, max_len):
        for example in examples:




