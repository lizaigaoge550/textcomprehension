
PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'

class Vocab():
    def __init__(self, vocab_file, max_size):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # keeps track of total number of words in the Vocab

        #unk 和 pad是0,1
        for w in [UNKNOWN_TOKEN, PAD_TOKEN]:
            self._word_to_id[w.lower()] = self._count
            self._id_to_word[self._count] = w.lower()
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r', encoding='utf-8') as vocab_f:
            for line in vocab_f:
                pieces = line.rsplit(' ', 1)
                if len(pieces) != 2:
                    print('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
                    continue
                w = pieces[0].lower()
                if w in [ UNKNOWN_TOKEN, PAD_TOKEN]:
                    raise Exception(
                        '[UNK], [PAD] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (
                    max_size, self._count))
                    break
        print("Finished constructing vocabulary of %i total words. Last word added: %s" % (
        self._count, self._id_to_word[self._count - 1]))

    def word2id(self, word):
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN.lower()]
        return self._word_to_id[word.lower()]

    def id2word(self, word_id):
        if word_id not in self._id_to_word:
            return self._id_to_word[0]
        return self._id_to_word[word_id]

    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count




