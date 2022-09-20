class CountVectorizer:
    """ Converts text to vector """

    def __init__(self):
        self._vocabulary = []

    @staticmethod
    def create_vocabulary(corp: [str]) -> [str]:
        vocabulary = []
        for text in corp:
            for word in text.lower().split(' '):
                if word not in vocabulary:
                    vocabulary.append(word)
        return vocabulary

    def fit_transform(self, corp: [str]) -> []:
        self._vocabulary = self.create_vocabulary(corp)
        matrix = []
        for text in corp:
            word_nums = dict.fromkeys(self._vocabulary, 0)
            for word in text.lower().split(' '):
                word_nums[word] += 1
            matrix.append(list(word_nums.values()))
        return matrix

    def get_feature_names(self):
        return self._vocabulary


if __name__ == '__main__':
    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)
    answer_feature_names = ['crock', 'pot', 'pasta', 'never', 'boil', 'again', 'pomodoro',
                            'fresh', 'ingredients', 'parmesan', 'to', 'taste']
    answer_count_matrix = [[1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    assert vectorizer.get_feature_names() == answer_feature_names and \
           count_matrix == answer_count_matrix
