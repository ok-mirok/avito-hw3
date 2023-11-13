class CountVectorizer:
    def __init__(self):
        self.vocabulary = {}
        self.feature_names = []

    def fit_transform(self, corpus: list[str]) -> list[list]:
        """
        Train the CountVectorizer on a corpus of text and
        transform the text into a feature matrix.
        """
        self.vocabulary = {}
        self.feature_names = []

        if not isinstance(corpus, list):
            raise TypeError('It is not a corpus list.')
        for text in corpus:
            if not isinstance(text, str):
                raise TypeError(f'Str object expected but get {type(text)}.')

        for text in corpus:
            for word in text.split():
                word = word.lower()
                if word not in self.vocabulary:
                    self.vocabulary[word] = len(self.vocabulary)
                    self.feature_names.append(word)

        X = []
        for text in corpus:
            words_counts = [0] * len(self.feature_names)
            for word in text.split():
                word = word.lower()
                words_counts[self.vocabulary[word]] += 1
            X.append(words_counts)

        return X

    def get_feature_names(self):
        """Get output feature names for transformation."""
        return self.feature_names


if __name__ == '__main__':
    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names())
    print(count_matrix)
