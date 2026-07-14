def remove_stopwords(tokens, stopwords):
    filtered = []

    for token in tokens:
        if token not in stopwords:
            filtered.append(token)

    return filtered
            