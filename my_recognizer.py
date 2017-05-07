import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    for word_id in range(0, len(test_set.get_all_Xlengths())):
        cur_seq = test_set.get_item_sequences(word_id)
        cur_x, cur_len = test_set.get_item_Xlengths(word_id)
        best_score = -float('INF')
        best_guess = None
        prob_dict = {}
        for word, model in models.items():
            score = -float('INF')
            try:
                score = model.score(cur_x, cur_len)
            except:
                pass
            prob_dict[word] = score
            if score >= best_score:
                best_score = score
                best_guess = word
        probabilities.append(prob_dict)
        guesses.append(best_guess)
    return (probabilities, guesses)
