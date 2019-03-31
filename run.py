from dictionary import Dictionary
from word_markov_model import WordMarkovModel
import language_prediction

dictionary = Dictionary('dictionaries/english_alpha.txt')
dictionary.filter_words(lambda word: len(word) > 2)
dictionary.map_words(lambda word: word.lower())
word_generator = WordMarkovModel(dictionary, 5)
word_generator.train()

polish_dictionary = Dictionary('dictionaries/polish_alpha.txt')
polish_dictionary.filter_words(lambda word: len(word) > 2)
polish_dictionary.map_words(lambda word: word.lower())
polish_word_generator = WordMarkovModel(polish_dictionary, 5)
polish_word_generator.train()

for i in range(1000):
	print('Generating word: ' + str(i+1))
	print()

	word = word_generator.generate_word()
	probability = word_generator.find_word_probability(word)
	print('English word generated, no constraints: ')
	print(word, probability[0])
	print()

	word = word_generator.generate_word(min_length=5, max_length=20)
	probability = word_generator.find_word_probability(word)
	print('English word generated, length [5, 20]: ')
	print(word, probability[0])
	print()
	
	word = word_generator.generate_new_word()
	probability = word_generator.find_word_probability(word)
	print('English word generated, new word: ')
	print(word, probability[0])
	print()

	word = word_generator.generate_new_word(min_length=5, max_length=20)
	probability = word_generator.find_word_probability(word)
	print('English word generated, new word length [5, 20]: ')
	print(word, probability[0])
	print()

	word = polish_word_generator.generate_word()
	probability = polish_word_generator.find_word_probability(word)
	print('Polish word generated, no constraints: ')
	print(word, probability[0])
	print()
	
	word = polish_word_generator.generate_word(min_length=5, max_length=20)
	probability = polish_word_generator.find_word_probability(word)
	print('Polish word generated, length [5, 20]: ')
	print(word, probability[0])
	print()

	word = polish_word_generator.generate_new_word()
	probability = polish_word_generator.find_word_probability(word)
	print('Polish word generated, new word: ')
	print(word, probability[0])
	print()
	
	word = polish_word_generator.generate_new_word(min_length=5, max_length=20)
	probability = polish_word_generator.find_word_probability(word)
	print('Polish word generated, new word length [5, 20]: ')
	print(word, probability[0])

	print('--------------')

print(word_generator.find_word_probability('qv'))

def test_word(word, models):
	print(word)
	print(language_prediction.language_probabilities(word, models))
	print(language_prediction.language_membership(word, models))

test_word('test', [word_generator, polish_word_generator])
test_word('zażółć', [word_generator, polish_word_generator])
test_word('autobus', [word_generator, polish_word_generator])
test_word('bus', [word_generator, polish_word_generator])
test_word('dictionary', [word_generator, polish_word_generator])
test_word('dictionalization', [word_generator, polish_word_generator])
test_word('marginal', [word_generator, polish_word_generator])
test_word('marginalny', [word_generator, polish_word_generator])
test_word('marginalization', [word_generator, polish_word_generator])
test_word('marginalizacja', [word_generator, polish_word_generator])
test_word('mastereo', [word_generator, polish_word_generator])
