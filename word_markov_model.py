import math
from operator import itemgetter, mul
from random import randint
from functools import reduce
from dictionary import Dictionary

class WordMarkovModel:
	"""Word markov model is a nth order markov model of language words

	Takes dictionary of words and model order
	Trains the model (finds transition probabilities from ngrams/prechains to next characters/postchains)
	Generates words using trained transition probabilities
	Finds words/ngrams probabilities
	"""

	def __init__(self, dictionary, model_order=1):
		"""Create a new word markov model, takes dictionary of words and model order"""
		self.dictionary = dictionary
		self.ngrams_counter = dictionary.build_counter_of_ngrams(model_order + 1)
		self.model_order = model_order

	def train(self):
		"""Train the model, find the transition matrix between ngrams and following characters"""
		transitions = {}
		ngrams = list(self.ngrams_counter.items())
		ngrams.sort(key=itemgetter(0))

		for ngram, occurences in ngrams:
			prechain = WordMarkovModel.build_prechain_from_ngram(ngram)
			if prechain not in transitions:
				transitions[prechain] = {}
			postchain = WordMarkovModel.build_postchain_from_ngram(ngram)
			transitions[prechain][postchain] = occurences

		for prechain, transition_dict in transitions.items():
			occurences_sum = sum(transition_dict.values())
			transition_dict['##sum'] = occurences_sum

		self.transitions = transitions

	def generate_word(self, min_length=0, max_length=math.inf):
		"""Generate a word from trained transition matrix
		additionally, satisfy (optional) constraints on word length (min_length and max_length)
		note, depending on dictionary (the training data) given, satisfying the constraints may be impossible
		but that is rather unlikely to be the case
		for instance, end character might be the only transition possible from current ngram,
		while the word length is still below required minimum length, in such case end character is appended
		and shorter word is returned
		or a word migth be cut at max length, even if there is no suitable transition to end character at that point
		when no constraints are given, the word can have any length (from 0 to infinty)
		in such case, the function might not terminate, if there are loops in the markov model
		however, with some dependence on the training data, it is very unlikely that the function does not terminate
		"""
		word = ''
		next_character = ''

		while next_character != Dictionary.END_CHARACTER and len(word) <= max_length:
			if len(word) < min_length:
				next_character = self.generate_next_character(Dictionary.build_ngram_for(word, self.model_order), False)
			else:
				next_character = self.generate_next_character(Dictionary.build_ngram_for(word, self.model_order))
			word += next_character

		return word.replace(Dictionary.START_CHARACTER, '').replace(Dictionary.END_CHARACTER, '')

	def generate_new_word(self, min_length=0, max_length=math.inf):
		"""Generate new word, ie, a word that was not in dictionary used
		see generate_word() method for more information
		it might be impossible to generate a new word from the training data
		in case where model order is bigger than length of any word in the dictionary
		the model is overfitting then, and it will only be able to generate words
		that were already in the training data with their appropriate probabilities
		"""
		new_word = self.generate_word(min_length, max_length)
		while new_word in self.dictionary:
			new_word = self.generate_word(min_length, max_length)
		return new_word

	def find_word_probability(self, word, with_ending=True):
		"""Find probability of given word according to the trained model, ie, the transition matrix
		with_ending indicates if the probability of the word that ends immediately afterwards is to be calculated
		ie appending the end character to the word, 
		otherwise the probability of word starting with given string is calculated
		Return a pair, with first entry being the total probability of given word (product of transition probabilities)
		second entry being a list of transition probabilities into next characters
		"""
		word_ngrams = Dictionary.build_ngrams_of_word(word, self.model_order + 1, with_ending)
		ngrams_probabilities = list(map(self.find_ngram_probability, word_ngrams))
		total_probability = reduce(mul, ngrams_probabilities)
		return (total_probability, ngrams_probabilities)

	def find_ngram_probability(self, ngram):
		"""Find probability of given ngram, ie, the transition from prechain to postchain represented by the ngram"""
		prechain = WordMarkovModel.build_prechain_from_ngram(ngram)
		postchain = WordMarkovModel.build_postchain_from_ngram(ngram)

		if prechain not in self.transitions:
			return 0
		transition = self.transitions[prechain]

		probability = 0
		if postchain in transition:
			probability = transition[postchain] / transition['##sum']
		return probability

	def generate_next_character(self, word_ngram, end_character=True):
		"""Generate next character for the word ngram being built
		end character can be excluded from transitions, but if there are no other variants for the ngram,
		then end character will be returned
		"""
		transition = self.transitions[word_ngram]
		transition_items = transition.items()
		transition_items = list(filter(lambda x: x[0] != '##sum', transition_items))

		ending_character_occurences = 0
		if end_character == False and Dictionary.END_CHARACTER in transition:
			ending_character_occurences = transition[Dictionary.END_CHARACTER]
			transition_items = list(filter(lambda x: x[0] != Dictionary.END_CHARACTER, transition_items))

		if len(transition_items) == 0:
			return Dictionary.END_CHARACTER

		random_number = randint(1, transition['##sum'] - ending_character_occurences)

		# generate next character according to the probability mass function
		# indicated by ngram transitions
		# ie, find the index of suitable transition item
		current_sum = 0
		transition_index = -1
		while True:
			transition_index += 1
			current_sum += transition_items[transition_index][1]
			if random_number <= current_sum:
				break

		return transition_items[transition_index][0]


	def build_prechain_from_ngram(ngram):
		"""Build prechain for an ngram, ie, all the characters in the beginning that influence the postchain
		this should be a string/sequence (of states) of length equal to model order
		"""
		return ngram[:-1]

	def build_postchain_from_ngram(ngram):
		"""Build postchain for an ngram, ie, all the characters in the end that are influenced by the prechain
		this should be a string/sequence of length equal to number of characters/states being generated from the prechain
		this model currently predicts one character ahead at each step, so the postchain length is one character
		"""
		return ngram[-1:]
	