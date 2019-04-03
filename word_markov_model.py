import math
from operator import itemgetter, mul
from random import randint
from functools import reduce
from queue import PriorityQueue
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
		total_probability = reduce(mul, ngrams_probabilities, 1)
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

	def find_most_probable_words(self, words_count=1):
		"""Find most probable words

		Find most probable words according to the trained model transitions
		words_count is an integer indicating how many of most probable words to return
		"""
		most_probable_words = []
		most_probable_queue = PriorityQueue()

		def word_entry(word, word_probability=None):
			"""Return a word entry for the priority queue
			The entry is a pair, first element is the priority which must reverse the ordering
			ie, the highest probability goes first, so the priority is the complementary probability
			priority is set to 1 - word probability
			the second element is the word itself
			the optional argument, word_probability, is the probability of the word to use
			if not given, the word probability is calculated using model's find_word_probability() method
			the ability to pass word probability as an argument is given
			to avoid recalculating word probability as the algorithm searches for next (eventually longer) words
			the probability can be efficiently updated at each transition, by multiplying words probability with transition probability"""
			if word_probability == None:
				word_probability = self.find_word_probability(word, with_ending=False)[0]
			return (1 - word_probability, word)

		current_word = ''
		most_probable_queue.put(word_entry(current_word))

		iteration = 0
		max_queue_size = 0

		# use priority queue to keep track of most probable words
		# starting with an empty word
		# append possible transitions to word currently expanded
		# and calculate the probability of such newly built words (by multiplying with corresponding transition probabilities)
		# the most probable words will have highest priorities so will be returned first from the priority queue
		# they will be also considered first, and expanded till the words are terminated with end character
		# once a terminated word is retrieved from the priority queue, 
		# it must have highest probability (higher than any other words that can be built from substrings that are in the queue)
		# hence greedily expanding substrings of highest probability 
		# till they are terminated with an end character (and retrieved from the priority queue)
		# gives most probable words
		while most_probable_queue.empty() == False and len(most_probable_words) < words_count:
			iteration += 1
			current_queue_size = most_probable_queue.qsize()
			if current_queue_size > max_queue_size:
				max_queue_size = current_queue_size

			current_entry = most_probable_queue.get()
			current_word = current_entry[1]
			current_word_probability = 1 - current_entry[0]

			if current_word[-1:] == Dictionary.END_CHARACTER:
				most_probable_words.append(current_word[:-1])
				continue

			current_ngram = Dictionary.build_ngram_for(current_word, self.model_order)
			current_transitions = WordMarkovModel.transition_occurences_to_probabilities(self.transitions[current_ngram])
			for postchain, probability in current_transitions:
				most_probable_queue.put(word_entry(current_word + postchain, current_word_probability * probability))

		return [list(map(lambda word: (word, self.find_word_probability(word)), most_probable_words)), iteration, max_queue_size]

	def transition_occurences_to_probabilities(transitions):
		"""Transform transition with occurences to transitions with probabilities"""
		occurences_sum = transitions['##sum']
		transition_items = list(filter(lambda x: x[0] != '##sum', transitions.items()))
		return list(map(lambda x: (x[0], x[1] / occurences_sum), transition_items))



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
	