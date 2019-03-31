from collections import Counter

class Dictionary:
	"""Dictionary holds words and represents the language to use for model training

	Loads words, builds substrings and ngrams and generates counters of them
	"""

	START_CHARACTER = '^'
	END_CHARACTER = '$'

	def __init__(self, dictionary_filename, sort=True):
		"""Create a new dictionary object, loads words from dictionary_filename
		and sort the list of words if sort is True
		"""
		self._dictionary_filename = dictionary_filename
		# currently keeps track of words as a list and set
		# could be merged into a multiset/dictionary/counter
		# to allow for word repetitions/occurences/weights
		# to keep track of their frequency (and frequency of constituent ngrams)
		# with efficient retrieval/membership testing (eg: hashing or binary trees)
		self.words = Dictionary.load_words(dictionary_filename)
		self.words_set = set(self.words)
		if sort:
			self.words.sort()

	def load_words(dictionary_filename):
		"""Load words from dictionary text file into a list"""
		with open(dictionary_filename) as file:
			return file.read().splitlines()

	def filter_words(self, filter_predicate):
		"""Filter words in the dictionary"""
		self.words = list(filter(filter_predicate, self.words))
		self.words_set = set(filter(filter_predicate, self.words_set))

	def map_words(self, transform):
		"""Map words in the dictionary to new values with transform function"""
		self.words = list(map(transform, self.words))
		self.words_set = set(map(transform, self.words))

	def build_counter_of_substrings(self, substring_length):
		"""Build counter made of substrings of given length of dictionary words"""
		return Counter(self.build_substrings(substring_length))

	def build_counter_of_ngrams(self, ngram_length):
		"""Build counter made of ngrams of given length of dictionary words"""
		return Counter(self.build_ngrams(ngram_length))

	def build_ngram_for(substring, ngram_length):
		"""Build ngram for a given substring
		Include starting characters if needed to pad to required ngram length
		"""
		return (Dictionary.START_CHARACTER * ngram_length + substring)[-ngram_length:]

	def build_ngrams(self, ngram_length):
		"""Build and return a list of ngrams of required length made from dictionary words"""
		ngrams = []
		for word in self.words:
			ngrams.extend(Dictionary.build_ngrams_of_word(word, ngram_length))
		return ngrams

	def build_ngrams_of_word(word, ngram_length, with_ending=True):
		"""Build ngrams of required length of given word
		Augment the word with appropriate number of start and end characters
		and build substrings of required length of such augmented word
		"""
		augmented_word = Dictionary.START_CHARACTER * (ngram_length - 1) + word
		augmented_word += Dictionary.END_CHARACTER if with_ending else ''
		ngrams = []
		for i in range(len(augmented_word) - ngram_length + 1):
			ngrams.append(augmented_word[i:i + ngram_length])
		return ngrams

	def build_substrings(self, substring_length):
		"""Build a list of substrings of current length of dictionary words
		include a word itself, when it is shorter than required substring length
		"""
		substrings = []
		for word in self.words:
			substrings.extend(Dictionary.build_substrings_of_word(word, substring_length))
		return substrings

	def build_substrings_of_word(word, substring_length):
		"""Build substrings of required length of given word
		when the word is shorter than the substring length, return a list with the word
		"""
		if len(word) <= substring_length:
			return [word]
		substrings = []
		for i in range(len(word) - substring_length + 1):
			substrings.append(word[i:i + substring_length])
		return substrings

	def __contains__(self, word):
		"""Check if the dictionary contains given word"""
		return word in self.words_set
