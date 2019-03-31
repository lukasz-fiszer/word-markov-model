def language_probabilities(word, models):
	"""Return a list of probabilities for word in each model"""
	probabilities = list(map(lambda model: model.find_word_probability(word)[0], models))
	return probabilities

def language_membership(word, models):
	"""Return a list of relative probabilities for word in each model"""
	probabilities = language_probabilities(word, models)
	probabilities_sum = sum(probabilities)
	if probabilities_sum == 0:
		probabilities_sum = 1
	return list(map(lambda probability: probability / probabilities_sum, probabilities))
	