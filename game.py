from __future__ import print_function
from player import Player 
from player import get_phase_parameters

from constants import *
from copy import copy, deepcopy
from random import shuffle
from itertools import chain

import numpy as np

def gem_reduction(n_players):
	# number of gems to take out given 2, 3, and 4-player games
	if n_players==3:
		return 2
	elif n_players==2:
		return 3
	else:
		return 0

default_decision_weighting = get_phase_parameters(phase=1)

class Game(object):
	"""
	TODO: add some game-based hyperparameters that determine 
	how players use the posterior probabilities of their decisions
	in order to make an action

	"""
	def __init__(self, id, n_players=4, players=None, shuffle_players=True, record_plain_history=False,
		decision_weighting=default_decision_weighting, temperature=1):
		"""
		id is used for recordkeeping, ideally a counter
		n_players is the number of players in the game; if players are provided already, then this will revert to their length
		shuffle_players will shuffle the player order; recommended
		record_plain_history will record easier-to-understand data for each turn; this increases memory usage dramatically and makes 
			the program a lot slower, so use it only when debugging or when extracting simulation information for insights
		"""
		self.id = id 
		self.record_plain_history = record_plain_history
		if self.record_plain_history:
			self.plain_history = []
		else:
			self.plain_history = None 
		self.round=-1
		if players is not None:
			self.n_players = len(players)
			self.players=players 
			if shuffle_players:
				self.shuffle_players()
		else:
			self.n_players = n_players
			self.players = [
			Player(game=self, id=i, order=i, record_plain_history=record_plain_history,
				decision_weighting=decision_weighting, temperature=temperature) 
			for i in range(n_players)]
		self.turn = -1

		# make sure each player knows where other players are relative to them
		for player in self.players:
			player.get_other_players()

		self.generate_initial_cards()
		#self.gems = {color:COLOR_STOCKPILE_AMOUNT - (4-self.n_players) for color in COLOR_ORDER}
		#self.gems['gold'] = GOLD_STOCKPILE_AMOUNT
		self.gems = GEMS_PILE - gem_reduction(n_players) * EACH_COLOR
		self.last_turn = False

	#TODO: add methods to append 
	def run(self):
		while not self.last_turn:
			self.round+=1
			for i, player in enumerate(self.players):
				self.turn += 1
				player.active_turn=True
				player.take_turn()
				player.active_turn=False
				if self.record_plain_history:
					game_data = self.copy_plain_data(player_order=i)
					player_data = [a_player.copy_plain_data() for a_player in self.players]
					self.plain_history.append(player_data)

		self.assign_winner()
		for player in self.players:
			# allows Q1, Q3, and Q5 to be applied as if a new turn's stats were recorded
			player.record_q_state()
			# transfers history to extended history, which can be passed to an AI to train
			player.record_extended_history()

	def shuffle_players(self):
		"""
		before starting a new game when you get a previous list of players
		"""
		shuffle(self.players)
		for i, player in enumerate(self.players):
			player.order = i 

	def generate_initial_cards(self):
		self.tier_1_cards = deepcopy(TIER_1_CARDS)
		self.tier_2_cards = deepcopy(TIER_2_CARDS)
		self.tier_3_cards = deepcopy(TIER_3_CARDS)
		shuffle(self.tier_1_cards)
		shuffle(self.tier_2_cards)
		shuffle(self.tier_3_cards)
		objectives = deepcopy(OBJECTIVE_CARDS)
		shuffle(objectives)
		self.objectives = objectives[:self.n_players + 1]

		self.available_tier_1_cards = [self.tier_1_cards.pop() for _ in range(4)]
		self.available_tier_2_cards = [self.tier_2_cards.pop() for _ in range(4)]
		self.available_tier_3_cards = [self.tier_3_cards.pop() for _ in range(4)]

	def add_top_card_to_available(self, tier):
		"""
		after a card is purchased, a new card from the deck should be added
		returns False if deck run empty (probably only happens in tier 1)
		"""
		cards = self.get_deck(tier)
		if len(cards)==0:
			return False
		self.get_available_cards(tier).append(cards.pop())
		return True

	def get_deck(self, tier):
		if tier==1:
			return self.tier_1_cards
		elif tier==2:
			return self.tier_2_cards
		else:
			return self.tier_3_cards 

	def get_available_cards(self, tier):
		if tier==1:
			return self.available_tier_1_cards
		elif tier==2:
			return self.available_tier_2_cards
		else:
			return self.available_tier_3_cards

	def assign_winner(self):
		"""
		TODO: update player extended histories and assign the proper win (1 or 0) value to the second element of each row
		"""
		# score primary objective
		max_score = max([player.points for player in self.players])
		highest_scoring_players = [player for player in self.players if player.points==max_score]

		# nobles secondary objective
		max_objectives = max([len(player.objectives) for player in highest_scoring_players])
		highest_scoring_players = [player for player in highest_scoring_players if len(player.objectives)==max_objectives]

		# efficiency tertiary objective
		lowest_number_of_cards = min([player.n_cards for player in highest_scoring_players])
		winning_players = [player for player in highest_scoring_players if player.n_cards==lowest_number_of_cards]
		for player in winning_players:
			player.win = True 

	# length = 203-213 (188 + (3-5) * 5)
	def serialize(self, gem_change = None, available_card_change=None, reservation_change=None):
		"""
		describes the state of the cards and gems on the board in a numeric format using a numpy array

		IMPORTANT: the position of the cards on the board is unimportant within a tier. when implementing a 
		neural network, try to get the values of the connections from their nodes to be the same as each other 
		(at least within the same tier)

		available_card_change - {'tier': [1,2,3], 'position': [0,1,2,3], 'can_be_replaced': True/False, 'type': 'board'/'reserved'}
		"""

		# gem calculations
		if gem_change is not None:
			theoretical_gems = self.gems + gem_change
		else:
			theoretical_gems = self.gems 
		gem_serialization = theoretical_gems.serialize()

		# available card change
		
		tier_1_cards_serialization = [serialize_card(card) for card in self.available_tier_1_cards]
		tier_2_cards_serialization = [serialize_card(card) for card in self.available_tier_2_cards]
		tier_3_cards_serialization = [serialize_card(card) for card in self.available_tier_3_cards]

		# add blank cards if serializations are too short

		if len(tier_1_cards_serialization) < 4:
			difference = 4 - len(tier_1_cards_serialization)
			tier_1_cards_serialization += [PURE_BLANK_CARD_SERIALIZATION] * difference

		if len(tier_2_cards_serialization) < 4:
			difference = 4 - len(tier_2_cards_serialization)
			tier_2_cards_serialization += [PURE_BLANK_CARD_SERIALIZATION] * difference

		# not possible with tier 3 cards to end up with no cards left

		# make an adjustment to one of the row serializations if the reservation change is not null
		if reservation_change is not None:
			if reservation_change['type'] == 'board':
				tier = reservation_change['tier']
				position = reservation_change['position']
				if tier==1:
					target_serialization = tier_1_cards_serialization
					blank_value = (len(self.tier_1_cards) > 0) * 1
				elif tier==2:
					target_serialization = tier_2_cards_serialization
					blank_value = (len(self.tier_2_cards) > 0) * 1
				elif tier==3:
					target_serialization = tier_3_cards_serialization
					blank_value = (len(self.tier_3_cards) > 0) * 1

				target_serialization[position] = serialize_card(make_blank_card(tier=tier, blank_value=blank_value))

		available_cards_serializations = [tier_1_cards_serialization, tier_2_cards_serialization, tier_3_cards_serialization]

		if available_card_change is not None:
			if available_card_change['type'] == 'board':
				tier = available_card_change['tier']
				position = available_card_change['position']
				can_be_replaced = available_card_change.get('can_be_replaced', True)
				available_cards_serializations[tier-1][position] = serialize_card(
					make_blank_card(tier,can_be_replaced )
				)

		# objectives serializations
		objectives_serializations = [serialize_objective(objective) for objective in self.objectives]
		if len(objectives_serializations) < self.n_players + 1:
			difference = self.n_players + 1 - len(objectives_serializations)
			objectives_serializations += [ColorCombination().serialize()] * difference

		# turn and last turn
		turn_serialization = [np.asarray(self.turn)]
		last_turn_serialization = [np.asarray(1*self.last_turn)]

		return {
			'gems': gem_serialization, # 6
			'available_cards': available_cards_serializations, # 12 x 15
			'objectives': objectives_serializations, # (3-5) x 5
			'turn': turn_serialization + last_turn_serialization, #2
		}

	def __str__(self):
		return 'GAME PLAYERS:\n' + '\n'.join([str(player) for player in self.players])
		
	def copy_plain_data(self, player_order=None):
		return {
			'available_cards': deepcopy(self.available_cards),
			'gems': deepcopy(self.gems),
			'objectives': deepcopy(self.objectives),
			'turn': deepcopy(self.turn),
			'player_order': player_order,
			'round': self.round,
			'id': self.id 
		}



	def save_state(self):
		"""
		to be used in conjunction with save_state() from players to save a state and then undo it as necessary
		"""

	def load_state(self):
		"""
		to revert to previous state stored by self.save_state()
		"""