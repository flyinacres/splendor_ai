from __future__ import print_function
from constants import * 
from itertools import combinations
from itertools import zip_longest

import numpy as np 

"""
uncommon strategies not supported:

* using gold instead of a regular color in order to prevent a player from grabbing one or two gems of a particular color
on their turn (fairly uncommon)

* telling a player what your hidden card is so that it is more obvious to them that you can
purchase something/are more likely to purchase something that may further their interests in some way (very rare)

* take 1 gem each from 2 separate piles to allow another player to take a certain color or to allow yourself to take 3 next turn (never seen it)

"""

# TODO : ensure that players take turns properly
# TODO : make sure that proper move configs are passed to serializers
# TODO : implement AI and neural network structures
# TODO : make sure that AI-readable history is recorded 
#        includes Q-lag=1,3,5 (for points) and end-game result
#        see ticket_ai for examples
# TODO : make sure that human-readable history can be recorded
# TODO : make sure that some per-game statistics in human-readable history can be recorded (e.g., starting board state)
# TODO : test run

class Player(object):
	def __init__(self, game, id, order, ai=None, decision_weighting=None):
		self.game = game 
		self.id = id
		self.order = order 
		if ai is not None:
			self.ai = ai
		else:
			self.ai = SplendorAI()

		self.points = 0
		self.decision_weighting=decision_weighting
		#cards that contribute to cost reduction and points
		self.owned_cards = []
		#cards that can be purchased only by player
		self.reserved_cards = []
		#faster way of keeping track of cards
		self.n_cards = 0
		self.n_reserved_cards = 0
		self.n_reserved_cards_tiers = [{i:0 for i in range(3)}]
		self.gems = ColorCombination(True, {color:0 for color in COLOR_ORDER})
		self.n_gems = 0
		self.discounts = ColorCombination({color:0 for color in COST_COLOR_ORDER})
		self.objectives = []
		self.win = False
		#self.draw = False#will allow multiple victories in rare instances

		#describes the history for the current game
		#this should be in [[data], ...] format, where [data] is a numpy vector of the game state
		# from the perspective of the player
		self.history = []
		#describes history for all games; this will be used to train the neural network
		#this should be in [[data, win, game_id], ...] format
		self.extended_history = []

	def set_decision_weighting(self, weights):
		self.decision_weighting = weights

	def get_other_players(self):
		"""
		run when initializing the game to make it easy to access other players 
		for serializing
		"""
		n_players = len(self.game.players)
		if self.order==0:
			self.other_players = self.game.players[1:]
		elif self.order < n_players - 1:
			self.other_players = self.game.players[self.order+1:] + self.game.players[:self.order]
		else:
			self.other_players = self.game.players[:self.order]

	def take_turn(self):
		"""
		TODO: record player state at end of turn after action is made

		the player takes a turn

		"""
		self.make_turn_action()

		self.objective_check()
		if not self.game.last_turn:
			self.victory_check()

	def decide_among_options(self, purchasing_options, reserving_options, gem_taking_options):
		# determine parameters used to weight the 3 categories amongst themselves, then the options among the categories


		# there is a default weight associated with each major type of action

		# this multiplies the score of each action in a category

		# there is a temperature parameter in decision_weighting that uses boltzmann sampling from the resultant score

		# 


	def make_turn_action(self):
		"""
		
		"""
		purchasing_options = self.simulate_purchasing_options()
		reserving_options = self.simulate_reserving_options()
		gem_taking_options = self.simulate_gem_taking_options()

		# now determine best course of action based on q-weighting

		# this code no longer applies
		all_options = purchase_options + reserving_options + gem_taking_options 
		option = self.determine_best_option(all_options)
		if option is None:
			#really shouldn't happen
			print("PLAYER CANNOT TAKE ACTION")
			return 1
		#option[2] is action type
		if option[2]=='purchase_available_card':
			self.purchase_available_card(**option[1])
		elif option[2]=='purchase_reserved_card':
			self.purchase_reserved_card(**option[1])
		elif option[2]=='reserve_card_on_top':
			self.reserve_card_on_top(**option[1])
		elif option[2]=='reserve_card_on_board':
			self.reserve_card_on_board(**option[1])
		elif option[2]=='take_gems':
			self.take_gems(**option[1])
		return 0

	"""
	BELOW ARE SIMULATION FUNCTIONS
	for each of these functions, the following should be done:
		1. all possible moves should be determined
		2. for each possible move, a semi-simulation should be made of the resulting board state. 
		   I recommend implementing & using save_state()/load_state() for both the Player and Game classes
		   so that you can directly modify these; I have put a "move_cards" argument to some functions so that they won't 
		   shift around cards if you want to avoid that; you should use copy/deepcopy to copy lists/nested objects 
		   when creating these states
		3. for each state, there should be a serialization of the entire game from that player's point of view. For example,
		   a game of texas hold 'em would have a binary (1s and 0s) vector of length 52 describing  your hand, a binary vector
		   describing the state of the cards in the center, and perhaps some other vectors describing betting/folding of 
		   the other players (since you can't see their cards). In this game, you may be able to use integer values for gem costs.
		4. the SplendorAI class needs to be further developed, but it will take in these serializations and output the probabilities;
		   With these probabilities, a decision should be made by processing them through determine_best_option(). Initially I want the
		   actual probability to have less impact on the chosen decision in order to increase randomness, but as the network becomes
		   trained, I want the better probabilities to have a much higher chance of being chosen
		5. The chosen action will be taken with the keyword arguments provided in the option, and the turn will continue.


	"""

	def simulate_purchasing_options(self):
		"""
		creates all possible purchasing options and returns corresponding probabilities and action types

		FORMAT:
		 list of
		  [prob, action_kwargs, action_type]
		"""

		purchasing_options = []
		payment_options = []

		# cards on board
		for tier in [1,2,3]:
			available_cards = self.game.get_available_cards(tier=tier)
			deck_cards = self.game.get_deck(tier=tier)
			can_be_replaced = len(deck_cards) > 0
			for i, card in available_cards:
				net_cost = (card['cost'] - self.discount).truncate_negatives()
				if self.gems.can_pay_for(net_cost):
					payment_options.append(self.gems.calculate_actual_cost(net_cost))
					card_purchase = {
						'position': i,
						'tier': tier,
						'can_be_replaced': can_be_replaced,
						'type': 'board',
						'card': card
					}
					purchasing_options.append(card_purchase)

		# reserved cards
		for i, card in enumerate(self.reserved_cards):
			if card is not None:
				net_cost = (card['cost'] - self.discount).truncate_negatives()
				if self.gems.can_pay_for(net_cost):
					payment_options.append(self.gems.calculate_actual_cost(net_cost))
					card_purchase = {
						'position': i,
						'tier': tier,
						'can_be_replaced': False,
						'type': 'reserved',
						'card': card

					}
					purchasing_options.append(card_purchase)

		if len(purchasing_options) > 0:
			purchasing_serializations = self.full_serialization(
				gem_changes=payment_options,
				card_changes=purchasing_options,
			)
			predictions = self.ai.make_predictions(purchasing_serializations)
			return {
				'predictions': predictions,
				'actions': {
					'gem_changes': payment_options,
					'card_changes': purchasing_options,
				}
			}
		else:
			return None

	def simulate_reserving_options(self):
		"""
		creates all possible reserving options and returns corresponding probabilities and action types

		FORMAT:
		 list of
		  [prob, action_kwargs, action_type]
		"""
		# skip if 3 cards already reserved
		if len(self.reserved_cards) == 3:
			return []

		# cannot get gold if none exists or you have 10 gems
		if self.game.gems.gold==0 or self.gems.count() == 10:
			gem_change = ColorCombination(uses_gold=True)
		else:
			gem_change = ColorCombination(gold=1,uses_gold=True)

		reservation_options = []
		# cards on board
		for tier in [1,2,3]:
			tier_cards = self.game.get_available_cards(tier=tier)
			for i, card in enumerate(tier_cards):
				if card is not None:
					reservation_options.append({'tier':tier, 'position':position, 'type':'board'})

		# cards on top of deck
		for tier in [1,2,3]:
			if len(self.game.get_deck(tier=tier)) > 0:
				reservation_options.append({'tier':tier, 'position': 0, 'type':'topdeck'})

		if len(reservation_options) > 0:
			gem_changes = [gem_change] * len(reservation_options)
			reservation_serializations = self.full_serialization(
				gem_changes=gem_changes,
				reservation_changes = reservation_options
			)

			predictions = self.ai.make_predictions(reservation_serializations)
			return {
				'predictions':predictions, 
				'actions': {
					'gem_changes': gem_changes, 
					'reservation_changes':reservation_options
				}
			}
		else:
			return None


	def simulate_gem_taking_options(self):
		"""
		creates all possible gem-taking options and returns corresponding probabilities and action types

		FORMAT:
		 list of
		  [prob, action_kwargs, action_type]
		"""
		# skip if has 10 gems
		if self.gems.count() == 10:
			return []

		# determine all combinations
		gem_combinations = self.take_gems_options()
		if len(gem_combinations) > 0:
			gem_serializations = self.full_serialize(gem_changes=gem_combinations)
			predictions = self.ai.make_predictions(gem_serializations)
			return {
				'predictions': predictions,
				'actions': {
					'gem_changes': gem_combinations,
				}
			}
		else:
			return None

	def determine_best_option(self, all_options):
		"""
		will use game-based hyperparameters to decide on best options
		namely, a "fuzziness" parameter will select a choice based on the probabilities, and 
		earlier in the training process, there will be more leeway in determining 

		
		"""
		if len(all_options) == 0:
			return None

	def save_state(self):
		"""
		I recommend using this for storing a few attributes when simulating so you 
		can directly modify some of the player attribute variables;
		you can also have game.save_state() called from here
		"""

	def load_state(self):
		"""
		loads previous state after self.save_state() has been called;
		you can alsoe have game.load_state() called from here
		"""

	#GENERAL CHECKING AND ACTION FUNCTIONS
	def get_available_cards(self, tier):
		"""
		does not include reserved cards
		"""
		if tier==1:
			cards = self.game.available_tier_1_cards
		elif tier==2:
			cards = self.game.available_tier_2_cards
		else:
			cards = self.game.available_tier_3_cards
		return cards 

	def card_in_position_purchasing_cost(self, tier, position):
		card = self.get_available_cards(tier)[position]
		return self.card_purchasing_cost(card)

	def card_purchasing_cost(self, card):
		"""
		this doubles as a boolean check
		"""
		cost = card['cost'] - self.discounts
		

		#calculates the surplus of each color for each part of the cost
		difference = self.gems - cost 
		#checks to see if extra gold is enough to purchase 
		if not self.gems.can_pay_for(cost):
			return None
		else:
			return cost.truncate_negatives() #new_gems = self.gems.make_payment(cost)

	def purchase_available_card(self, tier, position, move_cards=True):
		card = self.get_available_cards(tier).pop(position)
		card_cost = self.card_purchasing_cost(card)
		#add points
		self.points+= card['points']
		#add color # still technically valid with new class
		self.discounts[card['color']] += 1
		#subtract gems
		self.gems = self.gems.make_payment(card_cost)
		'''
		for color, amount in card_cost.iteritems():
			self.gems[color] -= amount 
			self.n_gems -= amount
		'''
		#add card to inventory
		self.n_cards += 1
		if move_cards:
			self.owned_cards.append(card)
			#add new card, display warning if deck is empty
			new_card_added = self.game.add_top_card_to_available(tier)
			if not new_card_added:
				print("WARNING: tier %s deck ran empty!" % tier)

	def purchase_reserved_card(self,  position):
		if move_cards:
			card = self.reserved_cards.pop(position)
		else:
			card = self.reserved_cards[position]
		card_cost = self.card_purchasing_cost(card)
		#add points
		self.points+= card['points']
		#add color
		self.discounts[card['color']] += 1
		#subtract gems
		self.gems = self.gems.make_payment(card_cost)
		
		#add card to inventory
		self.n_cards += 1
		if move_cards:
			self.owned_cards.append(card)
		self.n_reserved_cards -=1
		self.n_reserved_cards_tiers[card['tier']] -= 1

	def reserve_card_on_top(self, tier, move_cards=True):
		self.n_reserved_cards += 1
		if move_cards:
			self.reserved_cards.append(self.game.get_deck(tier).pop())
		self.n_reserved_cards_tiers[tier] += 1
		if self.n_gems < 10 and game.gems['gold'] > 0:
			self.take_gems({'gold':1})
		return 0

	def reserve_card_on_board(self. tier, position):
		self.n_reserved_cards += 1
		self.reserved_cards.append(self.game.get_available_cards(tier).pop(position))
		self.game.add_top_card_to_available(tier)
		self.n_reserved_cards_tiers[tier] += 1
		if self.n_gems < 10 and game.gems['gold'] > 0:
			self.take_gems({'gold':1})
		return 0 

	def objective_check(self):
		"""
		at end of turn, checks to see if any objectives are met
		if no, then returns 0
		if only one, then it adds that and returns a 1
		if more than one, decides which one, then adds it, then returns 1
		"""
		possible_objective_ids = []
		for i, objective in enumerate(self.game.objectives):
			if self.discounts.can_pay_for(objective):
				possible_objective_ids.append(i)
		if len(possible_objectives) == 0:
			return 0
		elif len(possible_objectives) == 1:
			self.points+=3 
			self.objectives.append(self.game.objectives.pop(possible_objectives[0]))
		else:
			objective_id = self.decide_on_objective(possible_objective_ids)
			self.points+=3
			self.objectives.append(self.game.objectives.pop(objective_id))	
		return 1

	#TODO
	def decide_on_objective(self, possible_objectives):
		"""
		MEH METHOD: randomly choose an objective
		AI METHOD: forecast all possibilities (3 at most, but most likely 2) and choose the one that most likely results in winning
		DERIVED METHOD: check to see if any other players are within a turn of getting the other card; not 100% exact bc of reserved cards, and might be tedious
		"""

	def victory_check(self):
		"""
		if premature, will not compare number of cards and will 
		"""
		if self.score >= 15:
			self.game.last_turn = True

	def take_gems(self, gems):
		"""
		adds gems to inventory
		will not update code for ColorCombination class
		"""
		for color, amount in gems.iteritems():
			self.gems[color] += amount
			self.game.gems[color] -= amount
			self.n_gems +=1
		return 0

	def take_gems_options(self):
		# returns a list of ColorCombination objects

		# will not update
		possibilities = []
		if self.n_gems == 10:
			return []
		elif self.n_gems == 9:
			#check all piles that have at least one
			#can only take 1
			for color in COST_COLOR_ORDER:
				if self.game.gems[color] > 0:
					possibilities.append(ColorCombination({color:1}))
			return possibilities

		#check all piles that have at least one and all piles with at least 4
		#can only take 2 from >=4 or 1 from any 2 if player has 8 gems
		#otherwise no additional restrictions
		single_colors = set()
		double_colors = set()
		for color in COST_COLOR_ORDER:
			if self.game.gems[color] > 0:
				if self.game.gems[color] > 3:
					double_colors.add(color)
				single_colors.add(color)
		for color in double_colors:
			possibilities.append(ColorCombination({color:2}))
		if self.n_gems == 8:
			if len(single_colors)==1:
				color = list(single_colors)[0]
				possibilities.append(ColorCombination({color:1}))
			else:
				possibilities += color_combinations(single_colors, 2)
		else:
			if len(single_colors)==1:
				color = list(single_colors)[0]
				possibilities.append({color:1})
			elif len(single_colors)==2:
				possibilities += color_combinations(single_colors, 2)
			else:
				possibilities += color_combinations(single_colors, 3)
		return possibilities

	def reset(self, reset_extended_history=False):
		"""
		used when you want to keep the same players (saves some AI loading, possibly) 
		and start a new game; extended history is kept unless the flag is set to true to
		reset it
		"""
		self.points = 0
		#cards that contribute to cost reduction and points
		self.owned_cards = []
		#cards that can be purchased only by player
		self.reserved_cards = []
		#faster way of keeping track of cards
		self.n_cards = 0
		self.n_reserved_cards = 0
		self.n_reserved_cards_tiers = [{i:0 for i in range(3)}]
		self.gems = ColorCombination({color:0 for color in COLOR_ORDER})
		self.n_gems = 0
		self.discounts = ColorCombination({color:0 for color in COST_COLOR_ORDER})
		self.objectives = []
		self.win = False
		#self.draw = False#will allow multiple victories in rare instances

		#describes the history for the current game
		self.history = []
		if reset_extended_history:
			self.extended_history = []

	# length 59-61 (57 + number of players)
	def serialize(self, gem_change=None, card_change=None, reservation_change=None, from_own_perspective=True):
		"""
		# note: this is for purchasing a card, not reserving
		card_change - {'card': {...}, 'reserved_index': [0,1,2, None], 'position': [0,1,2,3, None]}
		  reserved_index is the index number of the reserved card in the player's inventory
		  position is the position of the card in its tier row if it's not reserved; does nothing in this function

		card_reservation - {'type': 'board'/'topdeck', 'tier': [1,2,3], 'position':[0,1,2,3]}
		  type is if itis on the board or if it is on top of a deck
		  tier corresponds to the row or deck tier
		  position corresponds to the position on the board; 0 if on top of the deck
		"""
		# gem calculations
		if gem_change is not None:
			theoretical_gems = self.gems + gem_change
		else:
			theoretical_gems = self.gems 

		gem_serialization = theoretical_gems.serialize()

		# card changes
		reserved_card_serializations = [
			serialize_card(card, not from_own_perspective)
			for card in self.reserved_cards
		]

		# hypothetical reservation
		if reservation_change is not None:
			tier = card_reservation['tier']
			if reservation_change['type'] == 'topdeck':
				reserved_card_serializations.append(
					serialize_card(make_blank_card(tier=tier))
				)
			else:
				position = reservation_change['position']
				reserved_card_serializations.append(
					serialize_card(self.game.get_available_cards(tier=tier)[position])
				)

		n_reserved = len(reserved_card_serializations)

		# fill in with blank reservation slots (they still need 0-filled serializations)
		if n_reserved < 3:
			reserved_card_serializations.extend((3-n_reserved) * [PURE_BLANK_CARD_SERIALIZATION])

		if card_change is not None:
			card = card_change['card']
			theoretical_points = card_change['card']['points'] + self.points 
			if card_change['type'] == 'reserved'
				reserved_index = card_change['position']
			else:
				reserved_index = None
			color = card['color']
			
			if reserved_index is not None:
				reserved_card_serializations[reserved_index] = PURE_BLANK_CARD_SERIALIZATION
			else:
				# do nothing because that doesn't affect the player, only the board state which is handled elsewhere
				pass
			theoretical_discount = self.discount + ColorCombination(**{color:1})
		else:
			theoretical_points = self.points
			theoretical_discount = self.discount 

		discount_serialization = theoretical_discount.serialize()
		points_serialization = np.asarray([theoretical_points])

		# order serialization to enforce symmetry requirements
		order_serialization = np.zeros(4)
		order_serialization[self.order] = 1.

		return {
			'gems': gem_serialization, #6
			'discount': discount_serialization, #5
			'points': points_serialization, # 1
			'reserved_cards': reserved_cards_serializations, # 3 * 15
			'order': order_serialization, #2-4 (number of players in the game)
		}



	def full_serializations(self, gem_changes=None, card_changes=None, reservation_changes=None):
		# first get serializations of other players, since those are static
		other_player_serializations = [
			player.serialize(from_own_perspective=False)
			for player in self.other_players
		]

		# for each change combination, get own serialization and board serialization
		self_serializations = []
		game_serializations = []
		for gem_change, card_change, reservation_change in zip_longest(gem_changes, card_changes, reservation_changes):

			self_serializations.append(self.serialize(gem_change=gem_change, card_change=card_change))

			game_serializations.append(self.game.serialize(
				gem_change=gem_change, 
				card_change=None, 
				reservation_change=reservation_change)
			)

		# return those values, which will be ready to be consumed by neural network input
		return {
			'other_players': other_player_serializations,
			'self': self_serializations,
			'game': game_serializations,
		}

def color_combinations(colors, n):
	"""
	returns the cost in the form of a dictionary 
	"""
	combos = combinations(colors, n)
	return [ColorCombination({color:1 for color in combo}) for combo in combinations]
