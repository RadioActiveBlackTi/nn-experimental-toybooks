import random
import numpy as np
from torch.utils.data import Dataset

pronouns = ['I', 'You', 'He', 'She', 'We', 'They', 'Human', 'Everyone', 'Someone', 'Nobody', 'Anybody', 'Everybody']
animals = ['Dog', 'Cat', 'Bird', 'Fish', 'Elephant', 'Lion', 'Tiger', 'Bear', 'Rabbit', 'Horse', 'Cow', 'Sheep', 'Pig']
foods = ['Apple', 'Bread', 'Meat', 'Cheese', 'Rice', 'Pasta', 'Vegetables', 'Fruits', 'Eggs', 'Fish', 'Chicken', 'Soup', 'Cake', 'Pizza', 'Salad', 'Sandwich', 'Steak', 'Pancakes', 'Waffles', 'Ice-Cream']
beverages = ['Water', 'Milk', 'Juice', 'Coffee', 'Tea', 'Soda', 'Beer', 'Wine', 'Smoothie', 'Cocktail', 'Lemonade', 'Hot-Chocolate', 'Iced-Tea', 'Energy-Drink', 'Sparkling-Water', 'Cider', 'Kombucha', 'Protein-Shake', 'Herbal-Tea', 'Mocktail']

adverbs = ['quickly', 'usually', 'often', 'always', 'never', 'sometimes', 'rarely', 'seldom', 'frequently', 'occasionally', 'regularly', 'constantly', 'periodically', 'intermittently', 'sporadically', 'gradually', 'suddenly', 'abruptly', 'immediately', 'instantly', 'similarly', 'differently', 'uniquely', 'distinctly', 'specifically', 'generally', 'broadly', 'narrowly', 'precisely', 'accurately', 'approximately', 'roughly', 'exactly', 'clearly', 'obviously', 'evidently', 'apparently', 'visibly']

food_verbs = ['eat', 'like', 'hate', 'prefer', 'store', 'dislike', 'want', 'deposit']
drink_verbs = ['drink', 'like', 'hate', 'prefer', 'store', 'dislike', 'want', 'deposit', 'empty', 'full']
animal_verbs = ['see', 'like', 'love', 'feed', 'pat', 'hug', 'embrace', 'abandon', 'hit']

semantic_frames = [
    (pronouns + animals, food_verbs, foods),
    # Type 1: Food Preference
    (pronouns, drink_verbs, beverages),
    # Type 2: Beverage Preference
    (pronouns, animal_verbs, animals),
    # Type 3: Animal Interaction
]

vocab = list(set(pronouns + animals + foods + beverages + adverbs + 
                 food_verbs + drink_verbs + animal_verbs))

def generate_semantic_sentence(sentence_type):
    subj_list, verb_list, obj_list = random.choice(semantic_frames)
    
    subj = random.choice(subj_list)
    verb = random.choice(verb_list)
    obj = random.choice(obj_list)
    adv = random.choice(adverbs)
    
    match sentence_type:
        case 'A': # Subj + Adv + Verb + Obj (Standard)
            return f"{subj} {adv} {verb} {obj}"
        case 'B': # Adv + Subj + Verb + Obj (Fronted Adverb)
            return f"{adv} {subj} {verb} {obj}"
        case 'C': # Subj + Verb + Adv + Obj (Mid-Adverb)
            return f"{subj} {verb} {adv} {obj}"
        case _:
            raise ValueError("Invalid type")


def sentence_to_ids(sentence, vocab):
    words = sentence.split()
    return [vocab.index(word) for word in words]

class ALToyDataset(Dataset):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples
        self.sentences = [
            generate_semantic_sentence(random.choice(['A', 'B', 'C'])) 
            for _ in range(num_samples)
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        ids = sentence_to_ids(sentence, vocab)
        return torch.tensor(ids, dtype=torch.long)


ALL_SUBJ = set(pronouns + animals)
ALL_ADV = set(adverbs)
ALL_VERB = set(food_verbs + drink_verbs + animal_verbs)
ALL_OBJ = set(foods + beverages + animals)

def evaluate_single_sentence(sentence):
    """
    Returns a tuple (is_syntax_valid, is_semantic_valid) for the given sentence.
    """
    words = sentence.strip().split()
        
    w1, w2, w3, w4 = words
    
    subj, adv, verb, obj = None, None, None, None
    is_syntax_valid = False
    
    # [Syntax Rule 2] Sentence structure validation based on predefined patterns
    # Type A: Subj + Adv + Verb + Obj
    if (w1 in ALL_SUBJ) and (w2 in ALL_ADV) and (w3 in ALL_VERB) and (w4 in ALL_OBJ):
        subj, adv, verb, obj = w1, w2, w3, w4
        is_syntax_valid = True
    # Type B: Adv + Subj + Verb + Obj
    elif (w1 in ALL_ADV) and (w2 in ALL_SUBJ) and (w3 in ALL_VERB) and (w4 in ALL_OBJ):
        adv, subj, verb, obj = w1, w2, w3, w4
        is_syntax_valid = True
    # Type C: Subj + Verb + Adv + Obj
    elif (w1 in ALL_SUBJ) and (w2 in ALL_VERB) and (w3 in ALL_ADV) and (w4 in ALL_OBJ):
        subj, verb, adv, obj = w1, w3, w2, w4
        is_syntax_valid = True
        
    # If the syntax is invalid, we cannot proceed to semantic validation
    if not is_syntax_valid:
        return False, False
        
    # [Semantic Rule] Check if the (subj, verb, obj) combination is valid based on predefined semantic frames
    is_semantic_valid = False
    for valid_subjs, valid_verbs, valid_objs in semantic_frames:
        if (subj in valid_subjs) and (verb in valid_verbs) and (obj in valid_objs):
            is_semantic_valid = True
            break
            
    return is_syntax_valid, is_semantic_valid


def evaluate_batch(generated_sentences):
    """
    Evaluates a batch of generated sentences for syntax and semantic accuracy.
    """
    syntax_results = []
    semantic_results = []
    
    for sent in generated_sentences:
        syn_ok, sem_ok = evaluate_single_sentence(sent)
        syntax_results.append(syn_ok)
        semantic_results.append(sem_ok)
        
    syntax_acc = np.mean(syntax_results) * 100
    semantic_acc = np.mean(semantic_results) * 100
    
    # Conditional Semantic Accuracy (only for syntactically valid sentences)
    valid_syntax_count = np.sum(syntax_results)
    cond_semantic_acc = (np.sum(semantic_results) / valid_syntax_count * 100) if valid_syntax_count > 0 else 0.0
    
    return {
        "Syntax Accuracy": syntax_acc,
        "Semantic Accuracy": semantic_acc,
        "Conditional Semantic Accuracy": cond_semantic_acc
    }