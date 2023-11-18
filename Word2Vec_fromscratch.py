<<<<<<< HEAD
#Tentative de re-création du modèle word2vec uniquement à l'aide de pandas et numpy et quelques autres packages de base

=======
# Attempt to recreate the word2vec model using only pandas, numpy, and a few other basic packages
>>>>>>> 8dd885f46119c6f01010742e87e38095cb9ba2f8
import random
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm

warnings.simplefilter("ignore", category=DeprecationWarning)

# I fetched a text on the theme of the beach from the internet as a training corpus. It's important to have a text revolving around the same term
# to have more effective learning. The downside is that the model can only predict words from texts revolving around the same theme as the
# training corpus

text = """La plage, cet espace de liberté où l'océan rencontre la terre, exerce sur nous une fascination intemporelle. 
Enveloppée de mystère et d'émerveillement, elle évoque des souvenirs doux et offre une échappatoire aux tracas du quotidien. 
Que ce soit le doux sable sous nos pieds, le bruit des vagues se brisant sur la côte ou le soleil caressant notre peau, la plage est un havre de paix pour l'âme et les sens.
La plage est un spectacle à part entière, où les éléments se mêlent harmonieusement. Le son des vagues, répétitif et régulier, a un effet hypnotique, nous transportant dans un état de relaxation profonde. 
Les nuances infinies de bleu de l'océan, se fondant avec le ciel, créent une toile de fond envoûtante. Le sable doré, caressé par le vent, invite à la contemplation et à la méditation. 
Les embruns salés emplissent l'air d'une fraîcheur revigorante. Enfin, le soleil généreux baigne tout de sa lumière chaleureuse, réchauffant notre corps et éveillant nos sens. 
La plage est une symphonie sensorielle qui nous transporte loin du tumulte de la vie quotidienne.
La plage a un pouvoir étonnant de libérer notre esprit des soucis et des contraintes du monde moderne. 
Marcher le long du rivage, les pieds dans l'eau, nous permet de nous reconnecter avec notre moi intérieur. L'immensité de l'océan nous rappelle notre place dans l'univers, nous offrant une perspective plus vaste. 
Les vagues qui se succèdent inlassablement symbolisent le cycle de la vie, nous invitant à accepter le flux constant du temps. La plage nous offre un espace pour réfléchir, méditer et nous recentrer sur l'essentiel. 
C'est un endroit où l'on peut se débarrasser du poids des responsabilités et laisser notre esprit errer librement.
La plage est un véritable sanctuaire de la nature, où les écosystèmes marins et côtiers coexistent en harmonie. 
Les plages abritent une grande diversité de vie, des coquillages colorés aux oiseaux marins en passant par les algues ondulant dans les marées. 
Observer ces merveilles naturelles nous rappelle la beauté fragile de notre planète et l'importance de la préserver. 
De plus, la plage est un lieu d'échange entre l'océan et la terre, où les vagues déposent des trésors marins échoués sur le sable, témoignant de la richesse des fonds marins. 
En explorant la plage, nous entrons en contact direct avec la nature, nous ressentons sa force et sa quiétude à la fois.
La plage est bien plus qu'un simple lieu de détente estivale. Elle est une source intarissable de joie, de sérénité et de renouveau. 
Que ce soit pour des moments de solitude réparatrice, des retrouvailles en famille ou des rencontres avec de nouveaux amis, la plage a le pouvoir de nous envelopper de son charme envoûtant. 
Elle nous rappelle notre lien indissoluble avec la nature et éveille nos sens à l'émerveillement devant sa beauté infinie. 
Alors, la prochaine fois que vous vous rendrez à la plage, laissez-vous porter par son éternelle magie et laissez votre âme se perdre dans l'horizon infini de l'océan.
Sous le chaud soleil estival, la plage s'étend comme un havre de paix où l'océan et la terre se rencontrent harmonieusement. C'est un lieu où les merveilles de la mer se révèlent à nos yeux émerveillés. 
À chaque pas que nous faisons sur le sable, nous découvrons des trésors marins échoués, des coquillages aux formes délicates et des étoiles de mer colorées. 
Ces fragments de l'océan témoignent de la diversité et de la richesse des fonds marins qui nous entourent.
En se promenant le long du rivage, nous entrons en communion avec la nature. Le bruit apaisant des vagues qui s'écrasent sur le rivage remplit nos oreilles, 
tandis que la brise marine caresse notre peau, nous offrant une sensation de fraîcheur et de liberté. La plage nous offre un précieux répit loin de l'agitation 
de la vie quotidienne, nous permettant de nous ressourcer et de retrouver notre équilibre intérieur.
Pourtant, la plage est bien plus qu'un simple refuge de tranquillité. Elle est un véritable élixir de joie et de vitalité. Que ce soit 
en partageant des moments de solitude réparatrice, en se réunissant en famille ou en faisant la rencontre de nouveaux amis, la plage 
possède une magie envoûtante qui nous enveloppe. Les jeux et les rires des enfants qui construisent des châteaux de sable, les 
promenades main dans la main le long de l'eau avec un être cher, ou les conversations animées entre amis autour d'un pique-nique improvisé,
 tous ces instants précieux nous rappellent l'importance des liens humains et du partage.
La plage est également un véritable spectacle visuel. Les nuances changeantes du ciel se reflétant sur la surface miroitante de l'océan 
nous invitent à la contemplation. Les couchers de soleil flamboyants peignent des tableaux célestes dignes d'admiration, tandis que les vagues 
scintillantes dansent en harmonie avec la lumière dorée. Chaque instant passé sur la plage nous rappelle la beauté infinie de notre monde naturel 
et nous incite à protéger et à préserver cette merveilleuse planète que nous appelons chez nous.
Alors, la prochaine fois que vous vous rendrez à la plage, abandonnez-vous à sa magie intemporelle. Laissez vos soucis s'envoler avec les mouettes 
qui planent au-dessus de l'eau. Plongez vos pieds dans l'océan, sentez le sable chaud glisser entre vos orteils et permettez à votre esprit de 
s'égarer dans l'horizon infini. La plage est un cadeau précieux qui nous rappelle notre connexion profonde avec la nature et nous offre un 
moment d'émerveillement et de renouveau. Sous le chaud soleil estival, la plage s'étend comme un havre de paix où l'océan et la terre se rencontrent 
harmonieusement. C'est un lieu où les merveilles de la mer se révèlent à nos yeux émerveillés. À chaque pas que nous faisons sur le sable, nous découvrons 
des trésors marins échoués, des coquillages aux formes délicates et des étoiles de mer colorées. Ces fragments de l'océan témoignent de la diversité et de 
la richesse des fonds marins qui nous entourent. En se promenant le long du rivage, nous entrons en communion avec la nature. Le bruit apaisant des vagues 
qui s'écrasent sur le rivage remplit nos oreilles, tandis que la brise marine caresse notre peau, nous offrant une sensation de fraîcheur et de liberté. La 
plage nous offre un précieux répit loin de l'agitation de la vie quotidienne, nous permettant de nous ressourcer et de retrouver notre équilibre intérieur. 
Pourtant, la plage est bien plus qu'un simple refuge de tranquillité. Elle est un véritable élixir de joie et de vitalité. Que ce soit en partageant des 
moments de solitude réparatrice, en se réunissant en famille ou en faisant la rencontre de nouveaux amis, la plage possède une magie envoûtante qui nous 
enveloppe. Les jeux et les rires des enfants qui construisent des châteaux de sable, les promenades main dans la main le long de l'eau avec un être cher, o
u les conversations animées entre amis autour d'un pique-nique improvisé, tous ces instants précieux nous rappellent l'importance des liens humains et du 
partage. La plage est également un véritable spectacle visuel. Les nuances changeantes du ciel se reflétant sur la surface miroitante de l'océan nous invitent 
à la contemplation. Les couchers de soleil flamboyants peignent des tableaux célestes dignes d'admiration, tandis que les vagues scintillantes dansent en 
harmonie avec la lumière dorée. Chaque instant passé sur la plage nous rappelle la beauté infinie de notre monde naturel et nous incite à protéger et à 
préserver cette merveilleuse planète que nous appelons chez nous. Alors, la prochaine fois que vous vous rendrez à la plage, abandonnez-vous à sa magie 
intemporelle. Laissez vos soucis s'envoler avec les mouettes qui planent au-dessus de l'eau. Plongez vos pieds dans l'océan, sentez le sable chaud glisser 
entre vos orteils et permettez à votre esprit de s'égarer dans l'horizon infini. La plage est un cadeau précieux qui nous rappelle notre connexion profonde 
avec la nature et nous offre un moment d'émerveillement et de renouveau."""

text = pd.Series(text)

text = text.str.lower()

# Remove punctuation using a regex expression
text = text.str.replace('[^\w\s]', '', regex=True)

tokens = text.str.split()[0]

# Create vocabulary (set of unique words in the corpus)
voc = list(set(tokens))

# Map words to an index (and vice versa) to convert a word into a vector and vice versa. This will help in
# creating the context and the one-hot encoded matrix
word2index = {word: index for index, word in enumerate(tokens)}
index2word = {index: word for index, word in enumerate(tokens)}
word2index_u = {word: index for index, word in enumerate(voc)}
index2word_u = {index: word for index, word in enumerate(voc)}
context_dict = {}
values = list(index2word.values())
max_index_ori = len(values) - 1



def extract_contexts_fully_random(tokens):
    """
    Extracts fully random contexts for each token in a list of tokens.
    """
    contexts = {}

    for i, token in enumerate(tokens):
        # Choisir une taille de contexte aléatoire
        context_size = random.randint(1, 10)

        # Choisir un indice de début aléatoire qui précède le mot cible
        start_index = random.randint(0, max(0, i))

        # Ajuster l'indice de début si nécessaire
        if i - start_index + 1 > context_size:
            start_index = i - context_size + 1

        # Déterminer l'indice de fin
        end_index = start_index + context_size

        # Ajuster si end_index dépasse la longueur des tokens
        if end_index > len(tokens):
            diff = end_index - len(tokens)
            start_index -= diff
            end_index -= diff

        # Extraire le contexte sans inclure le mot cible
        context = tokens[start_index:i] + tokens[i + 1:end_index + 1]

        # Associer le contexte à l'indice du mot cible dans le corpus
        contexts[i] = context

    return contexts


# Map words to unique index (and vice versa) to convert a word into a vector and vice versa
index2word_u = {index: word for index, word in enumerate(voc)}

one_hot_matrix = np.zeros((len(tokens), len(voc)), dtype=int)

for i, word in enumerate(tokens):
    one_hot_matrix[i,word2index_u[word]] = 1

# Map words to embeddings
def input(embed_etiq, context_dict, D, context_indices = None, context_unique = False):

    if context_unique == True :
        matrix = np.zeros((D, len(context_dict[context_indices])))
        for i, word in enumerate(context_dict[context_indices]):
            matrix[:, i] = embed_etiq[word]

        input = np.mean(matrix, axis=1)

    else :
        input = np.zeros((D, len(context_dict)))
        # Calcul de la moyenne des embeddings pour chaque contexte
        for i, words in enumerate(context_dict.values()):
            matrix = np.zeros((D, len(words)))

            for j, word in enumerate(words):
                matrix[:, j] = embed_etiq[word]

            input[:, i] = np.mean(matrix, axis=1)


    return input.T

def position_coef(target_word_index, len_tokens, context):
    # Si le mot cible est proche du début ou de la fin du corpus, ajuster le contexte

    if target_word_index < len(context) // 2:
        target_position = target_word_index

    elif len_tokens - target_word_index <= len(context) // 2:
        target_position = len(context) - (len_tokens - target_word_index)

    else:
        target_position = len(context) // 2


    # Fonction pour calculer le coefficient de position
    def compute_coef(distance):
        return 1 / (1 + np.log(1 + distance))


    # Créer un dictionnaire pour stocker les coefficients de position
    dct_word_pos_coef = {}

    for word in context:
        indices = [idx for idx, w in enumerate(context) if w == word]
        for i, index in enumerate(indices):
            if index >= target_position:
                indices[i] += 1


        # Si le mot apparaît plusieurs fois dans le contexte
        if len(indices) > 1:
            coefs = []
            for idx in indices:
                distance = abs(target_position - idx)
                coefs.append(compute_coef(distance))
            dct_word_pos_coef[word] = np.mean(coefs)
        else:
            idx = indices[0]
            distance = abs(target_position - idx)
            dct_word_pos_coef[word] = compute_coef(distance)

    return dct_word_pos_coef

position_coef(76, 1300, ['le', 'lapin', 'mange', 'pommes', 'dans', 'le', 'jardin'])

def softmax(x):
    if len(x.shape) == 1:
        e_x = np.exp(x - np.max(x)) # Soustraire le max pour des raisons de stabilité numérique
        output = e_x / np.sum(e_x)
    else:
        e_x = np.exp(x - np.max(x, axis=1, keepdims= True))
        output = e_x / np.sum(e_x, axis=1, keepdims=True)
    return output

def ADAM(i, gradient, m_i = 0, v_i = 0):
    beta1 = 0.9
    beta2 = 0.999
    if i == 0:
        m_i = (1 - beta1)*gradient
        v_i = (1 - beta2)*(gradient) ** 2
        m_i_hat = m_i / (1 - beta1 ** (i+1))
        v_i_hat = v_i / (1 - beta2 ** (i+1))

    else:
        m_i = (beta1 * m_i) + (1 - beta1) * gradient
        v_i = (beta2 * v_i) + (1 - beta2) * gradient ** 2
        m_i_hat = m_i / (1 - beta1 ** i)
        v_i_hat = v_i / (1 - beta2 ** i)

    return m_i, v_i, m_i_hat, v_i_hat

D = 75
gradient_accumulator = {word : np.zeros(D) for word in voc}
W1 = np.random.rand(D, len(voc)) - 0.5
embed_etiq = {word : W1[:, i] for i, word in enumerate(voc)}
eps = 10**(-8)
i = 0
m_i_w = 0
v_i_w = 0
m_i_b = 0
v_i_b = 0
nb_iterations = 10000
W2 = np.random.rand(D, len(voc)) - 0.5
B2 = np.random.randn(W2.shape[1]) - 0.5
alpha = 0.005
batch_size = int(len(context_dict)/100)
losses_graph = []
adam_context = {i : np.zeros(2) for i, word in enumerate(tokens)}
context_dict = extract_contexts_fully_random(tokens)



for j in tqdm.tqdm(range(nb_iterations)):
    batch_context = random.sample(context_dict.items(), batch_size)
    batch_context = dict(batch_context)

    for context_index, words in batch_context.items():

        batch_one_hot = one_hot_matrix[context_index]
        hidden_layer_input = input(embed_etiq, context_dict, D, context_index, context_unique= True)

        # Forward pass
        hidden_layer_output = np.matmul(hidden_layer_input, W2) + B2
        softmax_output = softmax(hidden_layer_output)

        losses = -np.sum(batch_one_hot * np.log(softmax_output))

        gradient = softmax_output - batch_one_hot
        gradient = gradient.reshape((-1, 1))
        hidden_layer_input = np.reshape(hidden_layer_input, (-1, 1))

        # Backward pass
        gradient_weight = np.dot(hidden_layer_input, gradient.T)
        gradient_bias = np.sum(gradient)

        m_i_w, v_i_w, m_i_hat_w, v_i_hat_w = ADAM(i, gradient_weight, m_i_w, v_i_w)
        m_i_b, v_i_b, m_i_hat_b, v_i_hat_b = ADAM(i, gradient_bias, m_i_b, v_i_b)
        W2 -= alpha * m_i_hat_w / (np.sqrt(v_i_hat_w) + eps)
        B2 -= alpha * m_i_hat_b / (np.sqrt(v_i_hat_b) + eps)

        gradient_embedding = np.dot(W2, gradient)/len(context_dict[context_index])
        gradient_embedding = gradient_embedding.reshape(-1)
        m_i_e, v_i_e, m_i_hat_e, v_i_hat_e = ADAM(j, gradient_embedding, adam_context[context_index][0], adam_context[context_index][1])

        for word in context_dict[context_index]:
            gradient_accumulator[word] = alpha * (m_i_hat_e / (np.sqrt(v_i_hat_e) + eps)) * position_coef(context_index, len(tokens), list(context_dict[context_index]))[word]
        # Mettre à jour les embeddings pour chaque mot dans le contexte
        for word in context_dict[context_index]:
            embed_etiq[word] -= gradient_accumulator[word]
        # Stocker les valeurs de l'optimiseur ADAM pour embeddings
        adam_context[context_index] = [m_i_e, v_i_e]


        i += 1

    hidden_layer_input_total = input(embed_etiq, context_dict, D)
    hidden_layer_output_total = np.matmul(hidden_layer_input_total, W2) + B2
    softmax_output_total = softmax(hidden_layer_output_total)
    losses = -np.sum(one_hot_matrix * np.log(softmax_output_total))
    losses_graph.append(losses)


    if (j + 1) % 100 == 0 and j > 299:
        print(f"la perte du modèle est de {np.mean(losses_graph[j - 299: j + 1])}")

plt.figure(figsize=(10, 5))

#create a list with the mean values of the losses every 100 iterations
losses_graph_400 = [np.mean(losses_graph[i - 400: i]) for i in range(400, len(losses_graph)) if i % 400 == 0]

plt.plot(losses_graph_400)
plt.xlabel('iterations')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
plt.show()



def testsurtrain(tokens, embed_etiq, context_dict, D, W2, B2, index2word_u):
    score = []
    for i in tqdm.tqdm(range(100)):
        hidden_layer_input = input(embed_etiq, context_dict, D)
        hidden_layer_output = np.matmul(hidden_layer_input, W2) + B2
        softmax_output = softmax(hidden_layer_output)
        if index2word_u[np.argsort(softmax_output[i])[-1]] == tokens[i]:
            score.append(1)
        else:
            score.append(0)
    return print(np.mean(np.array(score)))

testsurtrain(tokens, embed_etiq, context_dict, D, W2, B2, index2word_u)


arr = np.array([0, 1, 1, 0])


def minMoves(arr):
    # Write your code here
    last_index_arr = len(arr) - 1
    ones_indexes = sorted([index for index, value in enumerate(arr) if value == 1], reverse=True)
    print(ones_indexes)
    i = O
    for index in ones_indexes:
        if i == 0

        swaps += last_index_arr - index
    ones = 0
    swaps = 0

    for num in arr:
        swaps =
        if num == 1:
            ones += 1
        else:
            swaps += ones
    return swaps




def swaps_num(arr):
    ones_indexes_left = sorted([index for index, value in enumerate(arr) if value ==1])
    swaps_left = 0
    len_arr = len(arr)
    for i, index in enumerate(ones_indexes_left):
        if index == 0:
            continue
        else:
            swaps_left += index - i

    swaps_right = 0
    ones_indexes_right = sorted([index for index, value in enumerate(arr) if value == 1], reverse= True)
    for i, index in enumerate(ones_indexes_right):
        if index == len(arr) - 1:
            continue
        else:
            swaps_right += len(arr) - 1 - index - i
    return min(swaps_left, swaps_right)


swaps_num([0, 1, 1, 0, 1, 0, 1, 1, 1])

def swaps_num2(arr):
    zero_count_left = 0
    swaps_left = 0
    for element in arr:
        if element == 0:
            zero_count_left += 1
        else:
            swaps_left += zero_count_left

    reversed_array = arr[::-1]
    zero_count_right = 0
    swaps_right = 0
    for element in reversed_array:
        if element == 0:
            zero_count_right += 1
        else:
            swaps_right += zero_count_right

    return min(swaps_left, swaps_right)

swaps_num2([0, 1, 1, 0, 1, 0, 1, 1, 1])

import random


def generate_binary_list(list_size):
    """
    Génère une liste de taille list_size, remplie de 0 et 1 aléatoirement.

    :param list_size: Taille de la liste à générer.
    :type list_size: int
    :return: Liste de 0 et 1
    :rtype: list of int
    """
    return [random.choice([0, 1]) for _ in range(list_size)]


# Exemple d'utilisation :
binary_list = generate_binary_list(150000000)

import time

time_start = time.time()
swaps_num(binary_list)
total_time1 = time.time() - time_start
print(total_time1)

time_start = time.time()
swaps_num2(binary_list)
total_time2 = time.time() - time_start

print(total_time2)

assert swaps_num(binary_list) == swaps_num2(binary_list)

#Test

test_text = """Elle nous rappelle notre lien indissoluble avec la nature et éveille nos sens à l'émerveillement devant sa beauté infinie. 
Lorsque le soleil atteint son zénith, la plage se transforme en un véritable paradis balnéaire. Les rayons ardents du soleil inondent le littoral 
d'une chaleur enveloppante, incitant les baigneurs à plonger dans les eaux cristallines. Les vagues jouent joyeusement avec les nageurs, 
les enveloppant de leur douce caresse salée. Les corps se laissent flotter, portés par les mouvements harmonieux de l'océan, tandis que les 
surfeurs audacieux chevauchent les vagues déferlantes avec une grâce infinie. Les amateurs de sensations fortes trouvent leur bonheur sur la plage, 
où le vent marin alimente les passions des kitesurfeurs et des planchistes qui glissent sur les flots avec aisance et élégance. Leurs figures 
acrobatiques défient les lois de la gravité, émerveillant les spectateurs qui les regardent, fascinés. Le sable chaud sous nos pieds nous rappelle que 
la plage est aussi un terrain de jeux infini. Les matchs endiablés de beach-volley font vibrer le rivage, les joueurs s'élançant dans les airs pour 
effectuer des smashes spectaculaires. Les châteaux de sable, véritables chefs-d'œuvre éphémères, émergent sous les mains habiles des enfants, laissant 
libre cours à leur imagination débordante. La plage est également une véritable source de découverte. Lorsque nous enfilons notre masque et notre tuba, 
nous nous enfonçons dans les profondeurs de l'océan, révélant un monde sous-marin d'une beauté saisissante. Les poissons aux couleurs chatoyantes 
virevoltent parmi les coraux, tandis que les tortues marines majestueuses semblent glisser sans effort à travers les eaux claires. Chaque plongée est 
une aventure, un voyage au cœur d'un écosystème fragile et étonnant. Au-delà des activités ludiques, la plage offre des instants de pure contemplation. 
Se perdre dans les pages d'un livre en écoutant le murmure de l'océan, se laisser hypnotiser par le ballet incessant des vagues qui viennent s'échouer 
sur le rivage, ou simplement se prélasser au soleil en écoutant le chant des mouettes, chaque moment passé sur la plage est une évasion vers un monde 
de quiétude et de sérénité. La plage, véritable carrefour entre l'océan et la terre, est une invitation à la découverte, au partage et à la célébration 
de la beauté de notre environnement naturel. Elle nous rappelle que nous faisons partie intégrante de cette vaste étendue qui nous entoure et que nous 
avons la responsabilité de la préserver pour les générations futures. Alors, que vous soyez un amoureux de la mer, un aventurier en quête de sensations 
fortes ou un contemplatif à la recherche de paix intérieure, laissez-vous envoûter par l'incroyable diversité que la plage a à offrir. Plongez dans ses 
eaux salées, explorez ses recoins cachés et laissez votre cœur et votre esprit s'ouvrir à l'immensité infinie de ce joyau naturel."""

text = pd.Series(test_text)




text = text.str.lower()

text = text.str.replace('[^\w\s]', '', regex=True)

tokens = text.str.split()[0]
voc_vali = list(set(tokens))
#mapping des mots à un index (et inversement) en vue de convertir un mot en un vecteur et vise versa
word2index = {word: index for index, word in enumerate(tokens)}
index2word = {index: word for index, word in enumerate(tokens)}
word2index_u1 = {word: index for index, word in enumerate(voc_vali)}
index2word_u1 = {index : word for index, word in enumerate(voc_vali)}
context_dict = {}
values = list(index2word.values())  # liste des valeurs de index2word
max_index_ori = len(values) - 1  # index maximum

for key in index2word.keys():
    context_length = np.random.randint(3, 10, 1)[0]
    min_index = max(0, key - context_length)
    max_index = min(max_index_ori, key + context_length)
    context_dict[key] = tuple(values[min_index:max_index+1])
    context_dict[key] = tuple(word for word in context_dict[key] if word != index2word[key])


input_matrix = np.zeros((D, len(context_dict)))
embed_etiq_test = {}

c_already_comp = 0
c_not_comp = 0

for words in context_dict.values():
    for word in words:
        if word in voc:
            embed_etiq_test[word] = embed_etiq[word]
            c_already_comp += 1
        else:
            embed_etiq_test[word] = np.random.rand(D, 1) - 0.5
            c_not_comp += 1

    print(c_already_comp, c_not_comp)



input_matrix = np.zeros((D, len(context_dict)))

    # Calcul de la moyenne des embeddings pour chaque contexte
for i, words in enumerate(context_dict.values()):

    matrix = np.zeros((D, len(words)))

    for j, word in enumerate(words):
        matrix[:, j] = embed_etiq_test[word].reshape(-1)


    mean = np.mean(matrix, axis=1)
    input_matrix[:, i] = mean

one_hot_matrix = np.zeros((len(tokens), len(voc_vali)), dtype=int)

for i, word in enumerate(tokens):
    one_hot_matrix[i, word2index_u1[word]] = 1

input_matrix = input_matrix.T


score = []
for b in range(len(tokens)):
    hidden_layer_input = input_matrix[b, :].reshape(1, D)
    hidden_layer_output = np.matmul(hidden_layer_input, W2) + B2
    softmax_output = softmax(hidden_layer_output)
    if np.argsort(softmax_output)[0][-1] < len(voc_vali):
        if index2word_u1[np.argsort(softmax_output)[0][-1]] == tokens[b]:
            score.append(1)
            print(f"Youpi : {tokens[b]}")
        else:
            score.append(0)
            print(f"{index2word_u1[np.argsort(softmax_output)[0][-1]]} alors que le vrai mot est {tokens[b]}")
    else:
        score.append(0)

        embed_etiq['chant']

print(np.mean(np.array(score)))

np.argsort(softmax_output)
softmax_output[0][243]
softmax_output[0].argsort()[-1]
