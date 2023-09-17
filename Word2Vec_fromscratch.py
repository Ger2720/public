# Attempt to recreate the word2vec model using only pandas, numpy, and a few other basic packages
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
values = list(index2word.values())  # liste des valeurs de index2word
max_index_ori = len(values) - 1  # index maximum

for key in index2word.keys():
    # On définit des indices min et max pour la slice
    min_index = max(0, key - 3)
    max_index = min(max_index_ori, key + 3)
    # On récupère les mots du contexte en utilisant une slice
    context_dict[key] = tuple(values[min_index:max_index+1])
    context_dict[key] = tuple(word for word in context_dict[key] if word != index2word[key])


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

            mean = np.mean(matrix, axis=1)
            input[:, i] = mean

    return input.T

def softmax(x):
    if len(x.shape) == 1:
        e_x = np.exp(x - np.max(x)) # Soustraire le max pour des raisons de stabilité numérique
        output = e_x / np.sum(e_x)
    else:
        e_x = np.exp(x - np.max(x, axis=1, keepdims= True))
        output = e_x / np.sum(e_x, axis=1, keepdims=True)
    return output

def ADAM(i, gradient, m_i, v_i):
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
W1 = np.random.rand(D, len(voc)) - 0.5
embed_etiq = {word : W1[:, i] for i, word in enumerate(voc)}
eps = 10**(-8)
i = 0
m_i_w = 0
v_i_w = 0
m_i_b = 0
v_i_b = 0
nb_iterations = 30
W2 = np.random.rand(D, len(voc)) - 0.5
B2 = np.random.randn(W2.shape[1]) - 0.5
alpha = 0.005
batch_size = int(len(context_dict)/10)
losses_graph = []
adam_context = {i : np.zeros(2) for i, word in enumerate(tokens)}

for j in tqdm.tqdm(range(nb_iterations)):

    gradient_accumulator = {word: np.zeros(D) for word in voc}
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
            gradient_accumulator[word] = alpha * m_i_hat_e / (np.sqrt(v_i_hat_e) + eps)
        # Mettre à jour les embeddings pour chaque mot dans le contexte
        for word in context_dict[context_index]:
            embed_etiq[word] -= gradient_accumulator[word]
        #Stocker les valeurs de l'optimiseur ADAM pour embeddings
        adam_context[context_index] = [m_i_e, v_i_e]

        i += 1

    hidden_layer_input_total = input(embed_etiq, context_dict, D)
    hidden_layer_output_total = np.matmul(hidden_layer_input_total, W2) + B2
    softmax_output_total = softmax(hidden_layer_output_total)
    losses = -np.sum(one_hot_matrix * np.log(softmax_output_total))
    losses_graph.append(losses)

    if (j + 1) % 200 == 0:
        print(f"la perte du modèle est de {losses}")

plt.figure(figsize=(10, 5))
plt.plot(losses_graph)
plt.xlabel('iterations')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
plt.show()



def testsurtrain(tokens, embed_etiq, context_dict, D, W2, B2, index2word_u):
    score = []
    for i in tqdm.tqdm(range(30)):
        hidden_layer_input = input(embed_etiq, context_dict, D)
        hidden_layer_output = np.matmul(hidden_layer_input, W2) + B2
        softmax_output = softmax(hidden_layer_output)
        if index2word_u[np.argsort(softmax_output[i])[-1]] == tokens[i]:
            score.append(1)
        else:
            score.append(0)
    return print(np.mean(np.array(score)))

testsurtrain(tokens, embed_etiq, context_dict, D, W2, B2, index2word_u)

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
    # On définit des indices min et max pour la slice
    min_index = max(0, key - 3)
    max_index = min(max_index_ori, key + 3)
    # On récupère les mots du contexte en utilisant une slice
    context_dict[key] = tuple(values[min_index:max_index+1])
    context_dict[key] = tuple(word for word in context_dict[key] if word != index2word[key])


embed_etiq_test = {}
input_matrix = np.zeros((D, len(context_dict)))

embed_etiq_test = {}
i = 0
for words in context_dict.values():
    for word in words:
        if word in voc :
            embed_etiq_test[word] = embed_etiq[word]
            i += 1
        else:
            embed_etiq_test[word] = np.random.rand(D, 1) - 0.5
            i += 1



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
