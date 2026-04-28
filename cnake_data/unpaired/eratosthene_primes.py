def primalite_eratosthene(limite):
    """Naive odd-only sieve using repeated filtering."""
    if limite < 2:
        return []

    liste = list(range(3, limite, 2))
    listepremier = [2]

    while len(liste) > 1:
        compteur = liste[0]
        listepremier.append(compteur)
        liste_tmp = []
        for i in range(1, len(liste)):
            if liste[i] % compteur != 0:
                liste_tmp.append(liste[i])
        liste = liste_tmp

    if len(liste) == 1 and liste[0] not in listepremier:
        listepremier.append(liste[0])

    return listepremier
