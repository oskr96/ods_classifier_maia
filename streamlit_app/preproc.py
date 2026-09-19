import re
import nltk
import pandas as pd    

# Función utilizada para limpiar cada texto
def limpiar_texto(texto):
    texto = texto.lower()

    # Eliminar direcciones web
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)

    # Eliminar referencias directas al número de un ODS
    texto = re.sub(r"ods\s*\d+", " ", texto)
    texto = re.sub(
        r"objetivo(?:s)? de desarrollo sostenible\s*\d+",
        " ",
        texto,
    )
    texto = re.sub(r"objetivo\s+\d+", " ", texto)

    return texto

# Funcion que retorna el nombre del ods correspondiente a un label
def get_ods_name(label):
    #nombres ODS
    nombres_ods = {
        1: "Fin de la pobreza",
        2: "Hambre cero",
        3: "Salud y bienestar",
        4: "Educación de calidad",
        5: "Igualdad de género",
        6: "Agua limpia y saneamiento",
        7: "Energía asequible y no contaminante",
        8: "Trabajo decente y crecimiento económico",
        9: "Industria, innovación e infraestructura",
        10: "Reducción de las desigualdades",
        11: "Ciudades y comunidades sostenibles",
        12: "Producción y consumo responsables",
        13: "Acción por el clima",
        14: "Vida submarina",
        15: "Vida de ecosistemas terrestres",
        16: "Paz, justicia e instituciones sólidas",
        17: "Alianzas para lograr los objetivos",
    }
    return nombres_ods[label]

#funcion que retorna un df con los nombres de los ODS y su etiqueta
def get_ods_consd():
    #nombres ODS
    nombres_ods = {
        1: "Fin de la pobreza",
        2: "Hambre cero",
        3: "Salud y bienestar",
        4: "Educación de calidad",
        5: "Igualdad de género",
        6: "Agua limpia y saneamiento",
        7: "Energía asequible y no contaminante",
        8: "Trabajo decente y crecimiento económico",
        9: "Industria, innovación e infraestructura",
        10: "Reducción de las desigualdades",
        11: "Ciudades y comunidades sostenibles",
        12: "Producción y consumo responsables",
        13: "Acción por el clima",
        14: "Vida submarina",
        15: "Vida de ecosistemas terrestres",
        16: "Paz, justicia e instituciones sólidas",
        17: "Alianzas para lograr los objetivos",
    }
    return pd.DataFrame(list(nombres_ods.items()), columns=["label", "name"])
	