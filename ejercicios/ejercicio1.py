# Ejercicio de pre inteligencia artificial

import random
import time

def temperatura_aleatoria():
    return random.randint(10, 35)

def simular_temperatura():
    cont = 0 
    while cont < 5:
        temp = temperatura_aleatoria()
        if temp < 15:
            comando = "Activar extractor de aire."
        elif temp > 30:
            comando = "Encender la climatización interna."
        else:
            comando = "Sistema en estado optimo."

        print(f"Temperatura actual: {temp} °C | Comando: {comando}")
        
        time.sleep(2)
        cont += 1

simular_temperatura()