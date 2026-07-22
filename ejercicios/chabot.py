"""
PROYECTO SOCIAL: Chatbot de Apoyo en Salud Mental
Implementación de un asistente conversacional para apoyo emocional
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import random
import re

class MentalHealthChatbot:
    """
    Chatbot educativo para apoyo en salud mental
    Diseñado para proporcionar información y recursos de bienestar emocional
    """
    
    def __init__(self):
        self.intents = self._load_intents()
        self.responses = self._load_responses()
        self.user_context = defaultdict(dict)
        self.conversation_history = []
        
    def _load_intents(self):
        """Carga los patrones de intención"""
        return {
            'saludo': {
                'patterns': ['hola', 'buenas', 'hey', 'hi', 'hello', 'qué tal'],
                'keywords': ['saludo', 'hola', 'buenos días']
            },
            'estres': {
                'patterns': ['estrés', 'estresado', 'ansiedad', 'preocupado', 
                           'nervioso', 'tenso', 'abrumado'],
                'keywords': ['estrés', 'ansiedad', 'preocupación']
            },
            'tristeza': {
                'patterns': ['triste', 'deprimido', 'desanimado', 'melancólico',
                           'sin ganas', 'vacío', 'llorar'],
                'keywords': ['tristeza', 'depresión', 'triste']
            },
            'ayuda': {
                'patterns': ['ayuda', 'necesito ayuda', 'apoyo', 'consejo',
                           'qué hago', 'sugerencia'],
                'keywords': ['ayuda', 'apoyo', 'consejo']
            },
            'autocuidado': {
                'patterns': ['cuidado personal', 'autocuidado', 'relajación',
                           'meditar', 'respirar', 'mindfulness'],
                'keywords': ['autocuidado', 'relajación', 'meditación']
            },
            'despedida': {
                'patterns': ['adiós', 'chao', 'hasta luego', 'bye', 'nos vemos',
                           'gracias por todo'],
                'keywords': ['despedida', 'adiós', 'gracias']
            }
        }
    
    def _load_responses(self):
        """Carga las respuestas predefinidas"""
        return {
            'saludo': [
                "¡Hola! ¿Cómo te sientes hoy? Estoy aquí para escucharte.",
                "Hola, ¿cómo estás? Cuéntame cómo te sientes.",
                "¡Qué gusto verte! ¿Qué te gustaría compartir hoy?"
            ],
            'estres': [
                "Lamento que te sientas estresado. Recuerda que está bien tomarse un momento para respirar. ¿Te gustaría practicar una técnica de relajación?",
                "El estrés es una respuesta natural. Podemos explorar qué está causando tu estrés y cómo manejarlo juntos.",
                "Sé que el estrés puede ser abrumador. ¿Qué tal si practicamos algunos ejercicios de respiración?"
            ],
            'tristeza': [
                "Entiendo que te sientas triste. Es importante permitirte sentir esas emociones. ¿Qué te gustaría hacer para cuidarte hoy?",
                "La tristeza es una emoción válida. ¿Hay algo específico que te tenga preocupado?",
                "A veces, cuando nos sentimos tristes, hablar sobre ello puede ayudar. Estoy aquí para escucharte."
            ],
            'ayuda': [
                "Estoy aquí para apoyarte. ¿Puedes contarme más sobre lo que te preocupa?",
                "Reconocer que necesitas ayuda es un gran paso. ¿En qué área te gustaría que te apoye?",
                "Todos necesitamos ayuda en algún momento. Cuéntame cómo puedo ayudarte."
            ],
            'autocuidado': [
                "El autocuidado es fundamental. Te sugiero practicar la respiración profunda: inhala por 4 segundos, mantén por 4, exhala por 4.",
                "La meditación puede ayudar mucho. ¿Te gustaría probar una meditación guiada de 5 minutos?",
                "Recuerda que cuidarte no es egoísta, es necesario. ¿Qué actividad te ayuda a relajarte?"
            ],
            'despedida': [
                "Gracias por compartir tus sentimientos conmigo. Recuerda que siempre estaré aquí para ti.",
                "Ha sido un gusto conversar contigo. Cuídate mucho y recuerda que mereces ser feliz.",
                "Si necesitas apoyo en cualquier momento, aquí estoy. ¡Que tengas un buen día!"
            ],
            'default': [
                "¿Puedes contarme más sobre eso? Quiero entenderte mejor.",
                "Gracias por compartir eso. ¿Cómo te hace sentir eso exactamente?",
                "Eso es interesante. ¿Hay algo más que quieras compartir al respecto?"
            ]
        }
    
    def _classify_intent(self, text):
        """Clasifica la intención del mensaje del usuario"""
        text_lower = text.lower()
        
        # Verificar cada intención
        for intent_name, intent_data in self.intents.items():
            for pattern in intent_data['patterns']:
                if pattern in text_lower:
                    return intent_name
        
        # Verificar con keywords
        for intent_name, intent_data in self.intents.items():
            for keyword in intent_data.get('keywords', []):
                if keyword in text_lower:
                    return intent_name
        
        return 'default'
    
    def _get_response(self, intent):
        """Obtiene una respuesta basada en la intención"""
        responses = self.responses.get(intent, self.responses['default'])
        return random.choice(responses)
    
    def _analyze_sentiment(self, text):
        """Análisis básico de sentimiento"""
        positive_words = ['feliz', 'bien', 'mejor', 'contento', 'alegre', 
                         'excelente', 'genial', 'maravilloso']
        negative_words = ['triste', 'mal', 'preocupado', 'estresado', 
                         'ansioso', 'deprimido', 'cansado']
        
        text_lower = text.lower()
        score = 0
        
        for word in positive_words:
            if word in text_lower:
                score += 1
        
        for word in negative_words:
            if word in text_lower:
                score -= 1
        
        if score > 0:
            return 'positivo'
        elif score < 0:
            return 'negativo'
        else:
            return 'neutral'
    
    def _extract_keywords(self, text):
        """Extrae palabras clave del mensaje"""
        words = re.findall(r'\b[a-záéíóúñ]+\b', text.lower())
        return words
    
    def _generate_empathic_response(self, text, intent):
        """Genera una respuesta empática personalizada"""
        sentiment = self._analyze_sentiment(text)
        keywords = self._extract_keywords(text)
        
        base_response = self._get_response(intent)
        
        # Personalizar según sentimiento
        if sentiment == 'negativo':
            base_response += " Lamento que te sientas así. ¿Quieres hablar más sobre ello?"
        elif sentiment == 'positivo':
            base_response += " Me alegra que te sientas así. ¿Cómo podemos mantener ese estado?"
        
        # Agregar preguntas específicas
        if 'estrés' in keywords or 'ansiedad' in keywords:
            base_response += " ¿Qué técnicas de relajación has probado?"
        elif 'triste' in keywords or 'deprimido' in keywords:
            base_response += " ¿Hay algo que normalmente te ayude cuando te sientes así?"
        elif 'meditar' in keywords or 'relajación' in keywords:
            base_response += " ¡Qué bien! ¿Te gustaría compartir tu experiencia?"
        
        return base_response
    
    def _provide_resources(self, intent):
        """Proporciona recursos según la intención"""
        resources = {
            'estres': [
                "Técnica de respiración 4-7-8: Inhala 4s, mantén 7s, exhala 8s",
                "Aplicación de meditación: Headspace, Calm",
                "Ejercicio de grounding: 5-4-3-2-1 (cosas que ves, tocas, oyes, hueles, saboreas)"
            ],
            'tristeza': [
                "Línea de apoyo emocional: 800-123-4567",
                "Técnica de escritura: Escribe 3 cosas por las que estás agradecido",
                "Actividad: Salir a caminar 15 minutos"
            ],
            'autocuidado': [
                "Rutina de autocuidado: 10 minutos de meditación diaria",
                "Ejercicio: Estiramientos suaves",
                "Hábito: Beber suficiente agua durante el día"
            ]
        }
        
        return resources.get(intent, [])
    
    def chat(self):
        """Inicia una conversación con el chatbot"""
        print("\n" + "="*50)
        print("🧠 CHATBOT DE APOYO EMOCIONAL")
        print("="*50)
        print("\nHola, soy tu asistente de apoyo emocional.")
        print("Estoy aquí para escucharte y ofrecerte apoyo.")
        print("Puedes hablar sobre cualquier cosa que te preocupe.")
        print("Escribe 'salir' en cualquier momento para terminar.\n")
        
        while True:
            user_input = input("Tú: ").strip()
            
            if user_input.lower() in ['salir', 'bye', 'chao']:
                print("\nBot: Me despido con un abrazo virtual. Recuerda que siempre puedes hablar conmigo. ¡Cuídate mucho! 🫂")
                break
            
            if not user_input:
                print("Bot: ¿Hay algo en lo que pueda ayudarte?")
                continue
            
            # Registrar conversación
            self.conversation_history.append(('usuario', user_input))
            
            # Clasificar intención
            intent = self._classify_intent(user_input)
            
            # Generar respuesta
            response = self._generate_empathic_response(user_input, intent)
            
            # Agregar recursos si son relevantes
            resources = self._provide_resources(intent)
            if resources and len(self.conversation_history) % 3 == 0:
                response += "\n\n📚 Recursos que podrían ayudarte:\n"
                for resource in resources[:2]:
                    response += f"• {resource}\n"
            
            print(f"\nBot: {response}\n")
            
            # Registrar respuesta
            self.conversation_history.append(('bot', response))
            
            # Análisis simple de la conversación
            if len(self.conversation_history) > 10:
                self._analyze_conversation()
    
    def _analyze_conversation(self):
        """Analiza la conversación y ofrece recomendaciones"""
        user_messages = [msg for role, msg in self.conversation_history if role == 'usuario']
        
        # Analizar sentimiento general
        sentiments = [self._analyze_sentiment(msg) for msg in user_messages[-5:]]
        neg_count = sentiments.count('negativo')
        
        if neg_count >= 3:
            print("\n💡 He notado que has estado compartiendo sentimientos difíciles.")
            print("   Te sugiero que consideres hablar con un profesional de salud mental.")
            print("   Tu bienestar emocional es importante y buscar ayuda es un signo de fortaleza.\n")
    
    def get_conversation_summary(self):
        """Genera un resumen de la conversación"""
        if not self.conversation_history:
            return "No hay conversación para resumir."
        
        user_messages = [msg for role, msg in self.conversation_history if role == 'usuario']
        bot_messages = [msg for role, msg in self.conversation_history if role == 'bot']
        
        summary = f"""
        📊 RESUMEN DE LA CONVERSACIÓN
        --------------------------------
        Total de intercambios: {len(self.conversation_history)}
        Mensajes del usuario: {len(user_messages)}
        Mensajes del bot: {len(bot_messages)}
        
        Temas principales identificados:
        """
        
        # Identificar temas principales
        all_text = ' '.join(user_messages)
        intents_found = []
        for intent, data in self.intents.items():
            for keyword in data.get('keywords', []):
                if keyword in all_text:
                    intents_found.append(intent)
                    break
        
        if intents_found:
            for intent in set(intents_found):
                summary += f"\n• {intent.capitalize()}"
        else:
            summary += "\n• No se identificaron temas específicos"
        
        return summary

# Ejecutar chatbot
def run_mental_health_chatbot():
    """Ejecuta el chatbot de salud mental"""
    chatbot = MentalHealthChatbot()
    chatbot.chat()
    
    # Mostrar resumen
    print("\n" + "="*50)
    print(chatbot.get_conversation_summary())
    print("="*50)

# Ejecutar el chatbot
print("=== CHATBOT DE APOYO EMOCIONAL ===")
print("Este es un chatbot educativo para proporcionar apoyo emocional.")
print("Si estás experimentando pensamientos de autolesión o suicidio,")
print("por favor contacta a una línea de ayuda inmediatamente.\n")

run_mental_health_chatbot()