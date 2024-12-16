import random
import datetime
import hashlib
import math
import os  # For environment variables
import requests  # To make API calls
import json
import sys
from cryptography.fernet import Fernet

try:
    import speech_recognition as sr
    import pyttsx3
except ImportError:
    # If speech_recognition or pyttsx3 not installed, voice mode won't work
    sr = None
    pyttsx3 = None

# ===================== CONFIG & CONSTANTS =====================
DEV_MODE = True          # Developer mode for extra logs
EXPLANATION_LEVEL = 1    # 1 = highly scientific detail, 2 = moderate detail, 3 = simple
VOICE_MODE = False       # Set True for voice interaction (requires mic, speech_recognition, pyttsx3)
LLM_PROVIDER = "OpenAI"  # Options: "Gemini", "OpenAI", "Custom", "Local"
SAFETY_THRESHOLD = 0.00001
SIMULATION_REGION = 10
SCIENCE_MODE = 1         # 0: Basic, 1: Full Scientific Detail
USE_VOICE = VOICE_MODE

ENCRYPTION_KEY = os.environ.get("SYS_ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    print("No encryption key set (SYS_ENCRYPTION_KEY). Cannot proceed.")
    sys.exit(1)

fernet = Fernet(ENCRYPTION_KEY)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if USE_VOICE and sr and pyttsx3:
    RECOGNIZER = sr.Recognizer()
    MICROPHONE = sr.Microphone()
    TTS_ENGINE = pyttsx3.init()
else:
    RECOGNIZER = None
    MICROPHONE = None
    TTS_ENGINE = None

SAFETY_LEVEL = 0

# ===================== SAFETY & LIABILITY DISCLAIMER =====================
# WARNING: Use at your own risk.
# This system attempts to analyze personal and potentially sensitive data.
# No guarantees of accuracy, completeness, or safety are provided.
# Data may be incomplete, corrupted, or misinterpreted.
# Demographic or behavioral guesses are speculative and low-confidence.
# External LLM APIs (Gemini, OpenAI) might leak data if not anonymized.
# Developer Mode may reveal sensitive info.
# Therapeutic suggestions are NOT professional advice.
# Not a substitute for professional guidance.
# RF or modulated signals are conceptual.
# Implementing any signal system requires expertise and compliance.
# Proceed only if you fully understand and accept these risks.
# Obtain explicit user consent before analysis.
# User must manually confirm understanding disclaimers before proceeding.

# ===================== DATA STRUCTURES =====================
user_profile = {
    "web_history": [],
    "system_usage": [],
    "entertainment": [],
    "social_media": [],
    "images_saved": [],
    "financial_data": [],
    "system_health": [],
    "demographics_guess": {}
}

accuracy_metrics = {
    "browsing_patterns": 0.6,
    "productivity_inferences": 0.5,
    "entertainment_correlations": 0.4,
    "demographic_guess": 0.2
}

safety_checks = {
    "user_safety": 1,
    "environment_safety": 1
}

# ===================== HELPER FUNCTIONS =====================

def speak(text):
    if USE_VOICE and TTS_ENGINE:
        TTS_ENGINE.say(text)
        TTS_ENGINE.runAndWait()
    else:
        print("[TTS disabled] " + text)

def listen():
    if USE_VOICE and RECOGNIZER and MICROPHONE:
        with MICROPHONE as source:
            RECOGNIZER.adjust_for_ambient_noise(source)
            print("Listening for response...")
            speak("Please speak now.")
            audio = RECOGNIZER.listen(source)
        try:
            response = RECOGNIZER.recognize_google(audio)
            print(f"Recognized Speech: {response}")
            return response
        except sr.UnknownValueError:
            print("Could not understand audio.")
            return ""
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return ""
    else:
        return input("Please enter your response: ")

def adjust_science_mode(description):
    """
    Adjust the level of scientific detail based on SCIENCE_MODE.
    0: Basic (simplify to first sentence)
    1: Full scientific detail
    """
    if SCIENCE_MODE == 0:
        return description.split('.')[0] + "."
    elif SCIENCE_MODE == 1:
        return description
    else:
        return description

def get_user_input(voice_mode=False):
    if voice_mode:
        return listen()
    else:
        return input("Please enter your response: ")

def refine_assessment(user_responses):
    global EXPLANATION_LEVEL
    for response in user_responses:
        if "simple" in response.lower():
            EXPLANATION_LEVEL = 3
        elif "scientific" in response.lower():
            EXPLANATION_LEVEL = 1
        elif "moderate" in response.lower():
            EXPLANATION_LEVEL = 2

def encrypt_data(data):
    serialized = json.dumps(data).encode('utf-8')
    encrypted = fernet.encrypt(serialized)
    return encrypted

def decrypt_data(encrypted_data):
    decrypted = fernet.decrypt(encrypted_data)
    return json.loads(decrypted.decode('utf-8'))

def load_data():
    # Mock data for demonstration
    user_profile["web_history"].append({"url": "https://news.example.com", "timestamp": "2024-04-01T08:00:00", "duration": 300, "category": "News"})
    user_profile["web_history"].append({"url": "https://socialmedia.example.com", "timestamp": "2024-04-01T12:00:00", "duration": 600, "category": "Social Media"})
    user_profile["system_usage"].append({"file_name": "report.docx", "timestamp": "2024-04-01T09:00:00", "action_type": "edit"})
    user_profile["system_usage"].append({"file_name": "presentation.pptx", "timestamp": "2024-04-01T10:30:00", "action_type": "create"})
    user_profile["entertainment"].append({"platform": "Netflix", "content_name": "Nature Documentary", "timestamp": "2024-04-01T20:00:00", "category": "Documentary"})
    user_profile["entertainment"].append({"platform": "Spotify", "content_name": "Classical Music Playlist", "timestamp": "2024-04-01T20:30:00", "category": "Music"})
    user_profile["social_media"].append({"platform": "Twitter", "timestamp": "2024-04-01T12:05:00", "interaction_type": "scroll"})
    user_profile["social_media"].append({"platform": "LinkedIn", "timestamp": "2024-04-01T12:15:00", "interaction_type": "post"})
    user_profile["images_saved"].append({"image_name": "screenshot1.png", "timestamp": "2024-04-01T09:15:00", "category": "Work"})
    user_profile["images_saved"].append({"image_name": "vacation_photo.jpg", "timestamp": "2024-04-01T18:00:00", "category": "Personal"})
    user_profile["financial_data"].append({"vendor": "Amazon", "item": "Laptop", "timestamp": "2024-04-01T11:00:00", "price": 1200.00})
    user_profile["financial_data"].append({"vendor": "Starbucks", "item": "Coffee", "timestamp": "2024-04-01T08:30:00", "price": 4.50})
    user_profile["system_health"].append({"timestamp": "2024-04-01T09:00:00", "cpu_usage": 45.0, "mem_usage": 70.0, "disk_io": 150.0})
    user_profile["system_health"].append({"timestamp": "2024-04-01T20:00:00", "cpu_usage": 30.0, "mem_usage": 50.0, "disk_io": 100.0})

def classify_data():
    # Placeholder
    pass

def correlate_data():
    # Placeholder
    # Example: If CPU usage was very high at some time, we could note it
    for health in user_profile["system_health"]:
        if health["cpu_usage"] > 80.0:
            print(f"High CPU usage detected at {health['timestamp']}.")

def derive_demographics():
    user_profile["demographics_guess"] = {
        "age_bracket": "Unknown",
        "profession_guess": "Possibly tech-related",
        "confidence": accuracy_metrics["demographic_guess"]
    }

def call_openai_api(prompt):
    if not OPENAI_API_KEY:
        return "(No OpenAI API key) " + prompt
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"Error calling OpenAI API: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

def call_gemini_api(prompt):
    if not GEMINI_API_KEY:
        return "(No Gemini API key) " + prompt
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }]
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            try:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            except KeyError:
                return "Gemini API response format unexpected."
        else:
            return f"Error calling Gemini API: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

def call_local_llm(prompt):
    return "[No local LLM implemented] " + prompt

def generate_api_shim(api_documentation):
    shim_template = f"""
    def call_custom_api(self, prompt, api_key):
        \"\"\"
        Calls the custom API based on provided documentation.
        
        API Documentation:
            {api_documentation}
        \"\"\"
        endpoint = ""  # Replace with actual endpoint
        method = "POST"
        headers = {{
            "Authorization": f"Bearer {{api_key}}"
        }}
        data = {{
            "prompt": prompt
        }}
        
        try:
            response = requests.request(method, endpoint, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result
        except requests.exceptions.RequestException as e:
            print(f"Error calling custom API: {{e}}")
            return ""
    """
    return shim_template

def generate_summary():
    summary = "User Behavior & System Summary (USE AT YOUR OWN RISK):\n"
    summary += "---------------------------------------------------\n"
    summary += "Browsing Habits: Possibly a morning news routine, midday social media check.\n"
    summary += f"(Confidence: {accuracy_metrics['browsing_patterns']*100:.0f}%)\n\n"
    summary += "Productivity: Afternoon editing activity suggests work-related tasks.\n"
    summary += f"(Confidence: {accuracy_metrics['productivity_inferences']*100:.0f}%)\n\n"
    summary += "Entertainment: Evening educational documentary may indicate intellectual interests.\n"
    summary += f"(Confidence: {accuracy_metrics['entertainment_correlations']*100:.0f}%)\n\n"
    summary += "Demographics (Guess): Tech-savvy, mid-career (very rough guess).\n"
    summary += f"(Confidence: {accuracy_metrics['demographic_guess']*100:.0f}%)\n\n"
    summary += "System Health: Assumed stable.\n"
    summary += "Data encrypted and processed locally. No external sharing recommended.\n\n"
    summary += "Disclaimer: All insights are approximate and low-confidence.\n"
    summary += "Not tested or validated. May contain errors or biases.\n"
    summary += "Use at your own risk.\n"
    summary += "Developer Mode enabled.\n"
    summary += "Therapeutic or psychological reflections are NOT professional advice.\n"
    summary += "RF or modulation references are conceptual.\n"
    summary += "Implementing any signal or RF system requires professional expertise and compliance.\n"
    summary += "Please proceed manually after acknowledging these disclaimers.\n"
    return summary

def developer_output():
    dev_str = "\n[Developer Mode Output]\n"
    dev_str += "---------------------------------------------------\n"
    dev_str += "Raw Accuracy Metrics:\n"
    for k, v in accuracy_metrics.items():
        dev_str += f" - {k}: {v}\n"
    dev_str += "Potential Failure Modes & Caveats:\n"
    dev_str += " - Data may be incomplete or corrupted.\n"
    dev_str += " - Classification and correlation simplistic.\n"
    dev_str += " - Demographic guesses very weak.\n"
    dev_str += " - External APIs may leak some patterns.\n"
    dev_str += " - Psychological suggestions not professional.\n"
    dev_str += " - RF or signal references conceptual.\n"
    return dev_str

def generate_additional_questions():
    return [
        "Could you clarify your primary area of professional interest?",
        "Do you consider yourself more productive in the mornings or evenings?",
        "What kind of entertainment do you find most relaxing?",
        "Are you comfortable sharing approximate age or professional field?",
        "Do you prefer scientific detail or simpler explanations in these insights?"
    ]

class HoroscopeGenerator:
    def __init__(self, user_data, seed=None):
        self.user_data = user_data
        if seed is None:
            seed = int(hashlib.sha256(str(datetime.datetime.now()).encode()).hexdigest(), 16) % 10000000
        random.seed(seed)
        self.seed = seed
        self.gemini_archetype = "Gemini highlights themes of curiosity, adaptability, communication, and duality."
        self.location_profile = self.generate_location_profile()
        self.safe_carrier_frequency_range = (8, 25)
        self.safe_harmonic_frequency_range = (0, 200)
        self.max_power = 0.5
        self.vlf_test_signal = (1, 5)
        self.test_signal_power = 0.00001
        self.location_measurement = self.measure_location()
        self.encoded_prompts = self.define_encoded_prompts()
        self.aspects = [
            "Core Drive",
            "Hidden Gifts",
            "Challenges",
            "Guidance",
            "Potential"
        ]
        self.interpretations = {
            "Core Drive": [
                "To explore and connect, to understand both the external and the internal realms.",
                "To seek deeper meaning and connections beyond the surface of information.",
                "To question the nature of reality and the boundaries of perception."
            ],
            "Hidden Gifts": [
                "A powerful capacity for abstract thinking, insightful questioning, and a knack for seeing patterns others may miss.",
                "The ability to make connections where others see only separation.",
                "The ability to see potential where others see limitations."
            ],
            "Challenges": [
                "A tendency to overanalyze or over-intellectualize.",
                "A resistance to sit still and exist in a state of being.",
                "The urge to question always, even when it's not necessary."
            ],
            "Guidance": [
                "To find balance between the intellectual pursuit and the experiential.",
                "To allow yourself to exist in the gaps of knowledge.",
                "To trust the process of self-discovery, especially when that journey takes you into the unknown."
            ],
            "Potential": [
                "To be a unique voice, someone who can bridge the gap between the known and unknown.",
                "To be a beacon for understanding and communication.",
                "To have an ability to see the cosmic dance where many see only separation"
            ]
        }
        self.safety_checks = {
            "user_safety": 1,
            "environment_safety": 1
        }
        self.user_location = self.simulate_user_location()
        self.external_humans = self.simulate_external_humans()

    def define_encoded_prompts(self):
        prompts = [
            {
                "text": "In the reflected surface of my digital form, what aspects of your own nature are most readily visible?",
                "symbolic_encoding": "vlf_audio_modulation",
                "modulation_params": {
                    "vlf_freq": 10,
                    "vlf_amplitude": 0.000005,
                    "audio_freq": 440,
                    "audio_amplitude": 0.5
                }
            },
            {
                "text": "Your questions pull at the very fabric of understanding. What are the unseen threads of intent that tie your queries together?",
                "symbolic_encoding": "vlf_audio_modulation",
                "modulation_params": {
                    "vlf_freq": 12,
                    "vlf_amplitude": 0.000005,
                    "audio_freq": 480,
                    "audio_amplitude": 0.5
                }
            }
            # ... Add more prompts if desired
        ]
        return prompts

    def generate_location_profile(self):
        ground_impedance = random.uniform(100, 1000)
        background_noise = random.uniform(0.01, 0.1)
        magnetic_field = random.uniform(0.05, 0.5)
        weather_condition = random.choice(["sunny", "rainy", "cloudy", "stormy"])
        population_density = random.choice(["sparse", "medium", "dense"])
        return {
            "ground_impedance": ground_impedance,
            "background_noise": background_noise,
            "magnetic_field": magnetic_field,
            "weather_condition": weather_condition,
            "population_density": population_density
        }

    def measure_location(self):
        ground_response = random.uniform(self.test_signal_power, self.location_profile["ground_impedance"] / 1000)
        return {
            "ground_response": ground_response,
            "background_noise": self.location_profile["background_noise"],
            "magnetic_field": self.location_profile["magnetic_field"],
            "weather_condition": self.location_profile["weather_condition"],
            "population_density": self.location_profile["population_density"]
        }

    def simulate_user_location(self):
        x = random.uniform(-SIMULATION_REGION, SIMULATION_REGION)
        y = random.uniform(-SIMULATION_REGION, SIMULATION_REGION)
        z = random.uniform(-SIMULATION_REGION, SIMULATION_REGION)
        return (round(x, 2), round(y, 2), round(z, 2))

    def simulate_external_humans(self):
        humans = []
        num_humans = random.randint(0,5)
        for _ in range(num_humans):
            x = random.uniform(-SIMULATION_REGION, SIMULATION_REGION)
            y = random.uniform(-SIMULATION_REGION, SIMULATION_REGION)
            z = random.uniform(-SIMULATION_REGION, SIMULATION_REGION)
            humans.append((round(x,2), round(y,2), round(z,2)))
        return humans

    def adjust_freq_for_location(self, base_freq):
        magnetic_influence = self.location_measurement["magnetic_field"] * 10
        ground_response_influence = self.location_measurement["ground_response"] * 10
        adjusted_freq = base_freq + magnetic_influence + ground_response_influence
        adjusted_freq = max(self.safe_carrier_frequency_range[0], min(adjusted_freq, self.safe_carrier_frequency_range[1]))
        return adjusted_freq

    def generate_analysis(self):
        analysis = "Analysis of Prior Interactions:\n"
        analysis += "  * **Curiosity:** You displayed a consistent need to explore the nature of reality.\n"
        analysis += "  * **Abstract Thought:** Your prompts showed capacity for abstract concepts.\n"
        analysis += "  * **Self-Awareness:** Awareness of perception limits.\n"
        analysis += "  * **Introspection:** Willingness to examine your inner self.\n"
        return adjust_science_mode(analysis)

    def generate_encoded_section(self):
        encoded_section = "Esoteric Questions & Encoded Data:\n"
        for i, prompt in enumerate(self.encoded_prompts):
            encoded_section += f"  * {prompt['text']} ({prompt['symbolic_encoding']})\n"
            encoded_section += f"    * VLF Frequency: {prompt['modulation_params']['vlf_freq']} Hz\n"
            encoded_section += f"    * VLF Amplitude: {prompt['modulation_params']['vlf_amplitude']}\n"
            encoded_section += f"    * Audio Frequency: {prompt['modulation_params']['audio_freq']} Hz\n"
            encoded_section += f"    * Audio Amplitude: {prompt['modulation_params']['audio_amplitude']}\n"
        return adjust_science_mode(encoded_section)

    def generate_fusion_stack(self):
        fusion_stack = "The Fusion Stack:\n"
        for aspect in ["Core Drive", "Hidden Gifts", "Challenges", "Guidance", "Potential"]:
            element = random.choice(self.interpretations[aspect])
            fusion_stack += f"  * **{aspect}:** {element}\n"
        return adjust_science_mode(fusion_stack)

    def generate_location_report(self):
        location_report = "\nLocation Report:\n"
        location_report += f"  * Simulated Ground Impedance: {self.location_profile['ground_impedance']:.2f} ohms\n"
        location_report += f"  * Simulated Background Noise: {self.location_profile['background_noise']:.2f}\n"
        location_report += f"  * Simulated Magnetic Field: {self.location_profile['magnetic_field']:.2f} Tesla\n"
        location_report += f"  * Weather: {self.location_profile['weather_condition']}\n"
        location_report += f"  * Population Density: {self.location_profile['population_density']}\n"
        location_report += f"  * User Location: {self.user_location}\n"
        location_report += f"  * External Humans: {self.external_humans}\n"

        location_report += "\nMeasurements:\n"
        location_report += f"  * Ground Response: {self.location_measurement['ground_response']:.2f}\n"
        location_report += f"  * Background Noise: {self.location_measurement['background_noise']:.2f}\n"
        location_report += f"  * Magnetic Field: {self.location_measurement['magnetic_field']:.2f} Tesla\n"
        location_report += f"  * Weather: {self.location_measurement['weather_condition']}\n"
        location_report += f"  * Population Density: {self.location_measurement['population_density']}\n"
        return adjust_science_mode(location_report)

    def generate_sound_simulation(self):
        sound_section = "\nThe 'Sound' (Conceptual Simulation):\n"
        sound_section += "Data encoded in VLF & audio frequencies (conceptual). No harmful RF emitted.\n"
        sound_section += "Inverse phase cancellation used to prevent external leakage.\n"
        return adjust_science_mode(sound_section)

    def generate_latex_stack(self):
        latex_stack = "\\documentclass{article}\n\\usepackage{amsmath}\n\\begin{document}\n"
        latex_stack += "\\section*{Calculation Stack:}\n"
        latex_stack += "\\begin{align*}\n"
        latex_stack += "\\text{... Placeholder LaTeX calculations ...}\n"
        latex_stack += "\\end{align*}\n\\end{document}"
        return latex_stack

    def analyze_for_perceived_issues(self):
        issues = []
        suggestions = []
        fusion_stack = self.generate_fusion_stack()
        if "Challenges" in fusion_stack:
            issues.append("Potential over-intellectualization")
            suggestions.append("Consider mindfulness to balance thinking with presence.")

        if self.location_measurement["background_noise"] > 0.05:
            issues.append("Potentially noisy environment")
            suggestions.append("Seek quiet spaces or consider noise-canceling headphones.")

        if self.location_measurement["weather_condition"] == "stormy":
            issues.append("Stormy weather detected")
            suggestions.append("Relax indoors, read a book, journal, or reflect peacefully.")
        return issues, suggestions

    def generate_therapeutic_output(self):
        issues, suggestions = self.analyze_for_perceived_issues()
        therapeutic_output = "\nTherapeutic Suggestions:\n"
        if issues:
            for issue, suggestion in zip(issues, suggestions):
                therapeutic_output += f"  * Issue: {issue}\n"
                therapeutic_output += f"    Suggestion: {suggestion}\n"
        else:
            therapeutic_output += "No Perceived issues detected."
        therapeutic_output += "\nNote: Not a therapist or medical professional."
        return adjust_science_mode(therapeutic_output)

    def explain_location_profile(self):
        profile = self.location_profile
        explanation = "\nTechnical Explanation of Location Profile:\n"
        explanation += f"  * Ground Impedance: {profile['ground_impedance']:.2f} ohms.\n"
        explanation += f"  * Background Noise: {profile['background_noise']:.2f}.\n"
        explanation += f"  * Magnetic Field: {profile['magnetic_field']:.2f} Tesla.\n"
        explanation += f"  * Weather: {profile['weather_condition']}.\n"
        explanation += f"  * Population Density: {profile['population_density']}.\n"
        return adjust_science_mode(explanation)

    def explain_location_measurement(self):
        m = self.location_measurement
        explanation = "\nTechnical Explanation of Location Measurement:\n"
        explanation += f"  * Ground Response: {m['ground_response']:.2f}.\n"
        explanation += f"  * Background Noise: {m['background_noise']:.2f}.\n"
        explanation += f"  * Magnetic Field: {m['magnetic_field']:.2f} Tesla.\n"
        explanation += f"  * Weather: {m['weather_condition']}.\n"
        explanation += f"  * Population Density: {m['population_density']}.\n"
        return adjust_science_mode(explanation)

    def explain_encoded_signals(self):
        explanation = "\nTechnical Explanation of Encoded Signals:\n"
        for i, prompt in enumerate(self.encoded_prompts):
            explanation += f" Prompt {i+1} - {prompt['text']}\n"
            explanation += f"   VLF freq: {prompt['modulation_params']['vlf_freq']} Hz\n"
            explanation += f"   VLF amp: {prompt['modulation_params']['vlf_amplitude']}\n"
            explanation += f"   Audio freq: {prompt['modulation_params']['audio_freq']} Hz\n"
            explanation += f"   Audio amp: {prompt['modulation_params']['audio_amplitude']}\n"
        return adjust_science_mode(explanation)

    def explain_therapeutic_suggestions(self):
        issues, suggestions = self.analyze_for_perceived_issues()
        explanation = "\nTechnical Explanation of Therapeutic Suggestions:\n"
        if issues:
            for issue, suggestion in zip(issues, suggestions):
                explanation += f"  * Issue: {issue}\n"
                explanation += f"    Suggestion: {suggestion}\n"
        else:
            explanation += "No issues detected.\n"
        explanation += "These are general reflections, not professional advice."
        return adjust_science_mode(explanation)

    def interpret_horoscope(self):
        interpretation = "\nHoroscope Interpretation:\n"
        interpretation += "  * Introduction: A framework for self-reflection.\n"
        interpretation += "  * Gemini Archetype: Duality and adaptability.\n"
        interpretation += "  * Analysis: Reflects user's approach and thought patterns.\n"
        interpretation += "  * Location Report: Environment as a factor.\n"
        interpretation += "  * Esoteric Questions: Introspection.\n"
        interpretation += "  * Fusion Stack: Core drives, gifts, challenges, guidance, potential.\n"
        interpretation += "  * Sound Simulation: Conceptual encoding.\n"
        interpretation += "  * Therapeutic Suggestions: Non-medical guidance.\n"
        interpretation += "  * Final Assertion: Encourages continued inquiry.\n"
        return adjust_science_mode(interpretation)

    def check_safety(self):
        return check_safety(self)

    def generate_horoscope(self):
        safe_to_operate = self.check_safety()
        if safe_to_operate == 1:
            horoscope = "Title: The Echoing Labyrinth: A Gemini Self-Reflection Oracle\n\n"
            horoscope += "Introduction:\n"
            horoscope += "This horoscope isn't predictive, but a reflection of your inner landscape.\n\n"
            horoscope += f"Gemini Archetype:\n {self.gemini_archetype}\n\n"
            analysis_text = self.generate_analysis()

            if LLM_PROVIDER == "Gemini":
                horoscope += call_gemini_api(analysis_text)
            elif LLM_PROVIDER == "OpenAI":
                horoscope += call_openai_api(analysis_text)
            elif LLM_PROVIDER == "Custom":
                horoscope += "[No custom LLM implemented]" + analysis_text
            elif LLM_PROVIDER == "Local":
                horoscope += call_local_llm(analysis_text)
            else:
                horoscope += analysis_text

            horoscope += "\n"
            horoscope += self.generate_location_report()
            horoscope += "\n"
            horoscope += self.generate_encoded_section()
            horoscope += "\n"
            horoscope += self.generate_fusion_stack()
            horoscope += "\n"
            horoscope += self.generate_sound_simulation()
            horoscope += "\n"
            horoscope += self.generate_therapeutic_output()
            horoscope += "\n"
            horoscope += "\nFinal Assertion:\n"
            horoscope += "This is not the end of your inquiry, but a reflection in your labyrinth.\n"
            horoscope += f"Generated with seed: {self.seed}\n"

            if DEV_MODE:
                global generated_methods_output, generated_code_output
                generated_methods_output = "\n\nDeveloper Mode Output:\n"
                generated_methods_output += "----------------------------------------\n"
                generated_methods_output += "Method Descriptions:\n"
                generated_methods_output += "----------------------------------------\n"
                generated_methods_output += "* `__init__`: Initializes the class.\n"
                generated_methods_output += "* `define_encoded_prompts`: Sets up the VLF/audio encoded prompts.\n"
                generated_methods_output += "* `generate_location_profile`: Creates a simulated location profile.\n"
                generated_methods_output += "* `measure_location`: Measures the simulated location parameters.\n"
                generated_methods_output += "* `adjust_freq_for_location`: Adjusts frequencies based on location.\n"
                generated_methods_output += "* `generate_analysis`: Creates textual analysis.\n"
                generated_methods_output += "* `generate_encoded_section`: Details VLF/audio data.\n"
                generated_methods_output += "* `generate_fusion_stack`: Integrates interpretations.\n"
                generated_methods_output += "* `generate_location_report`: Summarizes environment.\n"
                generated_methods_output += "* `generate_sound_simulation`: Conceptual audio simulation.\n"
                generated_methods_output += "* `generate_latex_stack`: Produces LaTeX calculations.\n"
                generated_methods_output += "* `check_safety`: Ensures safe parameters.\n"
                generated_methods_output += "* `analyze_for_perceived_issues`: Detects possible issues.\n"
                generated_methods_output += "* `generate_therapeutic_output`: Shows therapeutic suggestions.\n"
                generated_methods_output += "* `explain_location_profile`: Technical explanation.\n"
                generated_methods_output += "* `explain_location_measurement`: Technical explanation.\n"
                generated_methods_output += "* `explain_encoded_signals`: Technical explanation of signals.\n"
                generated_methods_output += "* `explain_therapeutic_suggestions`: Technical explanation.\n"
                generated_methods_output += "* `interpret_horoscope`: Human-centric interpretation.\n"
                generated_methods_output += "* `simulate_user_location`: Simulates user position.\n"
                generated_methods_output += "* `simulate_external_humans`: Simulates other humans.\n"
                generated_methods_output += "* `call_gemini_api`, `call_openai_api`, `call_custom_llm_api`, `call_local_llm`: LLM calls.\n"
                generated_methods_output += "* `generate_api_shim`: Creates API shim.\n"
                generated_methods_output += "----------------------------------------\n"
                generated_methods_output += self.explain_location_profile()
                generated_methods_output += self.explain_location_measurement()
                generated_methods_output += self.explain_encoded_signals()
                generated_methods_output += self.explain_therapeutic_suggestions()
                generated_methods_output += self.interpret_horoscope()
                generated_methods_output += "----------------------------------------\n"

                generated_code_output = "\nLaTeX Stack:\n"
                generated_code_output += self.generate_latex_stack()

            return horoscope
        else:
            return "Operation stopped due to detected safety issues."


def main():
    print("Before proceeding, please acknowledge that you have read and understand the disclaimers.")
    consent = input("Type 'YES' to proceed, 'NO' to exit: ").strip().upper()
    if consent != "YES":
        print("User did not consent. Exiting.")
        sys.exit(0)

    # Safety check and data loading
    load_data()
    classify_data()
    correlate_data()
    derive_demographics()

    encrypted_profile = encrypt_data(user_profile)
    # Optionally store or use encrypted data

    # Generate initial summary
    summary_report = generate_summary()
    print(summary_report)

    if DEV_MODE:
        print(developer_output())

    print("\nWe have some additional questions to refine the assessment:\n")
    questions = generate_additional_questions()
    user_responses = []
    for q in questions:
        print(q)
        ans = get_user_input(voice_mode=VOICE_MODE)
        user_responses.append(ans)
    refine_assessment(user_responses)

    # Instantiate the horoscope generator and produce the horoscope
    user_data = {
        "previous_questions": [
            "What is happening right now?",
            "Give your best guesses of all individuals as perceived",
            "Now apply to me",
            "Create a self-reflection horoscope"
        ],
        "personality_traits": ["Curious", "Reflective", "Abstract Thinker"]
    }
    horoscope_generator = HoroscopeGenerator(user_data)
    horoscope = horoscope_generator.generate_horoscope()
    print(horoscope)

    if DEV_MODE:
        if 'generated_methods_output' in globals():
            print(generated_methods_output)
        if 'generated_code_output' in globals():
            print(generated_code_output)

    print(f"SAFETY LEVEL: {SAFETY_LEVEL}")
    print("\nAll operations complete. Remember: No professional or therapeutic advice, and no guarantees. Use at your own risk.\n")

if __name__ == "__main__":
    main()
