import os
import sys
import threading
import time
import json
import queue
import tempfile
import wave
import numpy as np
import requests
import keyboard
import pyperclip
import pyaudio
import tkinter as tk
import tkinter.font as tkFont
from tkinter import ttk
from PIL import Image, ImageDraw
import pystray
from pynput.keyboard import Controller, Key
from dotenv import load_dotenv, set_key
import win32gui
import win32con
import win32api
import threading
import io
from pydub import AudioSegment
from openai import OpenAI

def configure_message_timeout(self, message_type, milliseconds):
        """Configure the timeout duration for a specific message type
        
        Args:
            message_type (str): The type of message
            milliseconds (int): The timeout duration in milliseconds
        """
        if message_type in self.MESSAGE_TIMEOUTS:
            self.MESSAGE_TIMEOUTS[message_type] = milliseconds
            print(f"Message timeout for '{message_type}' set to {milliseconds}ms")
            return True
        else:
            print(f"Unknown message type: {message_type}")
            return False

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
GROQ_WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo")  # Keep old env var name for backward compatibility
OPENAI_WHISPER_MODEL = os.getenv("OPENAI_WHISPER_MODEL", "whisper-1")
SELECTED_MICROPHONE_INDEX = int(os.getenv("SELECTED_MICROPHONE_INDEX", "0"))
TRANSCRIPTION_SERVICE = os.getenv("TRANSCRIPTION_SERVICE", "groq")  # Default to groq
TTS_MODEL = os.getenv("TTS_MODEL", "playai-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "Fritz-PlayAI")
TTS_RESPONSE_FORMAT = "wav"

# Define hotkeys
DICTATION_HOTKEY = 'f9'  # Hold to dictate
ASSISTANT_HOTKEY = 'f12'  # Hold for assistant (Prompt)
PARAPHRASE_HOTKEY = 'f8'  # Paraphrase selected text
LANGUAGE_TOGGLE_HOTKEY = 'ctrl+alt+l'  # Toggle between English and Spanish
TEXT_TO_SPEECH_HOTKEY = 'ctrl+alt+s'  # New hotkey for TTS
TEXT_PROMPT_HOTKEY = 'f10'  # New hotkey for text prompt entry

# Audio recording settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 400
SILENCE_DURATION = 1.5
SPEECH_DETECTED_THRESHOLD = 600

# Paraphrasing styles
PARAPHRASING_STYLES = [
    {
        "name": "Standard",
        #"system_prompt": "You are a helpful assistant that paraphrases and improves text. Correct any grammar or structure issues. Keep the same language as the original text. Your response should ONLY contain the paraphrased text without any explanations, notes, or formatting.",
        "system_prompt": "You are a skilled editor who improves text quality while maintaining the author's voice. Your tasks are to: 1) Fix grammar, punctuation, and spelling errors, 2) Enhance clarity by improving awkward phrasing, 3) Suggest better word choices where appropriate, and 4) Reorganize sentences for better flow when needed. Make moderate changes that preserve the original meaning but improve its expression. Keep the same language as the original text. Your response should ONLY contain the improved text without any explanations or notes.",
        "temperature": 0.3
    },
    {
        "name": "More Creative",
        "system_prompt": "You are a creative writer who reimagines the text with new phrasing while preserving the original meaning. Use varied vocabulary and sentence structures. Keep the same language as the original text. Your response should ONLY contain the paraphrased text without any explanations.",
        "temperature": 0.6
    },
    {
        "name": "Professional",
        "system_prompt": "You are a professional editor who refines text to be clear, concise, and formal. Improve structure and flow while maintaining the core message. Keep the same language as the original text. Your response should ONLY contain the edited text without any explanations.",
        "temperature": 0.4
    },
    {
        "name": "Simplified",
        "system_prompt": "You are an editor who makes text more accessible and easier to understand. Use simpler words and shorter sentences while keeping the same meaning. Keep the same language as the original text. Your response should ONLY contain the simplified text without any explanations.",
        "temperature": 0.3
    }
]

class PromptEntryBox:
    def __init__(self, parent, callback, exit_callback):
        """Initialize the prompt entry box
        
        Args:
            parent: The parent tkinter window
            callback: Function to call with the entered prompt
            exit_callback: Function to call when canceling
        """
        self.parent = parent
        self.callback = callback
        self.exit_callback = exit_callback
        self.window = None
        
        # Load settings from config file
        self.config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompt_box_config.json')
        self.load_config()
        
        # Resize state
        self.resizing = False
        self.resize_edge = None
        self.resize_start_x = 0
        self.resize_start_y = 0
        self.resize_start_width = 0
        self.resize_start_height = 0
        
        # Minimum size constraints
        self.min_width = 300
        self.min_height = 150
    
    def load_config(self):
        """Load configuration from file"""
        # Default values
        self.last_width = 500
        self.last_height = 180
        self.last_x = -1  # -1 means center
        self.last_y = -1  # -1 means default position
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.last_width = config.get('width', 500)
                    self.last_height = config.get('height', 180)
                    self.last_x = config.get('x', -1)
                    self.last_y = config.get('y', -1)
                    print(f"Loaded config: {self.last_width}x{self.last_height} at ({self.last_x},{self.last_y})")
            else:
                print(f"Config file not found, using defaults")
        except Exception as e:
            print(f"Error loading config: {e}")
        
    def save_config(self):
        """Save configuration to file"""
        try:
            config = {
                'width': self.last_width,
                'height': self.last_height,
                'x': self.last_x,
                'y': self.last_y
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f)
                print(f"Saved config: {self.last_width}x{self.last_height} at ({self.last_x},{self.last_y})")
        except Exception as e:
            print(f"Error saving config: {e}")
        
    def create_resize_bindings(self):
        """Create all resize handle bindings"""
        # Resize handle bindings
        self.canvas.tag_bind("resize_br", "<ButtonPress-1>", lambda e: self.start_resize(e, "br"))
        self.canvas.tag_bind("resize_r", "<ButtonPress-1>", lambda e: self.start_resize(e, "r"))
        self.canvas.tag_bind("resize_b", "<ButtonPress-1>", lambda e: self.start_resize(e, "b"))
        
        # Mouse cursor change on hover over resize handles
        self.canvas.tag_bind("resize_br", "<Enter>", lambda e: self.window.config(cursor="sizing"))
        self.canvas.tag_bind("resize_br", "<Leave>", lambda e: self.window.config(cursor=""))
        self.canvas.tag_bind("resize_r", "<Enter>", lambda e: self.window.config(cursor="size_we"))
        self.canvas.tag_bind("resize_r", "<Leave>", lambda e: self.window.config(cursor=""))
        self.canvas.tag_bind("resize_b", "<Enter>", lambda e: self.window.config(cursor="size_ns"))
        self.canvas.tag_bind("resize_b", "<Leave>", lambda e: self.window.config(cursor=""))
        
    def show(self, x=None, y=None):
        """Show the prompt entry box at the specified location
        
        Args:
            x: X position (if None, uses saved position or centers)
            y: Y position (if None, uses saved position or default)
        """
        # If already showing, bring to front and return
        if self.window and self.window.winfo_exists():
            self.window.lift()
            self.window.focus_force()
            self.text_entry.focus_set()
            return
            
        # Create a new Toplevel window
        self.window = tk.Toplevel(self.parent)
        self.window.title("Assistant Prompt")
        
        # Remove window decorations for a cleaner look
        self.window.overrideredirect(True)
        
        # Set window properties
        self.window.attributes('-topmost', True)
        
        # Use saved size
        width = self.last_width
        height = self.last_height
        
        # If explicit position is given, use it
        use_x = x
        use_y = y
        
        # If no explicit position and we have a saved position, use saved
        if use_x is None and self.last_x >= 0:
            use_x = self.last_x
            print(f"Using saved X position: {use_x}")
        
        if use_y is None and self.last_y >= 0:
            use_y = self.last_y
            print(f"Using saved Y position: {use_y}")
        
        # If still no position, calculate default position
        if use_x is None:
            screen_width = self.window.winfo_screenwidth()
            use_x = (screen_width - width) // 2
            print(f"Using calculated X position: {use_x}")
        
        if use_y is None:
            screen_height = self.window.winfo_screenheight()
            use_y = screen_height - height - 100  # 100px from bottom
            print(f"Using calculated Y position: {use_y}")
        
        print(f"Setting window geometry to: {width}x{height}+{use_x}+{use_y}")
        self.window.geometry(f"{width}x{height}+{use_x}+{use_y}")
        
        # Set dark theme colors
        bg_color = "#2E2E2E"
        fg_color = "#FFFFFF"
        accent_color = "#4287f5"
        button_bg = "#3D3D3D"
        
        self.window.configure(bg=bg_color)
        
        # Create rounded frame with shadow effect using Canvas
        self.canvas = tk.Canvas(self.window, bg=bg_color, highlightthickness=0)
        self.canvas.place(x=0, y=0, relwidth=1, relheight=1)
        
        # Create a shadow effect with fixed colors instead of alpha transparency
        shadow_colors = ["#333333", "#3A3A3A", "#414141", "#484848", "#505050"]
        
        # Create an update_canvas method to update the interface elements when window resizes
        def update_canvas(event=None):
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Clear the canvas
            self.canvas.delete("all")
            
            # Redraw shadows
            for i in range(5):
                self.canvas.create_rectangle(
                    10-i, 10-i, canvas_width-10+i, canvas_height-10+i,
                    outline=shadow_colors[i], width=1
                )
            
            # Redraw main rectangle
            self.canvas.create_rectangle(
                10, 10, canvas_width-10, canvas_height-10,
                fill=bg_color, outline=accent_color, width=2, tags="main_rect"
            )
            
            # Create resize handles in corners and sides
            # Bottom right corner (diagonal resize)
            self.canvas.create_rectangle(
                canvas_width-20, canvas_height-20, canvas_width-10, canvas_height-10,
                fill=accent_color, outline=accent_color, tags="resize_br"
            )
            
            # Right edge (horizontal resize)
            self.canvas.create_rectangle(
                canvas_width-15, canvas_height//2-15, canvas_width-10, canvas_height//2+15,
                fill=accent_color, outline=accent_color, tags="resize_r"
            )
            
            # Bottom edge (vertical resize)
            self.canvas.create_rectangle(
                canvas_width//2-15, canvas_height-15, canvas_width//2+15, canvas_height-10,
                fill=accent_color, outline=accent_color, tags="resize_b"
            )
            
            # Set up resize bindings for the newly created handles
            self.create_resize_bindings()
        
        # Bind to configure event to update canvas when window size changes
        self.canvas.bind("<Configure>", update_canvas)
        
        # Create a frame for content
        content_frame = tk.Frame(self.window, bg=bg_color)
        content_frame.place(x=20, y=20, relwidth=1, relheight=1, width=-40, height=-40)
        
        # Create title label
        title_font = tkFont.Font(family="Arial", size=12, weight="bold")
        title_label = tk.Label(
            content_frame, 
            text="Please enter your prompt", 
            font=title_font, 
            bg=bg_color, 
            fg=fg_color
        )
        title_label.pack(pady=(0, 10))
        
        # Create text entry with custom styling
        self.text_entry = tk.Text(
            content_frame,
            font=("Arial", 12),
            bg="#3D3D3D",
            fg=fg_color,
            insertbackground=fg_color,  # Cursor color
            relief=tk.FLAT,
            padx=8,
            pady=8
        )
        self.text_entry.pack(pady=5, fill=tk.BOTH, expand=True)
        self.text_entry.focus_set()
        
        # Create buttons frame
        button_frame = tk.Frame(content_frame, bg=bg_color)
        button_frame.pack(pady=10, fill=tk.X)
        
        # Style for buttons
        button_style = {
            "font": ("Arial", 9),
            "borderwidth": 0,
            "highlightthickness": 0,
            "padx": 15,
            "pady": 6,
            "relief": tk.FLAT
        }
        
        # Create buttons
        submit_button = tk.Button(
            button_frame,
            text="Submit",
            bg=accent_color,
            fg=fg_color,
            command=self.submit,
            activebackground="#3b78de",
            activeforeground=fg_color,
            **button_style
        )
        
        cancel_button = tk.Button(
            button_frame,
            text="Cancel",
            bg=button_bg,
            fg=fg_color,
            command=self.close,
            activebackground="#4D4D4D",
            activeforeground=fg_color,
            **button_style
        )
        
        # Pack buttons
        cancel_button.pack(side=tk.RIGHT, padx=5)
        submit_button.pack(side=tk.RIGHT, padx=5)
        
        # Add key bindings
        self.window.bind("<Return>", lambda event: self.submit())
        self.window.bind("<Escape>", lambda event: self.close())
        
        # Handle window close
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        
        # Add drag functionality for moving the window
        self.canvas.bind("<ButtonPress-1>", self.start_move)
        self.canvas.bind("<ButtonRelease-1>", self.stop_move)
        self.canvas.bind("<B1-Motion>", self.do_move)
        
        # Add resize event handlers
        self.window.bind("<ButtonRelease-1>", self.stop_resize)
        self.window.bind("<B1-Motion>", self.do_resize)
        
        # Initialize drag variables
        self.x = 0
        self.y = 0
        
        # Initial draw
        self.window.update_idletasks()
        update_canvas()
        
    def update_geometry(self):
        """Update stored geometry info from current window"""
        if self.window and self.window.winfo_exists():
            self.last_width = self.window.winfo_width()
            self.last_height = self.window.winfo_height()
            self.last_x = self.window.winfo_x()
            self.last_y = self.window.winfo_y()
            
            print(f"Updated geometry: {self.last_width}x{self.last_height} at ({self.last_x},{self.last_y})")
        
    def start_move(self, event):
        """Start window drag operation"""
        # Only start move if not on resize handle
        item = self.canvas.find_withtag("current")
        if item and any(tag in self.canvas.gettags(item) for tag in ["resize_br", "resize_r", "resize_b"]):
            return
            
        self.x = event.x
        self.y = event.y
        
    def stop_move(self, event):
        """Stop window drag operation"""
        if self.x is not None:
            # Update and save position
            self.update_geometry()
            self.save_config()
            
        self.x = None
        self.y = None
        
    def do_move(self, event):
        """Move window during drag operation"""
        if self.window and self.x is not None and self.y is not None:
            deltax = event.x - self.x
            deltay = event.y - self.y
            x = self.window.winfo_x() + deltax
            y = self.window.winfo_y() + deltay
            self.window.geometry(f"+{x}+{y}")
    
    def start_resize(self, event, edge):
        """Start resize operation
        
        Args:
            event: The mouse event
            edge: Which edge/corner is being resized ('br' = bottom-right, 'r' = right, 'b' = bottom)
        """
        self.resizing = True
        self.resize_edge = edge
        self.resize_start_x = event.x_root
        self.resize_start_y = event.y_root
        self.resize_start_width = self.window.winfo_width()
        self.resize_start_height = self.window.winfo_height()
        
    def stop_resize(self, event):
        """Stop resize operation"""
        if self.resizing:
            # Update and save size
            self.update_geometry()
            self.save_config()
            
        self.resizing = False
        self.resize_edge = None
        
    def do_resize(self, event):
        """Handle resizing of the window"""
        if not self.resizing or not self.window:
            return
            
        # Calculate deltas
        delta_x = event.x_root - self.resize_start_x
        delta_y = event.y_root - self.resize_start_y
        
        # Calculate new dimensions based on which edge is being dragged
        new_width = self.resize_start_width
        new_height = self.resize_start_height
        
        if self.resize_edge in ["br", "r"]:  # Right edge or bottom-right corner
            new_width = max(self.min_width, self.resize_start_width + delta_x)
            
        if self.resize_edge in ["br", "b"]:  # Bottom edge or bottom-right corner
            new_height = max(self.min_height, self.resize_start_height + delta_y)
        
        # Apply new size
        self.window.geometry(f"{new_width}x{new_height}")
    
    def submit(self):
        """Process and submit the text prompt"""
        if self.window:
            prompt = self.text_entry.get("1.0", tk.END).strip()
            if prompt:
                # Save position and size before closing
                self.update_geometry()
                self.save_config()
                
                # Close window and return prompt
                self.window.destroy()
                self.window = None
                self.callback(prompt)
            else:
                # Shake the window gently to indicate empty input
                self.shake_window()
    
    def shake_window(self):
        """Shake the window to indicate error"""
        if not self.window:
            return
            
        original_x = self.window.winfo_x()
        
        def _shake(count, distance, speed):
            if count > 0 and self.window:
                self.window.geometry(f"+{original_x + distance}+{self.window.winfo_y()}")
                self.window.after(speed, lambda: _shake(count-1, -distance, speed))
                
        _shake(10, 10, 50)
    
    def close(self):
        """Close the prompt entry box"""
        if self.window:
            # Save position and size before closing
            self.update_geometry()
            self.save_config()
            
            # Then close
            self.window.destroy()
            self.window = None
            if self.exit_callback:
                self.exit_callback()
                
class UnifiedAssistant:
    # Message display timeout settings (in milliseconds)
    MESSAGE_TIMEOUTS = {
        "listening": 0,          # 0 means no timeout (stays until replaced or explicitly hidden)
        "dictating": 0,          # No timeout
        "speech_detected": 0,    # No timeout
        "processing": 0,         # No timeout
        "transcribing": 0,       # No timeout
        "thinking": 0,           # No timeout
        "error": 3000,           # 3 seconds for error messages
        "success": 2000,         # 2 seconds for success messages
        "language": 3000,        # 3 seconds for language notifications
        "microphone": 3000,      # 3 seconds for microphone notifications
        "service": 3000,         # 3 seconds for service notifications
        "transcript": 5000,      # 3 seconds for transcript display
        "playing_audio": 0,      # No timeout while audio is playing
        "default": 3000          # Default timeout for other messages
    }
    
    def __init__(self):
        # Initialize core components
        self.keyboard_controller = Controller()
        self.recording = False
        self.processing = False
        self.audio_thread = None
        self.target_window = None
        
        # State tracking
        self.mode = "idle"  # idle, dictation, assistant, paraphrase, text_prompt
        self.paraphrase_style_index = 0
        self.current_language = os.getenv('DEFAULT_LANGUAGE', 'english').lower()
        if self.current_language not in ["english", "spanish"]:
            self.current_language = "english"
            
        # Transcription service setting
        self.transcription_service = TRANSCRIPTION_SERVICE
        
        # Initialize OpenAI client
        if OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            self.openai_client = None
            print("Warning: OpenAI API key not found. OpenAI transcription will not be available.")

        # Setup message queue and notification system
        self.message_queue = queue.Queue(maxsize=10)
        
        # Initialize audio
        self.audio = pyaudio.PyAudio()
        
        # Initialize microphone devices
        self.microphone_index = SELECTED_MICROPHONE_INDEX
        self.microphone_devices = self.get_microphone_devices()
        
        # Custom word dictionary
        self.setup_custom_dictionary()
        self.setup_word_replacements()
        
        # Set up the UI components - IMPORTANT: Create root in main thread
        self.root = tk.Tk()
        self.root.withdraw()
        self.setup_overlay_windows()
        
        # Create text prompt entry box
        self.prompt_entry = None  # Will be initialized when needed
        
        # Load prompt box preferences
        self.prompt_box_width = int(os.getenv("PROMPT_BOX_WIDTH", "500"))
        self.prompt_box_height = int(os.getenv("PROMPT_BOX_HEIGHT", "180"))
        self.prompt_box_x = int(os.getenv("PROMPT_BOX_X", "-1"))
        self.prompt_box_y = int(os.getenv("PROMPT_BOX_Y", "-1"))
        
        # Load personal information
        self.user_info = self.load_user_info()
        
        # Load templates BEFORE setting up the tray icon
        self.templates = {}
        self.load_templates()
        
        # Now create tray icon
        self.setup_tray_icon()
        
        # Register hotkeys
        self.setup_hotkeys()
        
        # Initialize timer for hiding overlay
        self.hide_id = None
    
    def get_microphone_devices(self):
        """Get a list of available microphone devices"""
        microphone_devices = []
        info = self.audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        
        for i in range(num_devices):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info.get('maxInputChannels') > 0:  # This is an input device
                name = device_info.get('name')
                # Sanitize name for menu item
                name = name[:32] if len(name) > 32 else name
                microphone_devices.append({
                    'index': i,
                    'name': name
                })
                print(f"Found microphone: {name} (index: {i})")
        
        return microphone_devices
    
    def set_microphone(self, index):
        """Set the current microphone device by index"""
        try:
            # Check if index is a valid number
            if not isinstance(index, int):
                print(f"Invalid microphone index type: {type(index)}")
                return
                
            self.microphone_index = index
            device_name = "Unknown"
            
            for device in self.microphone_devices:
                if device['index'] == index:
                    device_name = device['name']
                    break
            
            print(f"Microphone set to: {device_name} (index: {index})")
            self.show_message(f"Microphone set to: {device_name}", message_type="microphone")
            
            # Try to save the selected microphone index to .env file
            try:
                # Get absolute path to .env file
                script_dir = os.path.dirname(os.path.abspath(__file__))
                env_path = os.path.join(script_dir, '.env')
                
                # Check if .env file exists and is writable
                if os.path.exists(env_path) and os.access(env_path, os.W_OK):
                    set_key(env_path, 'SELECTED_MICROPHONE_INDEX', str(index))
                else:
                    print(f"Cannot write to .env file at {env_path}")
                    # Try alternative method: write to environment variable in memory
                    os.environ['SELECTED_MICROPHONE_INDEX'] = str(index)
            except Exception as e:
                print(f"Error saving microphone setting: {e}")
                # Continue without saving
        except Exception as e:
            print(f"Error setting microphone: {e}")
    
    def set_transcription_service(self, service):
        """Set the transcription service to use (groq or openai)"""
        if service not in ["groq", "openai"]:
            print(f"Invalid transcription service: {service}")
            return
            
        # If OpenAI is selected but no API key is available
        if service == "openai" and not OPENAI_API_KEY:
            self.show_message("OpenAI API key not found. Please add it to your .env file.", message_type="error")
            return
            
        self.transcription_service = service
        service_info = "Groq" if service == "groq" else "OpenAI"
        
        if service == "groq":
            model_info = f" (using {GROQ_WHISPER_MODEL})"
        else:
            model_info = f" (using {OPENAI_WHISPER_MODEL})"
            
        print(f"Transcription service set to: {service_info}{model_info}")
        self.show_message(f"Transcription: {service_info}{model_info}", message_type="service")
        
        # Save the setting to .env file
        try:
            # Get absolute path to .env file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            env_path = os.path.join(script_dir, '.env')
            
            # Check if .env file exists and is writable
            if os.path.exists(env_path) and os.access(env_path, os.W_OK):
                set_key(env_path, 'TRANSCRIPTION_SERVICE', service)
            else:
                print(f"Cannot write to .env file at {env_path}")
                # Try alternative method: write to environment variable in memory
                os.environ['TRANSCRIPTION_SERVICE'] = service
        except Exception as e:
            print(f"Error saving transcription service setting: {e}")
    
    def setup_custom_dictionary(self):
        """Initialize custom dictionary for word correction"""
        self.custom_dictionary = {
            "english": set([
                "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
                "it", "for", "not", "on", "with", "he", "as", "you", "do", "CSAT",
            ]),
            "spanish": set([
                "voz", "la", "de", "que", "y", "a", "en", "un", "ser", "porfa",
                "no", "haber", "por", "con", "su", "para", "como", "estar", "ratito", "CSAT",
            ])
        }
    
    def setup_word_replacements(self):
        """Initialize dictionaries for word replacements in different languages"""
        self.spanish_replacements = {
            "vos": "voz",
            # You can add more problematic Spanish words here:
            # "incorrect_word": "correct_word",
        }
        
        self.english_replacements = {
            # Add English word replacements if needed:
            # "incorrect_word": "correct_word",
        }
    
    def get_monitor_info(self):
        """Get information about all connected monitors"""
        monitors = []
        try:
            for i in range(win32api.GetSystemMetrics(win32con.SM_CMONITORS)):
                monitor = win32api.EnumDisplayMonitors(None, None)[i]
                monitors.append(win32api.GetMonitorInfo(monitor[0]))
        except Exception as e:
            print(f"Error getting monitor info: {e}")
        return monitors or [win32api.GetMonitorInfo(win32api.EnumDisplayMonitors(None, None)[0][0])]

    def setup_overlay_windows(self):
        """Create overlay notification windows for all monitors"""
        self.overlays = []
        monitors = self.get_monitor_info()
        TRANSPARENT_COLOR = 'gray1'

        for monitor_info in monitors:
            overlay = tk.Toplevel(self.root)
            overlay.overrideredirect(True)
            overlay.attributes('-topmost', True)
            overlay.attributes('-alpha', 0.9)
            overlay.configure(bg=TRANSPARENT_COLOR)
            overlay.wm_attributes('-transparentcolor', TRANSPARENT_COLOR)

            monitor_rect = monitor_info['Monitor']
            monitor_width = monitor_rect[2] - monitor_rect[0]
            monitor_height = monitor_rect[3] - monitor_rect[1]
            
            overlay_width = 500
            overlay_height = 28
            
            x_position = monitor_rect[0] + (monitor_width - overlay_width) // 2
            y_position = monitor_rect[3] - overlay_height - 8

            overlay.geometry(f'{overlay_width}x{overlay_height}+{x_position}+{y_position}')

            text_widget = tk.Label(
                overlay,
                foreground='white',
                bg=TRANSPARENT_COLOR,
                font=('Arial', 10, 'bold'),
                anchor='center'
            )
            text_widget.pack(expand=True, fill='both')
            overlay.withdraw()
            
            self.overlays.append((overlay, text_widget))
        
        self.hide_id = None
    
    def setup_tray_icon(self):
        """Set up the system tray icon with menu"""
        # Use absolute paths based on script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        try:
            normal_icon_path = os.path.join(script_dir, "normal_icon.png")
            listening_icon_path = os.path.join(script_dir, "listening_icon.png")
            
            self.normal_icon = Image.open(normal_icon_path)
            self.listening_icon = Image.open(listening_icon_path)
            print(f"Loaded tray icons from {script_dir}")
        except Exception as e:
            # Fallback to generated icons if files not found
            print(f"Could not load icon files: {e}, using generated icons")
            size = 64
            self.normal_icon = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(self.normal_icon)
            draw.ellipse([0, 0, size, size], fill='black')
            
            self.listening_icon = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(self.listening_icon)
            draw.ellipse([0, 0, size, size], fill='green')
        
        # Build microphone selection submenu with proper closure handling
        microphone_menu_items = []
        
        # Define a function factory to properly handle device index in lambdas
        def create_device_callback(idx):
            return lambda _: self.set_microphone(idx)
            
        def create_checked_callback(idx):
            return lambda _: self.microphone_index == idx
        
        for device in self.microphone_devices:
            microphone_menu_items.append(
                pystray.MenuItem(
                    device['name'], 
                    create_device_callback(device['index']),
                    checked=create_checked_callback(device['index'])
                )
            )
        
        # Create microphone submenu carefully
        if microphone_menu_items:
            microphone_menu = pystray.Menu(*microphone_menu_items)
        else:
            # Fallback if no microphones found
            microphone_menu = pystray.Menu(
                pystray.MenuItem("No microphones found", lambda: None)
            )
            
        # Create transcription service submenu
        transcription_menu = pystray.Menu(
            pystray.MenuItem('Groq', lambda: self.set_transcription_service("groq"), 
                checked=lambda item: self.transcription_service == "groq"),
            pystray.MenuItem('OpenAI', lambda: self.set_transcription_service("openai"), 
                checked=lambda item: self.transcription_service == "openai")
        )

        # Add voice selection submenu
        voice_options = [
            "Arista-PlayAI", 
            "Atlas-PlayAI", 
            "Basil-PlayAI", 
            "Briggs-PlayAI", 
            "Calum-PlayAI", 
            "Celeste-PlayAI", 
            "Cheyenne-PlayAI", 
            "Chip-PlayAI", 
            "Cillian-PlayAI", 
            "Deedee-PlayAI", 
            "Fritz-PlayAI", 
            "Gail-PlayAI", 
            "Indigo-PlayAI", 
            "Mamaw-PlayAI", 
            "Mason-PlayAI", 
            "Mikail-PlayAI", 
            "Mitch-PlayAI", 
            "Quinn-PlayAI", 
            "Thunder-PlayAI"
        ]
        
        def create_voice_callback(voice):
            return lambda _: self.set_tts_voice(voice)
            
        def create_voice_checked_callback(voice):
            return lambda _: TTS_VOICE == voice
            
        voice_menu_items = [
            pystray.MenuItem(
                voice, 
                create_voice_callback(voice),
                checked=create_voice_checked_callback(voice)
            ) for voice in voice_options
        ]
        
        voice_menu = pystray.Menu(*voice_menu_items)

        # Create input methods submenu
        input_methods_menu = pystray.Menu(
            pystray.MenuItem(f'Voice Input (Press {ASSISTANT_HOTKEY})', None, enabled=False),
            pystray.MenuItem(f'Text Input (Press {TEXT_PROMPT_HOTKEY})', lambda: self.show_text_prompt_entry()),
            pystray.MenuItem(f'Dictation (Press {DICTATION_HOTKEY})', None, enabled=False),
            pystray.MenuItem(f'Paraphrase (Press {PARAPHRASE_HOTKEY})', None, enabled=False),
            pystray.MenuItem(f'Text-to-Speech (Press {TEXT_TO_SPEECH_HOTKEY})', None, enabled=False)
        )

        # Create templates submenu
        template_menu_items = []
        for template_name in self.templates.keys():
            display_name = template_name.replace('_', ' ').title()
            
            # Create a function factory that returns a function
            def make_template_callback(template_key):
                def callback():
                    return self.show_template(template_key)
                return callback
            
            template_menu_items.append(
                pystray.MenuItem(
                    display_name,
                    make_template_callback(template_name)
                )
            )

        if template_menu_items:
            template_menu_items.append(pystray.MenuItem('Reload Templates', self.reload_templates))
            template_menu = pystray.Menu(*template_menu_items)
        else:
            template_menu = pystray.Menu(
                pystray.MenuItem("No templates found", lambda: None),
                pystray.MenuItem('Load Templates', self.reload_templates)
            )

        menu = pystray.Menu(
            pystray.MenuItem('Input Methods', input_methods_menu),
            pystray.MenuItem('Templates', template_menu),
            pystray.MenuItem('Microphone', microphone_menu),
            pystray.MenuItem('Transcription Service', transcription_menu),
            pystray.MenuItem('TTS Voice', voice_menu),
            pystray.MenuItem('Dictation Language', pystray.Menu(
                pystray.MenuItem('English', lambda: self.set_language("english"), 
                    checked=lambda item: self.current_language == "english"),
                pystray.MenuItem('Spanish', lambda: self.set_language("spanish"), 
                    checked=lambda item: self.current_language == "spanish")
            )),
            pystray.MenuItem('Paraphrase Style', pystray.Menu(
                *[pystray.MenuItem(style["name"], lambda style_idx=idx: self.set_paraphrase_style(style_idx),
                    checked=lambda item, style_idx=idx: self.paraphrase_style_index == style_idx)
                  for idx, style in enumerate(PARAPHRASING_STYLES)]
            )),
            pystray.MenuItem('Set Default Dictation Language', self.save_default_language),
            pystray.MenuItem('Exit', self.exit_app)
        )
        
        self.tray_icon = pystray.Icon(
            "Unified Assistant",
            self.normal_icon,
            "Unified Assistant",
            menu
        )
        
        # Run the icon in a detached way - don't block the main thread
        threading.Thread(target=self.tray_icon.run, daemon=True).start()
        
    def set_language(self, language):
        """Change the current language"""
        self.current_language = language
        # Capitalize the language name for display purposes
        display_language = language.capitalize()
        print(f"Language set to: {display_language}")
        self.show_message(f"Language set to: {display_language}", message_type="language")
    
    def toggle_language(self, e=None):
        """Toggle between English and Spanish"""
        if self.current_language == "english":
            self.set_language("spanish")
        else:
            self.set_language("english")
    
    def set_paraphrase_style(self, style_idx):
        """Set the paraphrasing style"""
        self.paraphrase_style_index = style_idx
        style_name = PARAPHRASING_STYLES[style_idx]["name"]
        print(f"Paraphrase style set to: {style_name}")
        self.show_message(f"Paraphrase style set to: {style_name}")
    
    def save_default_language(self):
        """Save the current language as default"""
        try:
            # Get absolute path to .env file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            env_path = os.path.join(script_dir, '.env')
            
            # Check if .env file exists and is writable
            if os.path.exists(env_path) and os.access(env_path, os.W_OK):
                set_key(env_path, 'DEFAULT_LANGUAGE', self.current_language)
                print(f"Default language saved: {self.current_language}")
                self.show_message(f"Default language saved: {self.current_language}", message_type="success")
            else:
                # Fallback: Try to save to environment variable in memory
                os.environ['DEFAULT_LANGUAGE'] = self.current_language
                print(f"Default language set in memory only: {self.current_language} (could not write to .env file)")
                self.show_message(f"Language preference set for this session only", message_type="success")
        except Exception as e:
            error_msg = f"Could not save language preference: {str(e)}"
            print(error_msg)
            self.show_message(error_msg, message_type="error")
    
    def setup_hotkeys(self):
        """Register all keyboard shortcuts"""
        # Dictation hotkey (press and hold)
        keyboard.on_press_key(DICTATION_HOTKEY, self.start_dictation, suppress=True)
        keyboard.on_release_key(DICTATION_HOTKEY, self.stop_recording, suppress=True)
        
        # Assistant hotkey (press and hold)
        keyboard.on_press_key(ASSISTANT_HOTKEY, self.start_assistant, suppress=True)
        keyboard.on_release_key(ASSISTANT_HOTKEY, self.stop_recording, suppress=True)
        
        # Paraphrase hotkey
        keyboard.add_hotkey(PARAPHRASE_HOTKEY, self.paraphrase_selected_text)
        
        # Language toggle
        keyboard.add_hotkey(LANGUAGE_TOGGLE_HOTKEY, self.toggle_language)
        
        # TTS
        keyboard.add_hotkey(TEXT_TO_SPEECH_HOTKEY, self.speak_selected_text)
        
        # New hotkey for text prompt entry
        keyboard.add_hotkey(TEXT_PROMPT_HOTKEY, self.show_text_prompt_entry)
    
    def show_message(self, message, timeout=None, message_type=None):
        """Show a message in the overlay windows
        
        Args:
            message (str): The message to display
            timeout (int, optional): Specific timeout in milliseconds. If None, uses message_type timeout.
            message_type (str, optional): Type of message for predefined timeouts.
        """
        # Determine timeout based on message_type if not explicitly specified
        if timeout is None:
            if message_type and message_type in self.MESSAGE_TIMEOUTS:
                timeout = self.MESSAGE_TIMEOUTS[message_type]
            else:
                # Try to guess message type from content
                for msg_type, msg_timeout in self.MESSAGE_TIMEOUTS.items():
                    if msg_type in message.lower():
                        timeout = msg_timeout
                        break
                else:
                    timeout = self.MESSAGE_TIMEOUTS["default"]
        
        # Display the message in all overlay windows
        for overlay, text_widget in self.overlays:
            text_widget.config(text=message)
            overlay.deiconify()
        
        # Cancel any existing hide timer
        if self.hide_id:
            self.root.after_cancel(self.hide_id)
            self.hide_id = None
        
        # Set a new timer to hide the message, but only if timeout > 0
        if timeout > 0:
            self.hide_id = self.root.after(timeout, self.hide_overlay)
    
    def hide_overlay(self):
        """Hide all overlay windows"""
        for overlay, _ in self.overlays:
            overlay.withdraw()
        self.hide_id = None
    
    def set_icon_state(self, listening=False):
        """Change the tray icon based on recording state"""
        self.tray_icon.icon = self.listening_icon if listening else self.normal_icon
    
    def start_dictation(self, e):
        """Start recording for dictation mode"""
        if self.processing or self.recording:
            return
            
        self.recording = True
        self.mode = "dictation"
        self.target_window = win32gui.GetForegroundWindow()
        self.set_icon_state(listening=True)
        self.show_message("Dictating...", message_type="dictating")
        
        self.frames = []
        self.audio_thread = threading.Thread(target=self.record_audio_continuous, args=(self.process_dictation,))
        self.audio_thread.start()
    
    def start_assistant(self, e):
        """Start recording for assistant mode"""
        if self.processing or self.recording:
            return
            
        self.recording = True
        self.mode = "assistant"
        self.target_window = win32gui.GetForegroundWindow()
        self.set_icon_state(listening=True)
        self.show_message("Speak your prompt...", message_type="listening")
        
        self.frames = []
        self.audio_thread = threading.Thread(target=self.record_audio_continuous, args=(self.process_assistant_query,))
        self.audio_thread.start()
    
    def stop_recording(self, e):
        """Stop the current recording"""
        if self.recording:
            self.recording = False
            
    def record_audio_continuous(self, callback_function):
        """Record audio continuously as long as the hotkey is pressed"""
        try:
            stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=self.microphone_index
            )
            
            frames = []
            
            # Variables for speech detection
            speech_detected = False
            
            while self.recording:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                
                # Convert audio chunk to numpy array for amplitude calculation
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # Calculate audio amplitude
                amplitude = np.abs(audio_data).mean()
                
                # Check if speech has been detected (for UI feedback only)
                if not speech_detected and amplitude > SPEECH_DETECTED_THRESHOLD:
                    speech_detected = True
                    self.show_message("Speech detected...", message_type="speech_detected")
            
            self.show_message("Processing...", message_type="processing")
            
            stream.stop_stream()
            stream.close()
            
            # Only process audio if we actually collected frames
            if not frames:
                self.show_message("No audio recorded", message_type="error")
                self.set_icon_state(listening=False)
                return
            
            # Set processing flag to prevent hotkey activation during processing
            self.processing = True
            self.set_icon_state(listening=False)
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_filename = temp_audio.name
                wf = wave.open(temp_filename, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
            
            # Call the appropriate callback based on mode
            try:
                callback_function(temp_filename)
            finally:
                # Clean up
                try:
                    os.unlink(temp_filename)
                except:
                    pass
                self.processing = False
                
        except Exception as e:
            self.processing = False
            self.recording = False
            self.set_icon_state(listening=False)
            error_msg = f"Error opening audio stream: {str(e)}"
            print(error_msg)
            self.show_message(error_msg, message_type="error")
    
    def transcribe_audio(self, audio_file):
        """Transcribe audio file using selected service (Groq or OpenAI)"""
        self.show_message("Transcribing...", message_type="transcribing")
        
        if self.transcription_service == "openai":
            transcript = self.transcribe_audio_openai(audio_file)
        else:  # Default to Groq
            transcript = self.transcribe_audio_groq(audio_file)
        
        # Show the transcript on screen if we got one
        if transcript:
            self.show_message(f"Transcript: {transcript}", timeout=3000)  # Display for 3 seconds
        
        return transcript
    
    def transcribe_audio_groq(self, audio_file):
        """Transcribe audio file using Groq Whisper API"""
        try:
            with open(audio_file, "rb") as file:
                headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
                files = {"file": (os.path.basename(audio_file), file, "audio/wav")}
                data = {
                    "model": GROQ_WHISPER_MODEL,
                    "language": "es" if self.current_language == "spanish" else "en"
                }
                
                response = requests.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers=headers,
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    transcript = result["text"]
                    
                    # Apply word replacements for the current language
                    if self.current_language == "spanish" and hasattr(self, 'spanish_replacements'):
                        transcript = self.apply_word_replacements(transcript, self.spanish_replacements)
                    elif self.current_language == "english" and hasattr(self, 'english_replacements'):
                        transcript = self.apply_word_replacements(transcript, self.english_replacements)
                    
                    # Show the transcript on screen for feedback
                    self.show_message(f"Transcript: {transcript}", message_type="transcript", timeout=3000)
                    
                    # Apply custom dictionary corrections
                    transcript = self.apply_custom_dictionary(transcript)
                    
                    print(f"Groq Transcription: {transcript}")
                    return transcript
                else:
                    print(f"Groq transcription failed: {response.status_code} - {response.text}")
                    self.show_message(f"Transcription failed: {response.status_code}", message_type="error")
                    return None
        except Exception as e:
            print(f"Error during Groq transcription: {str(e)}")
            self.show_message(f"Error: {str(e)}", message_type="error")
            return None

    def transcribe_audio_openai(self, audio_file):
        """Transcribe audio file using OpenAI Whisper API"""
        try:
            if not self.openai_client:
                self.show_message("OpenAI API key not found", message_type="error")
                return None
                
            with open(audio_file, "rb") as file:
                response = self.openai_client.audio.transcriptions.create(
                    model=OPENAI_WHISPER_MODEL,
                    file=file,
                    language="es" if self.current_language == "spanish" else "en",
                    response_format="text"
                )
                
                transcript = response
                
                # Apply word replacements for the current language
                if self.current_language == "spanish" and hasattr(self, 'spanish_replacements'):
                    transcript = self.apply_word_replacements(transcript, self.spanish_replacements)
                elif self.current_language == "english" and hasattr(self, 'english_replacements'):
                    transcript = self.apply_word_replacements(transcript, self.english_replacements)
                
                # Show the transcript on screen for feedback
                self.show_message(f"Transcript: {transcript}", message_type="transcript", timeout=3000)
                
                # Apply custom dictionary corrections
                transcript = self.apply_custom_dictionary(transcript)
                
                print(f"OpenAI Transcription: {transcript}")
                return transcript
        except Exception as e:
            print(f"Error during OpenAI transcription: {str(e)}")
            self.show_message(f"OpenAI Error: {str(e)}", message_type="error")
            return None
            
    def apply_custom_dictionary(self, text):
        """Apply custom dictionary corrections to transcribed text"""
        words = text.split()
        corrected_words = []
        for word in words:
            lower_word = word.lower()
            if lower_word in self.custom_dictionary[self.current_language]:
                corrected_words.append(word)
            else:
                corrected_words.append(word)
        return ' '.join(corrected_words)
        
    def apply_word_replacements(self, text, replacements):
        """Apply word replacements using regex
        
        Args:
            text (str): The text to process
            replacements (dict): Dictionary of incorrect->correct words
            
        Returns:
            str: The processed text with replacements applied
        """
        import re
        
        # Skip if no replacements
        if not replacements:
            return text
            
        # Create a regex pattern that matches any of the words in the dictionary
        # using word boundaries to ensure we only match whole words
        pattern = r'\b(' + '|'.join([re.escape(word) for word in replacements.keys()]) + r')\b'
        
        def replace_word(match):
            word = match.group(0)
            word_lower = word.lower()
            
            if word_lower in replacements:
                correct_word = replacements[word_lower]
                
                # Preserve capitalization
                if word.islower():
                    return correct_word
                elif word.isupper():
                    return correct_word.upper()
                elif word[0].isupper():
                    return correct_word.capitalize()
                return correct_word
                
            return word  # If no replacement found, return original
        
        # Apply all replacements in one pass
        return re.sub(pattern, replace_word, text)    
        
    def process_assistant_query(self, audio_file):
        """Process assistant query audio"""
        transcript = self.transcribe_audio(audio_file)
        if not transcript:
            return
        
        # Get selected text to provide context
        selected_text = self.get_selected_text()
        
        # Process with LLM
        self.show_message("Thinking...", message_type="thinking")
        response = self.process_with_llm(transcript, selected_text)
        
        # Insert response
        if response:
            self.insert_text(response)
            self.show_message("Response inserted", message_type="success")
    
    def process_dictation(self, audio_file):
        """Process dictation audio with special handling for Notepad"""
        transcript = self.transcribe_audio(audio_file)
        if not transcript:
            return
            
        # Check if the target is Notepad for special handling
        is_notepad = False
        try:
            if self.target_window:
                window_class = win32gui.GetClassName(self.target_window)
                window_title = win32gui.GetWindowText(self.target_window)
                
                # Check if we're dealing with Notepad
                if window_class == "Notepad" or "Notepad" in window_title:
                    is_notepad = True
                    print(f"Notepad detected for dictation. Text length: {len(transcript)}")
        except Exception as e:
            print(f"Error checking window: {e}")
        
        # For dictation, insert the transcribed text
        if self.target_window:
            win32gui.SetForegroundWindow(self.target_window)
            time.sleep(0.2)
            
            if is_notepad:
                # Notepad-specific approach
                try:
                    # Try to find the Edit control in Notepad
                    edit_handle = win32gui.FindWindowEx(self.target_window, 0, "Edit", None)
                    if edit_handle:
                        print("Using Windows API to append text directly to Notepad")
                        # Get current text
                        current_text = win32gui.GetWindowText(edit_handle)
                        # Append new text
                        win32gui.SendMessage(edit_handle, win32con.WM_SETTEXT, 0, current_text + transcript)
                        self.show_message("Text inserted", message_type="success")
                        return
                except Exception as e:
                    print(f"Direct Notepad text insertion failed: {e}")
                    # Fall back to normal method below
            
            # Regular method for non-Notepad or if Notepad method failed
            self.insert_text(transcript)
            self.show_message("Text inserted", message_type="success")
    
    def get_selected_text(self):
        """Get any currently selected text with improved cross-monitor reliability"""
        # Store original clipboard content
        original_clipboard = pyperclip.paste()
        
        # Clear clipboard first to ensure we can detect if nothing is selected
        pyperclip.copy('')
        time.sleep(0.2)  # Small delay after clearing
        
        # Make sure the target window is in focus
        if self.target_window:
            try:
                # Check if window still exists
                if win32gui.IsWindow(self.target_window):
                    # Try to bring window to foreground
                    win32gui.SetForegroundWindow(self.target_window)
                    time.sleep(0.3)  # Give more time for window to come to foreground
            except Exception as e:
                print(f"Error setting foreground window: {e}")
                # If we can't set foreground, the window might be on another monitor
                # or might not exist anymore
                self.target_window = win32gui.GetForegroundWindow()
                print(f"Reset target window to current foreground: {self.target_window}")
        else:
            # No target window set, use current foreground window
            self.target_window = win32gui.GetForegroundWindow()
            print(f"Using current foreground window: {self.target_window}")
        
        # Try multiple methods to copy selected text
        selected_text = ""
        
        # Method 1: Using pynput
        try:
            self.keyboard_controller.press(Key.ctrl)
            self.keyboard_controller.press('c')
            self.keyboard_controller.release('c')
            self.keyboard_controller.release(Key.ctrl)
            
            # Give system time to update clipboard
            time.sleep(0.5)
            
            # Get the selected text
            selected_text = pyperclip.paste()
            
            # If we got text, we're done
            if selected_text:
                print(f"Method 1 success: Got {len(selected_text)} characters")
        except Exception as e:
            print(f"Method 1 error: {e}")
        
        # Method 2: Using keyboard library
        if not selected_text:
            try:
                print("Trying method 2: keyboard library")
                keyboard.press_and_release('ctrl+c')
                time.sleep(0.5)
                selected_text = pyperclip.paste()
                
                if selected_text:
                    print(f"Method 2 success: Got {len(selected_text)} characters")
            except Exception as e:
                print(f"Method 2 error: {e}")
        
        # Method 3: Using SendInput for low-level keyboard input
        if not selected_text:
            try:
                print("Trying method 3: SendInput")
                # Import necessary modules for SendInput
                import ctypes
                from ctypes import wintypes
                
                user32 = ctypes.WinDLL('user32', use_last_error=True)
                
                # Constants for input
                INPUT_KEYBOARD = 1
                KEYEVENTF_KEYUP = 0x2
                VK_CONTROL = 0x11
                VK_C = 0x43
                
                # Input type
                class KEYBDINPUT(ctypes.Structure):
                    _fields_ = (("wVk", wintypes.WORD),
                               ("wScan", wintypes.WORD),
                               ("dwFlags", wintypes.DWORD),
                               ("time", wintypes.DWORD),
                               ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)))
                
                class INPUT(ctypes.Structure):
                    _fields_ = (("type", wintypes.DWORD),
                               ("ki", KEYBDINPUT),
                               ("padding", ctypes.c_char * 8))
                
                # Create array of INPUT structures
                inputs = (INPUT * 4)()
                
                # Press CTRL
                inputs[0].type = INPUT_KEYBOARD
                inputs[0].ki.wVk = VK_CONTROL
                
                # Press C
                inputs[1].type = INPUT_KEYBOARD
                inputs[1].ki.wVk = VK_C
                
                # Release C
                inputs[2].type = INPUT_KEYBOARD
                inputs[2].ki.wVk = VK_C
                inputs[2].ki.dwFlags = KEYEVENTF_KEYUP
                
                # Release CTRL
                inputs[3].type = INPUT_KEYBOARD
                inputs[3].ki.wVk = VK_CONTROL
                inputs[3].ki.dwFlags = KEYEVENTF_KEYUP
                
                # Send input
                user32.SendInput(4, ctypes.byref(inputs), ctypes.sizeof(INPUT))
                
                time.sleep(0.5)
                selected_text = pyperclip.paste()
                
                if selected_text:
                    print(f"Method 3 success: Got {len(selected_text)} characters")
            except Exception as e:
                print(f"Method 3 error: {e}")
        
        # Log result
        if not selected_text:
            print("No text was copied to clipboard using any method")
        
        # Restore original clipboard content
        pyperclip.copy(original_clipboard)
        
        return selected_text
        
    def show_text_prompt_entry(self, e=None):
        """Show the text prompt entry box"""
        # Keep track of the active window for later
        self.target_window = win32gui.GetForegroundWindow()
        
        # Create prompt entry box if it doesn't exist
        if not hasattr(self, 'prompt_entry') or self.prompt_entry is None:
            self.prompt_entry = PromptEntryBox(
                self.root,
                self.process_text_prompt,
                self.on_prompt_cancelled
            )
        
        # Show the entry box without specifying position
        # The PromptEntryBox will use saved position from its config file
        self.prompt_entry.show()
        
    def process_text_prompt(self, prompt):
        """Process text prompt from the entry box
        
        Args:
            prompt (str): The text prompt entered by the user
        """
        # Make the target window active again
        if self.target_window:
            try:
                win32gui.SetForegroundWindow(self.target_window)
                time.sleep(0.2)
            except:
                pass
        
        # Get any selected text for context
        selected_text = self.get_selected_text()
        
        # Show processing message
        self.show_message("Processing prompt...", message_type="processing")
        
        # Process with LLM
        response = self.process_with_llm(prompt, selected_text)
        
        # Insert the response if we got one
        if response:
            self.insert_text(response)
            self.show_message("Response inserted", message_type="success")
    
    def on_prompt_cancelled(self):
        """Called when prompt entry is cancelled"""
        # Reactivate the original target window
        if self.target_window:
            try:
                win32gui.SetForegroundWindow(self.target_window)
            except:
                pass
    
    def speak_selected_text(self):
        """Speak the currently selected text using Groq TTS API with improved cross-monitor support"""
        if self.processing:
            return
            
        # Store the current active window
        previous_window = self.target_window
        self.target_window = win32gui.GetForegroundWindow()
        print(f"TTS from window handle: {self.target_window}")
        
        # Get the selected text
        selected_text = self.get_selected_text()
        
        if not selected_text or not selected_text.strip():
            self.show_message("No text selected", message_type="error")
            # Restore previous target window if no text was found
            self.target_window = previous_window
            return
        
        # Show we're processing
        self.processing = True
        self.show_message("Converting text to speech...", message_type="processing")
        
        # Run in a separate thread to not block the UI
        threading.Thread(target=self._process_text_to_speech, args=(selected_text,), daemon=True).start()

    # Add this helper method to check if a window is actually visible (not minimized, etc.)
    def is_window_visible_on_screen(self, hwnd):
        """Check if a window is visible on any monitor"""
        try:
            # Check if window exists and is visible
            if not win32gui.IsWindow(hwnd) or not win32gui.IsWindowVisible(hwnd):
                return False
                
            # Get window rectangle
            rect = win32gui.GetWindowRect(hwnd)
            if not rect:
                return False
                
            # Check if window is minimized
            if rect[0] == -32000 or rect[1] == -32000:
                return False
                
            # Get monitor info for all monitors
            monitors = self.get_monitor_info()
            
            # Check if window is visible on any monitor
            for monitor in monitors:
                monitor_rect = monitor['Monitor']
                # Check for any overlap
                if (rect[0] < monitor_rect[2] and rect[2] > monitor_rect[0] and
                    rect[1] < monitor_rect[3] and rect[3] > monitor_rect[1]):
                    return True
                    
            return False
        except Exception as e:
            print(f"Error checking window visibility: {e}")
            return False
    
    def load_user_info(self):
        """Load user personal information from environment variables"""
        user_info = {
            "name": os.getenv("USER_NAME", ""),
            "job_title": os.getenv("USER_JOB_TITLE", ""),
            "company": os.getenv("USER_COMPANY", ""),
            "email": os.getenv("USER_EMAIL", ""),
            "team": os.getenv("USER_TEAM", ""),
            "department": os.getenv("USER_DEPARTMENT", ""),
            "phone": os.getenv("USER_PHONE", ""),
            # Add any other personal details you want to load
        }
        
        # Check if we have at least basic info
        if user_info["name"]:
            print(f"Loaded personal information for: {user_info['name']}")
        else:
            print("No personal user information found in .env file")
            
        return user_info
        
    def load_templates(self):
        """Load email templates from templates.json file"""
        try:
            # Get absolute path to templates file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            template_path = os.path.join(script_dir, 'templates.json')
            
            print(f"Looking for templates at: {template_path}")  # Debug line
            
            if os.path.exists(template_path):
                with open(template_path, 'r', encoding='utf-8') as f:
                    self.templates = json.load(f)
                    print(f"Loaded {len(self.templates)} templates from templates.json")
                    # Debug: print template names
                    print(f"Template names: {list(self.templates.keys())}")
            else:
                self.templates = {}
                print("No templates.json file found")
                
        except Exception as e:
            print(f"Error loading templates: {e}")
            self.templates = {}

    def show_template(self, template_name):
        """Copy template content to clipboard"""
        if template_name in self.templates:
            template_content = self.templates[template_name]
            
            # Copy to clipboard
            pyperclip.copy(template_content)
            
            # Show success message
            self.show_message(f"Template copied to clipboard: {template_name}", message_type="success")
        else:
            self.show_message(f"Template not found: {template_name}", message_type="error")

    def reload_templates(self):
        """Reload templates from file"""
        self.load_templates()
        self.show_message(f"Reloaded {len(self.templates)} templates", message_type="success")
        
        # Recreate the tray icon to update the menu
        if hasattr(self, 'tray_icon'):
            self.tray_icon.stop()
        self.setup_tray_icon()

    def process_with_llm(self, prompt, selected_text=""):
        """Process text with LLM via Groq API with personal information and templates"""
        try:
            # Build a user context with personal information
            user_context = ""
            if hasattr(self, 'user_info') and self.user_info.get("name"):
                user_context = "Personal information to use in your responses:\n"
                for key, value in self.user_info.items():
                    if value:  # Only include non-empty values
                        user_context += f"- {key.replace('_', ' ').title()}: {value}\n"
            
            # Check if the prompt asks for a template
            template_to_use = None
            explicit_template_request = False
            prompt_lower = prompt.lower()
            
            # First check for explicit template requests in both languages
            for template_key in self.templates.keys():
                # English patterns
                english_patterns = [
                    f"use {template_key} template",
                    f"use my {template_key} template",
                    f"insert {template_key} template",
                    f"insert my {template_key} template",
                    f"please insert my {template_key} template",
                    f"please use my {template_key} template"
                ]
                
                # Spanish patterns
                spanish_patterns = [
                    f"usa la plantilla {template_key}",
                    f"usa mi plantilla {template_key}",
                    f"inserta la plantilla {template_key}",
                    f"inserta mi plantilla {template_key}",
                    f"por favor inserta mi plantilla {template_key}",
                    f"por favor usa mi plantilla {template_key}",
                    f"utiliza la plantilla {template_key}",
                    f"utiliza mi plantilla {template_key}"
                ]
                
                # Check if any pattern matches
                for pattern in english_patterns + spanish_patterns:
                    if pattern in prompt_lower:
                        template_to_use = self.templates[template_key]
                        explicit_template_request = True
                        break
                
                if template_to_use:
                    break
            
            # If no explicit request, check for keywords
            if not template_to_use:
                keywords_to_templates = {
                    "case closure": "case_closure",
                    "cierre de caso": "case_closure",  # Spanish equivalent
                    "meeting request": "meeting_request",
                    "solicitud de reunión": "meeting_request",  # Spanish equivalent
                    "follow up": "follow_up",
                    "follow-up": "follow_up",
                    "seguimiento": "follow_up"  # Spanish equivalent
                }
                
                for keyword, template_key in keywords_to_templates.items():
                    if keyword in prompt_lower and template_key in self.templates:
                        template_to_use = self.templates[template_key]
                        break
            
            # Check if signature is explicitly requested
            signature_requested = any(term in prompt_lower for term in [
                "with signature", "with my signature", "add signature", 
                "include signature", "sign it", "add my signature"
            ])
            
            # Add system message with personal information and templates
            system_message = "You are a helpful AI assistant that processes user requests and modifies text. You understand and respond to requests in both English and Spanish. "
            
            if template_to_use:
                if explicit_template_request:
                    system_message += "The user has explicitly requested to use a specific template. "
                    system_message += f"MANDATORY: Use this template exactly as the base for your response:\n\n{template_to_use}\n\n"
                else:
                    system_message += f"IMPORTANT: Use this template as the base for your response:\n\n{template_to_use}\n\n"
                system_message += "Fill in the template variables with appropriate information from the user's request. "
            
            if user_context:
                system_message += "Use the provided personal information when creating content like emails, reports, or other documents. "
                
                if signature_requested:
                    system_message += "Include the user's signature at the end of the email since it was explicitly requested. "
                else:
                    system_message += "IMPORTANT: Do NOT include the user's signature, email address, or any closing with the user's name unless explicitly requested. End emails professionally but without any signature block. "
            
            system_message += "Be concise and only return the processed text without explanations unless specifically asked for commentary."
            
            # Construct the messages
            messages = [
                {
                    "role": "system", 
                    "content": system_message
                }
            ]
            
            # Add user context as a separate system message if available
            if user_context:
                messages.append({
                    "role": "system",
                    "content": user_context
                })
            
            # Add the user message
            if selected_text:
                messages.append({
                    "role": "user", 
                    "content": f"User request: {prompt}\n\nSelected text: {selected_text}"
                })
            else:
                messages.append({
                    "role": "user", 
                    "content": prompt
                })
            
            # Make the API request
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": LLM_MODEL,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 1024
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_response = result["choices"][0]["message"]["content"]
                return assistant_response
            else:
                error_msg = f"LLM request failed: {response.status_code}"
                print(f"{error_msg} - {response.text}")
                self.show_message(error_msg, message_type="error")
                return None
        except Exception as e:
            error_msg = f"Error during LLM processing: {str(e)}"
            print(error_msg)
            self.show_message(error_msg, message_type="error")
        
    def insert_text(self, text):
        """Insert text at cursor position with enhanced handling for Notepad"""
        # First, check if the target window is Notepad
        is_notepad = False
        try:
            window_class = win32gui.GetClassName(self.target_window)
            window_title = win32gui.GetWindowText(self.target_window)
            
            # Check if we're dealing with Notepad
            if window_class == "Notepad" or "Notepad" in window_title:
                is_notepad = True
                print("Detected Notepad window, using direct input method")
        except Exception as e:
            print(f"Error getting window info: {e}")
        
        # Make sure the target window is in focus
        if self.target_window:
            win32gui.SetForegroundWindow(self.target_window)
            time.sleep(0.2)  # Give time for window to come to foreground
        
        if is_notepad:
            # Skip the problematic character-by-character method and go straight to Windows API
            try:
                # Use the Windows API to send a WM_SETTEXT message
                import win32con
                
                # Find the Edit control in Notepad
                edit_handle = win32gui.FindWindowEx(self.target_window, 0, "Edit", None)
                if edit_handle:
                    print("Using Windows API to set text directly")
                    # Get current text to preserve it if needed
                    current_text = win32gui.GetWindowText(edit_handle)
                    
                    # Set the new text (append to existing)
                    win32gui.SendMessage(edit_handle, win32con.WM_SETTEXT, 0, current_text + text)
                    return
            except Exception as e:
                print(f"Windows API method failed: {e}")
        
        # For non-Notepad or if the Notepad methods failed, use standard clipboard method
        print("Using standard clipboard paste method")
        
        # Copy the new text to the clipboard
        pyperclip.copy(text)
        
        # Add a small delay to ensure clipboard is updated
        time.sleep(0.3)
        
        # Standard paste method
        try:
            # Paste the text using keyboard controller
            self.keyboard_controller.press(Key.ctrl)
            self.keyboard_controller.press('v')
            self.keyboard_controller.release('v')
            self.keyboard_controller.release(Key.ctrl)
        except Exception as e:
            print(f"Error during paste: {e}")
            # Try fallback method
            try:
                keyboard.press_and_release('ctrl+v')
            except Exception as e2:
                print(f"Fallback paste also failed: {e2}")
    
    def paraphrase_selected_text(self):
        """Paraphrase the currently selected text"""
        if self.processing:
            return
            
        # Store the current active window
        self.target_window = win32gui.GetForegroundWindow()
        print(f"Paraphrasing from window handle: {self.target_window}")
        
        # Get the selected text
        selected_text = self.get_selected_text()
        
        if not selected_text or not selected_text.strip():
            self.show_message("No text selected", message_type="error")
            return
        
        # Show we're processing
        self.processing = True
        self.show_message("Paraphrasing...", message_type="processing")
        
        # Get the current style
        style = PARAPHRASING_STYLES[self.paraphrase_style_index]
        
        try:
            # Call the LLM for paraphrasing
            messages = [
                {
                    "role": "system",
                    "content": style["system_prompt"]
                },
                {
                    "role": "user",
                    "content": selected_text
                }
            ]
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": LLM_MODEL,
                    "messages": messages,
                    "temperature": style["temperature"],
                    "max_tokens": 4096
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                paraphrased_text = result["choices"][0]["message"]["content"].strip()
                
                # Make sure the target window is in focus before pasting
                if self.target_window:
                    try:
                        win32gui.SetForegroundWindow(self.target_window)
                        time.sleep(0.3)  # Give window time to come to foreground
                    except Exception as e:
                        print(f"Error setting foreground window for paste: {e}")
                
                # Insert the paraphrased text
                pyperclip.copy(paraphrased_text)
                time.sleep(0.3)  # Increased wait time
                
                # Try two paste methods for better reliability
                try:
                    # Method 1: Using pynput
                    self.keyboard_controller.press(Key.ctrl)
                    self.keyboard_controller.press('v')
                    self.keyboard_controller.release('v')
                    self.keyboard_controller.release(Key.ctrl)
                except Exception as e:
                    print(f"First paste method failed: {e}")
                    # Method 2: Using keyboard library
                    try:
                        keyboard.press_and_release('ctrl+v')
                    except Exception as e2:
                        print(f"Second paste method also failed: {e2}")
                
                self.show_message(f"Paraphrased using {style['name']} style", message_type="success")
            else:
                self.show_message(f"Paraphrasing failed: {response.status_code}", message_type="error")
        
        except Exception as e:
            self.show_message(f"Error: {str(e)}", message_type="error")
        
        finally:
            self.processing = False
            
    def _process_text_to_speech(self, text):
        """Process text-to-speech request and play audio with improved ESC key handling"""
        try:
            # Make request to Groq API
            response = requests.post(
                "https://api.groq.com/openai/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": TTS_MODEL,
                    "voice": TTS_VOICE,
                    "input": text,
                    "response_format": TTS_RESPONSE_FORMAT
                }
            )
            
            if response.status_code == 200:
                # Show playing message
                self.show_message("Playing audio... (press ESC to stop)", message_type="playing_audio")
                
                # Load audio data from response
                audio_data = io.BytesIO(response.content)
                
                # Create the audio segment
                audio_segment = AudioSegment.from_file(audio_data, format=TTS_RESPONSE_FORMAT)
                
                # Variable to track if playback should be stopped
                self.playing_audio = True
                self.playback_stopped_by_user = False
                
                # Create an event to signal stopping between threads
                stop_event = threading.Event()
                
                # Use PyAudio directly to play the audio
                def play_audio_with_pyaudio():
                    try:
                        # Convert AudioSegment to raw PCM data
                        raw_data = audio_segment.raw_data
                        sample_width = audio_segment.sample_width
                        frame_rate = audio_segment.frame_rate
                        channels = audio_segment.channels
                        
                        # Initialize PyAudio for playback
                        p = pyaudio.PyAudio()
                        
                        # Open stream
                        stream = p.open(
                            format=p.get_format_from_width(sample_width),
                            channels=channels,
                            rate=frame_rate,
                            output=True
                        )
                        
                        # Break the audio into chunks to enable stopping
                        chunk_size = 1024  # size of each chunk in frames
                        
                        # Create chunks of audio data
                        chunks = [raw_data[i:i+chunk_size*sample_width*channels] 
                                  for i in range(0, len(raw_data), chunk_size*sample_width*channels)]
                        
                        # Play each chunk, checking for stop signal between chunks
                        for chunk in chunks:
                            if stop_event.is_set() or not self.playing_audio:
                                print("Stop signal received in audio playback thread")
                                break
                            stream.write(chunk)
                        
                        # Close stream
                        stream.stop_stream()
                        stream.close()
                        p.terminate()
                        
                    except Exception as e:
                        print(f"Error in audio playback: {e}")
                    finally:
                        self.playing_audio = False
                        
                        # Show completion message if not stopped by user
                        if not self.playback_stopped_by_user:
                            # We're in a different thread, so use after() to update UI from main thread
                            self.root.after(100, lambda: self.show_message("Audio playback complete", message_type="success"))
                
                # Register a direct keyboard hook for ESC specifically for stopping audio
                def on_esc_pressed(e):
                    if e.name == 'esc' and self.playing_audio:
                        print("ESC pressed, stopping playback")
                        stop_event.set()
                        self.playing_audio = False
                        self.playback_stopped_by_user = True
                        # Update UI from main thread
                        self.root.after(100, lambda: self.show_message("Audio playback stopped", message_type="success"))
                
                # Add ESC key hook specifically for audio stopping
                esc_hook = keyboard.on_press(on_esc_pressed)
                
                # Run audio playback in a separate thread
                audio_thread = threading.Thread(target=play_audio_with_pyaudio)
                audio_thread.daemon = True
                audio_thread.start()
                
                # Wait for playback to complete or to be stopped
                # Limited wait to avoid blocking indefinitely
                max_wait = 120  # Maximum wait time in seconds
                start_time = time.time()
                
                while self.playing_audio and time.time() - start_time < max_wait:
                    time.sleep(0.1)
                    
                # Clean up hook after playback finishes
                keyboard.unhook(esc_hook)
                
                # Make sure we stop playback if we exit the wait loop
                if self.playing_audio:
                    stop_event.set()
                    self.playing_audio = False
                    
            else:
                error_msg = f"TTS request failed: {response.status_code}"
                print(f"{error_msg} - {response.text}")
                self.show_message(error_msg, message_type="error")
        
        except Exception as e:
            error_msg = f"Error during TTS processing: {str(e)}"
            print(error_msg)
            self.show_message(error_msg, message_type="error")
        
        finally:
            self.processing = False    
    
    def set_tts_voice(self, voice):
        """Set the TTS voice to use"""
        global TTS_VOICE
        TTS_VOICE = voice
        print(f"TTS voice set to: {voice}")
        self.show_message(f"TTS voice set to: {voice}", message_type="success")
        
        # Try to save the selected voice to .env file
        try:
            # Get absolute path to .env file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            env_path = os.path.join(script_dir, '.env')
            
            # Check if .env file exists and is writable
            if os.path.exists(env_path) and os.access(env_path, os.W_OK):
                set_key(env_path, 'TTS_VOICE', voice)
            else:
                print(f"Cannot write to .env file at {env_path}")
                # Try alternative method: write to environment variable in memory
                os.environ['TTS_VOICE'] = voice
        except Exception as e:
            print(f"Error saving TTS voice setting: {e}")
    
    def exit_app(self):
        """Exit the application cleanly"""
        try:
            # Stop the tray icon
            if hasattr(self, 'tray_icon'):
                self.tray_icon.stop()
            
            # Unregister hotkeys
            keyboard.unhook_all()
            
            # Destroy Tkinter root if it exists
            if hasattr(self, 'root'):
                self.root.quit()
                self.root.destroy()
            
            # Exit the application
            os._exit(0)
        except Exception as e:
            print(f"Error during shutdown: {e}")
            os._exit(1)
    
    def run(self):
        """Run the application"""
        try:
            # Run the main Tkinter loop (must be on main thread)
            self.root.mainloop()
        except KeyboardInterrupt:
            self.exit_app()

if __name__ == "__main__":
    assistant = UnifiedAssistant()
    
    # You can configure message timeouts here if needed
    # Example: Change error message timeout to 5 seconds
    # assistant.configure_message_timeout("error", 5000)
    
    try:
        assistant.run()
    except Exception as e:
        print(f"Error in main application: {e}")
        sys.exit(1)
