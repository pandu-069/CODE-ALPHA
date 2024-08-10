from deep_translator import GoogleTranslator
import tkinter as tk
from tkinter import scrolledtext


def translate_text(text, dest_language):
    # Translate the text
    translated = GoogleTranslator(source='auto', target=dest_language).translate(text)
    return translated

def display_text_in_frame(text):
    # Create a new Tkinter window
    window = tk.Tk()
    window.title("Translated Text")

    # Create a ScrolledText widget
    text_area = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=60, height=20, font=("Helvetica", 14), bg="#f0f0f0")
    text_area.pack(padx=10, pady=10)

    # Configure tags for different styles
    text_area.tag_configure("colored_text", foreground="blue")
    text_area.tag_configure("large_text", font=("Helvetica", 14, "bold"))

    # Insert the text into the ScrolledText widget with the configured tags
    text_area.insert(tk.END, text, ("colored_text", "large_text"))
    
    # Make the text read-only
    text_area.config(state=tk.DISABLED)
    
    # Run the Tkinter event loop
    window.mainloop()
    

if __name__ == "__main__":
    text = input("Enter the text to be translated: ")
    lang = input("Enter the name of the language to be translated: ")
    
    translated_text = translate_text(text,lang)
    
    display_text_in_frame("Original-Text: "+text+"\n\n" +"Language-Name: "+lang+ "\n\n Translated-Text: " + translated_text)