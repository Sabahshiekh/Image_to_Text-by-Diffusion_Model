import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import threading

class ImageCaptioningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Captioning Tool")
        self.root.geometry("800x600")

        # Initialize the image captioning model and processor
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        self.images = []
        self.captions = {}
        self.row_count = 0  # To track the row in the grid layout
        self.col_count = 0  # To track the column in the grid layout

        self.upload_button = tk.Button(
            self.root, text="Upload Images", command=self.upload_images, font=("Arial", 14)
        )
        self.upload_button.pack(pady=20)

        # Image frame will now use a grid layout
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(fill=tk.BOTH, expand=True)

    def upload_images(self):
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
        )
        for file_path in file_paths:
            if file_path not in self.images:
                self.images.append(file_path)
                # Use threading to avoid blocking the main UI thread
                threading.Thread(target=self.display_image, args=(file_path,)).start()

    def display_image(self, file_path):
        # Load and resize the image (resize before processing)
        img = Image.open(file_path)
        img_resized = img.resize((150, 150))

        # Create a frame for each image with a delete button
        frame = tk.Frame(self.image_frame, borderwidth=2, relief="groove")
        
        # Add the frame to the grid layout
        frame.grid(row=self.row_count, column=self.col_count, padx=10, pady=10)

        # Display the resized image
        img_tk = ImageTk.PhotoImage(img_resized)
        img_label = tk.Label(frame, image=img_tk)
        img_label.image = img_tk  # Keep a reference to avoid garbage collection
        img_label.pack()

        # Generate caption using the captioning model in a separate thread
        caption = self.generate_caption(file_path)

        # Display the caption
        caption_label = tk.Label(frame, text=caption, wraplength=150, font=("Arial", 10))
        caption_label.pack(pady=5)

        # Add a delete button for removing the image
        delete_button = tk.Button(frame, text="Delete", command=lambda: self.delete_image(file_path, frame))
        delete_button.pack(pady=5)

        # Update grid layout to move to the next column
        self.col_count += 1
        if self.col_count > 5:  # Adjust this value to control number of columns per row
            self.col_count = 0
            self.row_count += 1
            

    def generate_caption(self, file_path):
        # Load image and prepare it for the model (resize for efficiency)
        img = Image.open(file_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")

        # Generate caption (run model inference)
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)

        return caption  # Return the generated caption

    def delete_image(self, file_path, frame):
        # Remove the image from the list and the frame
        if file_path in self.images:
            self.images.remove(file_path)

        # Destroy the frame (this removes the image and its delete button)
        frame.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCaptioningApp(root)
    root.mainloop()
