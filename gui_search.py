import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import subprocess
from search import search, ft_model


class PhotoSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Photo Finder")
        self.root.geometry("900x700")

        self.top_frame = ttk.Frame(root, padding="10")
        self.top_frame.pack(fill=tk.X)

        self.query_var = tk.StringVar()
        self.entry = ttk.Entry(self.top_frame, textvariable=self.query_var,
                               font=("Arial", 14))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.entry.bind("<Return>", lambda e: self.perform_search())

        self.btn = ttk.Button(self.top_frame, text="Найти",
                              command=self.perform_search)
        self.btn.pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(root)
        self.scrollbar = ttk.Scrollbar(root, orient="vertical",
                                       command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame,
                                  anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.thumbnails = []

    def perform_search(self):
        query = self.query_var.get()
        if not query: return

        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.thumbnails = []

        results = search(query, ft_model, top_k=20)

        for i, (path, score) in enumerate(results):
            self.add_thumbnail(path, score, i)

    def add_thumbnail(self, path, score, index):
        try:
            img = Image.open(path)
            img.thumbnail((200, 200))
            photo = ImageTk.PhotoImage(img)
            self.thumbnails.append(photo)

            frame = ttk.Frame(self.scrollable_frame, padding="5")
            frame.grid(row=index // 3, column=index % 3, sticky="nw")

            lbl_img = tk.Label(frame, image=photo, cursor="hand2")
            lbl_img.pack()
            lbl_img.bind("<Double-Button-1>",
                         lambda e, p=path: self.open_file(p))

            short_name = os.path.basename(path)
            lbl_text = tk.Label(frame, text=f"{short_name}\nSim: {score:.2f}",
                                font=("Arial", 9))
            lbl_text.pack()

        except Exception as e:
            print(f"Не удалось загрузить {path}: {e}")

    def open_file(self, path):
        subprocess.run(['xdg-open', path])


if __name__ == "__main__":
    root = tk.Tk()
    app = PhotoSearchApp(root)
    root.mainloop()