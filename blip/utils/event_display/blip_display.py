"""

"""
from ctypes import sizeof
import uproot
import os
import numpy as np
import socket
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats as st
from datetime import datetime
import tkinter as tk
import sv_ttk
import customtkinter
import tkinter.messagebox
from PIL import ImageTk, Image

from blip.utils.logger import Logger
from blip.dataset.common import *
from blip.dataset.blip import BlipDataset

customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("blue")

class BlipDisplay(customtkinter.CTk):
    """
    """
    def __init__(self,
        dataset
    ):
        super().__init__()
        self.title("BLIP Event Display")
        self.geometry(f"{1100}x{580}")

        self.dataset = dataset

        # self.display = tk.Tk()
        # self.canvas = tk.Canvas(self.display, width=2000, height=1200)

        #self.canvas.grid(columnspan=5, rowspan=5)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure((1,4), weight=3)
        self.grid_columnconfigure((2,3), weight=5)

        # create the sidebar
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=5, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)
        # self.logo_label = customtkinter.CTkLabel(
        #     self.sidebar_frame, text="BLIP", 
        #     font=customtkinter.CTkFont(size=20,weight="bold")
        # )
        self.logo_image = customtkinter.CTkImage(
            Image.open(f"{os.path.dirname(__file__)}/neutrino.png"),
            size=(200,200)
        )
        self.logo_label = customtkinter.CTkLabel(
            self.sidebar_frame, image=self.logo_image,
            text=""
        )
        self.logo_label.grid(row=0,column=0,padx=20,pady=(20,10))

        combobox_var = customtkinter.StringVar(value="option 2")
        combobox = customtkinter.CTkComboBox(self.sidebar_frame, values=["option 1", "option 2"],
                                            command=self.combobox_callback, variable=combobox_var)
        combobox_var.set("option 2")
        combobox.grid(column=0, row=1)

    def combobox_callback(self, choice):
        print("combobox dropdown clicked:", choice)




        # # logo and info
        # self.logo = ImageTk.PhotoImage(Image.open(f"{os.path.dirname(__file__)}/neutrino.png"))
        # self.logo_image = tk.Label(self.display, image=self.logo)
        # self.logo_image.image = self.logo
        # self.logo_image.grid(column=0, row=0, sticky="w", padx=2)
        # self.logo_text = tk.Label(self.display, text="BLIP Event Display", font=("Helvetica", 18, "bold"))
        # self.logo_text.grid(column=1, row=0, sticky="w")

        # sv_ttk.set_theme("dark")

        # self.display.title("BLIP Display")
        # #self.display.geometry("1000x800")
        # self.plot_button = tk.Button(
        #     self.display, text="Plot", command=self.plot
        # )
        # self.plot_button.grid(row=5, column=2)
        
        
        #self.buttons[button].pack()

    def plot(self):
        figure = plt.figure(figsize=(5,4), dpi=100)
        figure.add_subplot(111)

        chart = FigureCanvasTkAgg(figure, self.display)
        chart.get_tk_widget().grid(rowspan=3, row=1, column=2)        
    
        plt.grid()
        axes = plt.axes()
        return None

    def show(self):
        self.mainloop()
        return




