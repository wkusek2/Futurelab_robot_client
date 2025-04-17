import customtkinter as ctk
import tkinter as ttk
from tkinter import ttk

class button:
    def __init__(self, window, text_var=None, textvariable_var=None, command_var=None, row=0, column=0, padx=0, pady=0, sticky='nsew', color=None):
        # Inicjalizacja parametrów
        self.window = window
        self.text = text_var
        self.textvariable = textvariable_var
        self.command = command_var
        self.row = row
        self.column = column
        self.padx = padx
        self.pady = pady
        self.sticky = sticky
        self.color = color

        # Tworzenie przycisku
        self.create_button()

    def create_button(self):
        # Ustawienia przycisku
        button_config = {
            'command': self.command,
            'fg_color': self.color,
            'padx': self.padx,
            'pady': self.pady,
            'sticky': self.sticky
        }

        # Tworzenie przycisku na podstawie dostępnych atrybutów
        if self.text and self.color:
            self.button = ctk.CTkButton(self.window, text=self.text, **button_config)
        elif self.textvariable and self.color:
            self.button = ctk.CTkButton(self.window, textvariable=self.textvariable, **button_config)
        elif self.text:
            self.button = ctk.CTkButton(self.window, text=self.text, command=self.command)
        elif self.textvariable:
            self.button = ctk.CTkButton(self.window, textvariable=self.textvariable, command=self.command)

        # Umieszczanie przycisku w oknie
        self.button.grid(row=self.row, column=self.column, padx=self.padx, pady=self.pady, sticky=self.sticky)

    def change_color(self, new_color):
        self.color = new_color
        self.button.configure(fg_color=self.color)

class text_label:
    def __init__(self, window, text, font, font_size, row, column, padx, pady, sticky):
        self.window = window
        self.text = text
        self.font = font
        self.font_size = font_size
        self.row = row
        self.column = column
        self.padx = padx
        self.pady = pady
        self.sticky = sticky

        self.create_label()

    def create_label(self):
        label_config = {
            'row': self.row,
            'column': self.column,
            'padx': self.padx,
            'pady': self.pady,
            'sticky': self.sticky
        }

        self.label = ctk.CTkLabel(self.window, text=self.text,font=(self.font,self.font_size))
        self.label.grid(**label_config)

    def configure(self, text):
        self.label.configure(text=text)

class text_gap:
    def __init__(self, window, width, row, column, padx, pady, sticky):
        self.window = window
        self.width = width

        self.row = row
        self.column = column
        self.padx = padx
        self.pady = pady
        self.sticky = sticky

        self.create_gap()

    def create_gap(self):

        new_text_gap_config = {
            'row': self.row,
            'column': self.column,
            'padx': self.padx,
            'pady': self.pady,
            'sticky': self.sticky
        }

        self.new_text_gap = ctk.CTkEntry(self.window, width=self.width)
        self.new_text_gap.grid(**new_text_gap_config)

    def get(self):
        number = 0
        try:
            value = self.new_text_gap.get()
            number = float(value)
        except ValueError:
            print("Bład konwersji")
        return number

class dropdown_list:
    def __init__(self, window, variable, sequence, command, row, column, padx, pady):
        self.window = window
        self.variable = variable
        self.sequence = sequence
        self.command = command
        self.row = row
        self.column = column
        self.padx = padx
        self.pady = pady

        self.create_dropdown_list()

    def create_dropdown_list(self):
        self.dropdown_list = ttk.Combobox(self.window, textvariable=self.variable)
        self.dropdown_list.bind(self.sequence, self.command)
        self.dropdown_list.grid(row=self.row, column=self.column, padx=self.padx, pady=self.pady)

    def configure(self, variable):
        self.dropdown_list.configure(values=variable)

    def set(self, variable):
        self.dropdown_list.set(variable)

    def get(self):
        return self.dropdown_list.get()


class table:
    def __init__(self, window, columns, show, texts, width, row, column, padx, pady, sticky):
        self.window = window
        self.columns = columns
        self.show = show
        self.texts = texts
        self.width = width
        self.row = row
        self.column = column
        self.padx = padx
        self.pady = pady
        self.sticky = sticky
        self.rows = {}  # Słownik do przechowywania identyfikatorów wierszy

        self.create_table()

        self.style = ttk.Style()
        self.style.configure("Treeview", font=("Arial", 18), rowheight=40)  # Zmiana rozmiaru czcionki dla wierszy
        self.style.configure("Treeview.Heading", font=("Arial", 16))  # Zmiana rozmiaru czcionki dla nagłówków
        self.style

    def create_table(self):
        self.table = ttk.Treeview(self.window, columns=self.columns, show=self.show)
        for col, text in zip(self.columns, self.texts):
            self.table.heading(col, text=text)
            self.table.column(col, width=self.width)
        self.table.grid(row=self.row, column=self.column, padx=self.padx, pady=self.pady, sticky=self.sticky)

    def get_children(self):
        return self.table.get_children()

    def item(self, variable, values):
        self.table.item(variable, values)

    def insert_or_update(self, id, values):
        if id in self.rows:
            self.table.item(self.rows[id], values=values)
        else:
            new_row = self.table.insert("", "end", values=values)
            self.rows[id] = new_row

class slider:
    selected_slider = None  # Zmienna klasowa do przechowywania wybranego suwaka

    def __init__(self, window, row, column, min, max, padx, pady, id_number, text, move_function, com=None, cam=None, value=None):
        self.window = window
        self.row = row
        self.column = column
        self.min = min
        self.max = max
        self.padx = padx
        self.pady = pady
        self.id_number = id_number
        self.text = text
        self.move_function = move_function

        # Utworzenie etykiety wyświetlającej wartość slidera
        self.label_min = ctk.CTkLabel(master=self.window, text=str(self.min))
        self.label_min.grid(row=row, column=column, padx=30, pady=0, sticky="NW")
        self.label_max = ctk.CTkLabel(master=self.window, text=str(self.max))
        self.label_max.grid(row=row, column=column, padx=0, pady=0, sticky="NE")
        # Utworzenie slidera
        self.label_value = ctk.CTkLabel(master=self.window, text=f"ID: {self.id_number} Pos: 0")
        self.label_value.grid(row=row, column=column, padx=0, pady=0, sticky="S")

        self.slider = ctk.CTkSlider(master=self.window, from_=self.min, to=self.max)
        self.slider.grid(row=self.row, column=self.column, padx=30, pady=20)

        # Wiązanie zdarzeń klawiaturowych i myszy
        self.slider.bind('<Button-1>', self.select_slider)
        self.window.bind('<Left>', self.move_slider_left)
        self.window.bind('<Right>', self.move_slider_right)

        # Aktualizacja etykiety przy zmianie wartości suwaka
        self.slider.bind('<Motion>', self.update_label_from_slider)
        self.slider.bind('<ButtonRelease-1>', command=lambda event: self.on_slider_value_changed(com, cam))  # Wywołanie po zwolnieniu przycisku

        if value != None:
            self.slider.set(value)

    def select_slider(self, event):
        slider.selected_slider = self
        self.update_label_from_slider(None)

    def move_slider_left(self, event):
        if slider.selected_slider:
            current_value = slider.selected_slider.slider.get()
            new_value = max(current_value - 50, slider.selected_slider.min)
            slider.selected_slider.slider.set(new_value)
            slider.selected_slider.update_label_from_slider(None)

    def move_slider_right(self, event):
        if slider.selected_slider:
            current_value = slider.selected_slider.slider.get()
            new_value = min(current_value + 50, slider.selected_slider.max)
            slider.selected_slider.slider.set(new_value)
            slider.selected_slider.update_label_from_slider(None)

    def update_label_from_slider(self, event):
        value = self.slider.get()
        self.update_label(value)

    def update_label(self, value):
        self.label_value.configure(text=f"ID: {self.id_number} {self.text} {int(float(value))}")

    def on_slider_value_changed(self, com, cam):
        if self.move_function == 1 and com != None:
            if slider.selected_slider:
                id_number = slider.selected_slider.id_number
                position = slider.selected_slider.slider.get()
                com.move(id_number, position)
        if self.move_function == 2 and cam != None:
            if slider.selected_slider:
                id_number = slider.selected_slider.id_number
                position = slider.selected_slider.slider.get() 
                cam.change_param(id_number, position)      

    def get(self):
        return self.slider.get()
    
    def set(self, slider_value):
        self.slider.set(slider_value)
