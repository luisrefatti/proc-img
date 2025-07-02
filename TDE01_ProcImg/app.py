import customtkinter as ctk
from tkinter import Canvas


def normalize_rgb(r, g, b):
    return (r / 255.0, g / 255.0, b / 255.0)


def rgb_to_hsv(r, g, b):
    r_norm, g_norm, b_norm = normalize_rgb(r, g, b)
    cmax = max(r_norm, g_norm, b_norm)
    cmin = min(r_norm, g_norm, b_norm)
    delta = cmax - cmin

    if delta == 0:
        h = 0.0
    else:
        if cmax == r_norm:
            h = ((g_norm - b_norm) / delta) % 6
        elif cmax == g_norm:
            h = (b_norm - r_norm) / delta + 2
        else:
            h = (r_norm - g_norm) / delta + 4
        h *= 60
        if h < 0:
            h += 360

    s = (delta / cmax * 100) if cmax != 0 else 0.0
    v = cmax * 100
    return (round(h, 2), round(s, 2), round(v, 2))


def hsv_to_rgb(h, s, v):
    h = h % 360
    s /= 100.0
    v /= 100.0

    if s == 0:
        r = g = b = v
    else:
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c

        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        r += m
        g += m
        b += m

    r = max(0, min(1, r)) * 255
    g = max(0, min(1, g)) * 255
    b = max(0, min(1, b)) * 255
    return (round(r), round(g), round(b))


def rgb_to_cmyk(r, g, b):
    r_norm, g_norm, b_norm = normalize_rgb(r, g, b)
    k = 1 - max(r_norm, g_norm, b_norm)

    if k == 1:
        return (0.0, 0.0, 0.0, 100.0)

    c = (1 - r_norm - k) / (1 - k)
    m = (1 - g_norm - k) / (1 - k)
    y = (1 - b_norm - k) / (1 - k)

    return (
        round(c * 100, 2),
        round(m * 100, 2),
        round(y * 100, 2),
        round(k * 100, 2)
    )


def cmyk_to_rgb(c, m, y, k):
    c = max(0.0, min(100.0, c)) / 100.0
    m = max(0.0, min(100.0, m)) / 100.0
    y = max(0.0, min(100.0, y)) / 100.0
    k = max(0.0, min(100.0, k)) / 100.0

    r = 255 * (1 - c) * (1 - k)
    g = 255 * (1 - m) * (1 - k)
    b = 255 * (1 - y) * (1 - k)

    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    return (round(r), round(g), round(b))


def rgb_to_grayscale(r, g, b):
    return round(0.299 * r + 0.587 * g + 0.114 * b)


class ColorConverterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Conversor de Cores")
        self.geometry("750x450")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.main_frame = ctk.CTkFrame(self, corner_radius=15)
        self.main_frame.pack(padx=20, pady=20, fill="both", expand=True)

        self.rgb_entries = []
        self.hsv_entries = []
        self.cmyk_entries = []

        self.create_widgets()
        self.update_color_preview()

    def create_widgets(self):
        ctk.CTkLabel(self.main_frame, text="RGB (0-255)").grid(row=0,
                                                               column=0, columnspan=3, pady=5)
        for i in range(3):
            entry = ctk.CTkEntry(self.main_frame, width=80,
                                 placeholder_text="0")
            entry.grid(row=1, column=i, padx=5)
            self.rgb_entries.append(entry)

        ctk.CTkButton(self.main_frame, text="→ HSV", width=80,
                      command=self.convert_rgb_to_hsv).grid(row=1, column=3, padx=5)
        ctk.CTkButton(self.main_frame, text="→ CMYK", width=80,
                      command=self.convert_rgb_to_cmyk).grid(row=1, column=4, padx=5)

        ctk.CTkLabel(self.main_frame, text="HSV (H:0-360 S/V:0-100)").grid(row=2,
                                                                           column=0, columnspan=3, pady=10)
        for i in range(3):
            entry = ctk.CTkEntry(self.main_frame, width=80,
                                 placeholder_text="0")
            entry.grid(row=3, column=i, padx=5)
            self.hsv_entries.append(entry)
        ctk.CTkButton(self.main_frame, text="→ RGB", width=80,
                      command=self.convert_hsv_to_rgb).grid(row=3, column=3, padx=5)

        ctk.CTkLabel(self.main_frame, text="CMYK (0-100%)").grid(row=4,
                                                                 column=0, columnspan=3, pady=10)
        for i in range(4):
            entry = ctk.CTkEntry(self.main_frame, width=80,
                                 placeholder_text="0")
            entry.grid(row=5, column=i, padx=5)
            self.cmyk_entries.append(entry)
        ctk.CTkButton(self.main_frame, text="→ RGB", width=80,
                      command=self.convert_cmyk_to_rgb).grid(row=5, column=4, padx=5)

        ctk.CTkLabel(self.main_frame, text="Escala de Cinza").grid(
            row=6, column=0, pady=10)
        self.grayscale_entry = ctk.CTkEntry(self.main_frame, width=80)
        self.grayscale_entry.grid(row=6, column=1, padx=5)
        ctk.CTkButton(self.main_frame, text="Calcular", width=80,
                      command=self.convert_to_grayscale).grid(row=6, column=2, padx=5)

        self.color_canvas = Canvas(
            self.main_frame, width=120, height=120, bg='white', bd=0, highlightthickness=0)
        self.color_canvas.grid(row=0, column=5, rowspan=6, padx=20, pady=10)

        ctk.CTkButton(self.main_frame, text="Limpar Tudo", command=self.clear_all).grid(
            row=8, column=0, columnspan=5, pady=15)

    def get_rgb(self):
        try:
            return [int(entry.get()) for entry in self.rgb_entries]
        except:
            return [0, 0, 0]

    def update_color_preview(self):
        r, g, b = self.get_rgb()
        hex_color = f'#{r:02x}{g:02x}{b:02x}'
        self.color_canvas.configure(bg=hex_color)

    def convert_rgb_to_hsv(self):
        r, g, b = self.get_rgb()
        h, s, v = rgb_to_hsv(r, g, b)
        for i, entry in enumerate(self.hsv_entries):
            entry.delete(0, "end")
            entry.insert(0, f"{[h, s, v][i]:.2f}")
        self.update_color_preview()

    def convert_hsv_to_rgb(self):
        try:
            h = float(self.hsv_entries[0].get())
            s = float(self.hsv_entries[1].get())
            v = float(self.hsv_entries[2].get())
            r, g, b = hsv_to_rgb(h, s, v)
            for i, entry in enumerate(self.rgb_entries):
                entry.delete(0, "end")
                entry.insert(0, str(rgb))
            self.update_color_preview()
        except:
            pass

    def convert_rgb_to_cmyk(self):
        r, g, b = self.get_rgb()
        c, m, y, k = rgb_to_cmyk(r, g, b)
        for i, entry in enumerate(self.cmyk_entries):
            entry.delete(0, "end")
            entry.insert(0, f"{[c, m, y, k][i]:.2f}")
        self.update_color_preview()

    def convert_cmyk_to_rgb(self):
        try:
            c = float(self.cmyk_entries[0].get())
            m = float(self.cmyk_entries[1].get())
            y = float(self.cmyk_entries[2].get())
            k = float(self.cmyk_entries[3].get())
            r, g, b = cmyk_to_rgb(c, m, y, k)
            for i, entry in enumerate(self.rgb_entries):
                entry.delete(0, "end")
                entry.insert(0, str(rgb))
            self.update_color_preview()
        except:
            pass

    def convert_to_grayscale(self):
        r, g, b = self.get_rgb()
        gray = rgb_to_grayscale(r, g, b)
        self.grayscale_entry.delete(0, "end")
        self.grayscale_entry.insert(0, str(gray))

    def clear_all(self):
        for entry in self.rgb_entries + self.hsv_entries + self.cmyk_entries:
            entry.delete(0, "end")
            entry.insert(0, "0")
        self.grayscale_entry.delete(0, "end")
        self.grayscale_entry.insert(0, "0")
        self.color_canvas.configure(bg='white')


if __name__ == "__main__":
    app = ColorConverterApp()
    app.mainloop()
