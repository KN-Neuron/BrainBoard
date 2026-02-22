import math
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from simulation import Simulation


def draw_keyboard(combination, chars):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#d3d3d3')
    ax.set_aspect('equal')
    ax.axis('off')

    R = 1.0
    D = math.sqrt(3) * R

    hex_centers = []
    for i in range(6):
        angle = math.radians(90 - i * 60)
        cx = D * math.cos(angle)
        cy = D * math.sin(angle)
        hex_centers.append((cx, cy))

    center_hex = RegularPolygon((0, 0), numVertices=6, radius=R, orientation=math.radians(30),
                                facecolor='#d3d3d3', edgecolor='#5c637a', linewidth=3)
    ax.add_patch(center_hex)

    text_offset = 0.65

    for i in range(6):
        cx, cy = hex_centers[i]

        hexagon = RegularPolygon((cx, cy), numVertices=6, radius=R, orientation=math.radians(30),
                                 facecolor='#d3d3d3', edgecolor='#5c637a', linewidth=3)
        ax.add_patch(hexagon)

        for j in range(5):
            index = i * 5 + j
            if index >= len(combination): break

            char_idx = combination[index]
            letter = chars[char_idx]
            display_letter = '_' if letter == ' ' else letter

            text_angle = math.radians(210 - i * 60 - j * 60)

            tx = cx + text_offset * math.cos(text_angle)
            ty = cy + text_offset * math.sin(text_angle)

            ax.text(tx, ty, display_letter, ha='center', va='center', fontsize=18,
                    fontweight='bold', color='#1a1c23')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    plt.tight_layout()
    plt.show()

chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '.', ',', '!', ' ']

def dict_init(combination):
    dictionary = {}
    counter = 0
    for i in range(6):
        for j in range(5):
            dictionary[chars[combination[counter]]] = (i-2, j-2)
            counter += 1
    return dictionary

test_string = (
    "LITWO OJCZYZNO MOJA! TY JESTES JAK ZDROWIE. "
    "ILE CIE TRZEBA CENIC, TEN TYLKO SIE DOWIE, "
    "KTO CIE STRACIL. DZIS PIEKNOSC TWA W CALEJ OZDOBIE "
    "WIDZE I OPISUJE, BO TESKNIE PO TOBIE. "
    "PANNO SWIETA, CO JASNIEJ CZESTOCHOWY BRONISZ "
    "I W OSTREJ SWIECISZ BRAMIE! TY, CO GROD ZAMKOWY "
    "NOWOGRODZKI OCHRANIASZ Z JEGO WIERNYM LUDEM! "
    "JAK MNIE DZIECKIEM DO ZDROWIA POWROCILAS CUDEM, "
    "GDY OD PLACZACEJ MATKI POD TWOJA OPIEKE "
    "OFIAROWANY, MARTWA PODNIOSLEM POWIEKE "
    "I ZARAZ MOGLEM PIESZO DO TWECH SWIATYN PROGU "
    "ISC ZA WROCONE ZYCIE PODZIEKOWAC BOGU, "
    "TAK NAS POWROCISZ CUDEM NA OJCZYZNY LONO. "
    "TYMCZASEM PRZENOS MOJA DUSZE UTESKNIONA "
    "DO TYCH PAGORKOW LESNYCH, DO TYCH LAK ZIELONYCH, "
    "SZEROKO NAD BLEKITNYM NIEMNEM ROZCIAGNIONYCH. "
    "DO TYCH POL MALOWANYCH ZBOZEM ROZMAITEM, "
    "POZLACANYCH PSZENICA, POSREBRZANYCH ZYTEM. "
    "GDZIE BURSZYTYNOWY SWIERZOP, GRYKA JAK SNIEG BIALA, "
    "GDZIE PANIENSKIM RUMIENCEM DZIECIELINA PALA, "
    "A WSZYSTKO PRZEPASANE, JAKBY WREBEM, MIEDZA "
    "ZIELONA, NA NIEJ Z RZADKA CICHE GRUSZE SIEDZA."
)
sim = Simulation(300, 5000, 0.85, 0.2 , chars, test_string)
sim.run()
best_instance = sim.find_best()

print(str(best_instance.combination) + " = " + str(best_instance.fitness))
draw_keyboard(best_instance.combination, chars)