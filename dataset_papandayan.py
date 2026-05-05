# 1. DATASET GRAF JALUR (Adjacency List)

graph_papandayan = {
    'Camp_David': {'Pos_1': 1000, 'Kawah': 1200},
    'Pos_1': {'Camp_David': 1000, 'Pos_2': 800, 'Kawah': 900},
    'Kawah': {'Camp_David': 1200, 'Pos_1': 900, 'Pos_Bayangan': 1500, 'Hutan_Mati': 2000},
    'Pos_2': {'Pos_1': 800, 'Pos_3': 1000, 'Pos_Bayangan': 1100},
    'Pos_3': {'Pos_2': 1000, 'Pos_Bayangan': 700, 'Simpang_Saladah': 500},
    'Pos_Bayangan': {'Pos_2': 1100, 'Kawah': 1500, 'Pos_3': 700, 'Simpang_Saladah': 800},
    'Simpang_Saladah': {'Pos_3': 500, 'Pos_Bayangan': 800, 'Pondok_Saladah': 600},
    'Pondok_Saladah': {'Simpang_Saladah': 600, 'Hutan_Mati': 1000, 'Pos_4': 1200},
    'Hutan_Mati': {'Kawah': 2000, 'Pondok_Saladah': 1000, 'Tanjakan_Mamang': 1400, 'Tegal_Alun': 1800},
    'Pos_4': {'Pondok_Saladah': 1200, 'Tanjakan_Mamang': 900, 'Pos_5': 1500},
    'Tanjakan_Mamang': {'Pos_4': 900, 'Hutan_Mati': 1400, 'Tegal_Alun': 1200, 'Pos_5': 1000},
    'Tegal_Alun': {'Hutan_Mati': 1800, 'Tanjakan_Mamang': 1200, 'Sabana': 1500, 'Puncak': 2000},
    'Pos_5': {'Pos_4': 1500, 'Tanjakan_Mamang': 1000, 'Sabana': 800},
    'Sabana': {'Pos_5': 800, 'Tegal_Alun': 1500, 'Puncak': 1200},
    'Puncak': {'Sabana': 1200, 'Tegal_Alun': 2000}
}

# 2. DATASET HEURISTIK h(n)

heuristic_puncak = {
    'Camp_David': 6500,
    'Pos_1': 6000,
    'Kawah': 5300,
    'Pos_2': 5800,
    'Pos_Bayangan': 5200,
    'Pos_3': 5000,
    'Simpang_Saladah': 4600,
    'Pondok_Saladah': 4200,
    'Hutan_Mati': 3500,
    'Pos_4': 3300,
    'Tanjakan_Mamang': 2600,
    'Tegal_Alun': 1800,
    'Pos_5': 1800,
    'Sabana': 1000,
    'Puncak': 0  # Heuristik tujuan akhir selalu 0
}