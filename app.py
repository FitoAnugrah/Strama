# -*- coding: utf-8 -*-
"""
Aplikasi Interaktif Pathfinding Gunung Papandayan
Perbandingan Algoritma UCS vs A*
"""

import heapq
import time
import tracemalloc
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import math

# ============================================================
# DATASET
# ============================================================
graph_papandayan = {
    'Camp David':      {'Pos 1': 1000, 'Kawah': 1200},
    'Pos 1':           {'Camp David': 1000, 'Pos 2': 800, 'Kawah': 900},
    'Kawah':           {'Camp David': 1200, 'Pos 1': 900, 'Pos Bayangan': 1500, 'Hutan Mati': 2000},
    'Pos 2':           {'Pos 1': 800, 'Pos 3': 1000, 'Pos Bayangan': 1100},
    'Pos 3':           {'Pos 2': 1000, 'Pos Bayangan': 700, 'Simpang Saladah': 500},
    'Pos Bayangan':    {'Pos 2': 1100, 'Kawah': 1500, 'Pos 3': 700, 'Simpang Saladah': 800},
    'Simpang Saladah': {'Pos 3': 500, 'Pos Bayangan': 800, 'Pondok Saladah': 600},
    'Pondok Saladah':  {'Simpang Saladah': 600, 'Hutan Mati': 1000, 'Pos 4': 1200},
    'Hutan Mati':      {'Kawah': 2000, 'Pondok Saladah': 1000, 'Tanjakan Mamang': 1400, 'Tegal Alun': 1800},
    'Pos 4':           {'Pondok Saladah': 1200, 'Tanjakan Mamang': 900, 'Pos 5': 1500},
    'Tanjakan Mamang': {'Pos 4': 900, 'Hutan Mati': 1400, 'Tegal Alun': 1200, 'Pos 5': 1000},
    'Tegal Alun':      {'Hutan Mati': 1800, 'Tanjakan Mamang': 1200, 'Sabana': 1500, 'Puncak': 2000},
    'Pos 5':           {'Pos 4': 1500, 'Tanjakan Mamang': 1000, 'Sabana': 800},
    'Sabana':          {'Pos 5': 800, 'Tegal Alun': 1500, 'Puncak': 1200},
    'Puncak':          {'Sabana': 1200, 'Tegal Alun': 2000},
}

# Heuristik statis dihapus — digantikan oleh compute_heuristic() yang dinamis.
# Nilai h(n) kini dihitung otomatis berdasarkan goal node yang dipilih user.

ALL_NODES = list(graph_papandayan.keys())

# ============================================================
# ALGORITMA
# ============================================================
def ucs(graph, start, goal):
    queue = [(0, start, [start])]
    visited = set()
    nodes_expanded = 0
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        nodes_expanded += 1
        if node == goal:
            return cost, path, nodes_expanded
        for neighbor, weight in graph.get(node, {}).items():
            if neighbor not in visited:
                heapq.heappush(queue, (cost + weight, neighbor, path + [neighbor]))
    return float('inf'), [], nodes_expanded


def a_star(graph, heuristic, start, goal):
    queue = [(heuristic.get(start, 0), 0, start, [start])]
    visited = set()
    nodes_expanded = 0
    while queue:
        f, g, node, path = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        nodes_expanded += 1
        if node == goal:
            return g, path, nodes_expanded
        for neighbor, weight in graph.get(node, {}).items():
            if neighbor not in visited:
                ng = g + weight
                nf = ng + heuristic.get(neighbor, 0)
                heapq.heappush(queue, (nf, ng, neighbor, path + [neighbor]))
    return float('inf'), [], nodes_expanded


# ============================================================
# HEURISTIK DINAMIS (Dijkstra terbalik dari goal)
# ============================================================
def compute_heuristic(graph, goal):
    """
    Hitung jarak terpendek dari setiap node ke 'goal' menggunakan
    Dijkstra pada graf tak-berarah. Hasilnya adalah heuristik
    admissible & consistent (h* = biaya sesungguhnya) untuk A*.
    """
    dist = {node: float('inf') for node in graph}
    dist[goal] = 0
    queue = [(0, goal)]
    while queue:
        d, u = heapq.heappop(queue)
        if d > dist[u]:
            continue
        for v, w in graph.get(u, {}).items():
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(queue, (nd, v))
    return dist


def run_algorithm(algo_name, start, goal, heuristic):
    tracemalloc.start()
    t0 = time.perf_counter()
    if algo_name == "UCS":
        cost, path, expanded = ucs(graph_papandayan, start, goal)
    else:
        cost, path, expanded = a_star(graph_papandayan, heuristic, start, goal)
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "cost": cost,
        "path": path,
        "expanded": expanded,
        "time_ms": (t1 - t0) * 1000,
        "memory_kb": peak / 1024,
    }


# ============================================================
# VISUALISASI PLOTLY
# ============================================================
# Posisi node disusun secara manual mengikuti topologi jalur
NODE_POS = {
    'Camp David':      (0.0, 0.0),
    'Pos 1':           (1.5, 0.8),
    'Kawah':           (2.5, -0.5),
    'Pos 2':           (3.0, 1.2),
    'Pos Bayangan':    (4.0, 0.0),
    'Pos 3':           (4.5, 1.5),
    'Simpang Saladah': (5.5, 1.2),
    'Pondok Saladah':  (6.5, 0.8),
    'Hutan Mati':      (6.0, -0.8),
    'Pos 4':           (7.5, 0.5),
    'Tanjakan Mamang': (7.0, -1.5),
    'Tegal Alun':      (8.0, -2.5),
    'Pos 5':           (8.5, 0.0),
    'Sabana':          (9.5, -1.0),
    'Puncak':          (10.5, -2.0),
}


def build_plotly_figure(path_ucs, path_astar, heuristic=None, start='Camp David', goal='Puncak'):
    G = nx.Graph()
    for node, neighbors in graph_papandayan.items():
        for nb, w in neighbors.items():
            G.add_edge(node, nb, weight=w)

    pos = NODE_POS

    # --- Edge traces ---
    edge_x, edge_y, edge_labels = [], [], []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    base_edge = go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=1.5, color='rgba(180,200,220,0.35)'),
        hoverinfo='none', showlegend=False
    )

    def path_edge_trace(path, color, name, width=4):
        ex, ey = [], []
        for i in range(len(path) - 1):
            x0, y0 = pos[path[i]]
            x1, y1 = pos[path[i+1]]
            ex += [x0, x1, None]
            ey += [y0, y1, None]
        return go.Scatter(
            x=ex, y=ey, mode='lines',
            line=dict(width=width, color=color),
            name=name, hoverinfo='none'
        )

    traces = [base_edge]
    if path_ucs:
        traces.append(path_edge_trace(path_ucs, '#FF4B4B', 'Rute UCS', width=5))
    if path_astar:
        traces.append(path_edge_trace(path_astar, '#F7B731', 'Rute A*', width=3))

    # Edge weight annotations
    annotations = []
    seen = set()
    for u, v, data in G.edges(data=True):
        key = tuple(sorted([u, v]))
        if key in seen:
            continue
        seen.add(key)
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        annotations.append(dict(
            x=mx, y=my, text=f"{data['weight']}m",
            showarrow=False, font=dict(size=8, color='rgba(200,220,240,0.6)'),
            bgcolor='rgba(15,25,50,0.7)', borderpad=2,
        ))

    # --- Node trace ---
    node_x, node_y, node_text, node_color, node_size, hover = [], [], [], [], [], []

    path_ucs_set  = set(zip(path_ucs[:-1],  path_ucs[1:]))  if path_ucs  else set()
    path_astar_set = set(zip(path_astar[:-1], path_astar[1:])) if path_astar else set()

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

        h_val = heuristic.get(node, '?') if heuristic else '–'
        h_label = f"{h_val:,} m" if isinstance(h_val, (int, float)) and h_val != float('inf') else str(h_val)
        hover.append(
            f"<b>{node}</b><br>"
            f"Heuristik h(n) ke tujuan: {h_label}<br>"
            f"Tetangga: {', '.join(graph_papandayan.get(node, {}).keys())}"
        )

        if node == goal:
            node_color.append('#00E676')
            node_size.append(24)
        elif node == start:
            node_color.append('#40C4FF')
            node_size.append(24)
        elif path_ucs and node in path_ucs and path_astar and node in path_astar:
            node_color.append('#FF4B4B')
            node_size.append(18)
        elif path_ucs and node in path_ucs:
            node_color.append('#FF4B4B')
            node_size.append(16)
        elif path_astar and node in path_astar:
            node_color.append('#F7B731')
            node_size.append(16)
        else:
            node_color.append('#90A4AE')
            node_size.append(12)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        marker=dict(size=node_size, color=node_color,
                    line=dict(width=2, color='rgba(255,255,255,0.3)')),
        text=node_text,
        textposition='top center',
        textfont=dict(size=10, color='white'),
        hovertext=hover, hoverinfo='text',
        name='Lokasi'
    )
    traces.append(node_trace)

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            paper_bgcolor='rgba(10,18,40,0)',
            plot_bgcolor='rgba(10,18,40,0)',
            font=dict(color='white'),
            showlegend=True,
            legend=dict(
                bgcolor='rgba(15,25,60,0.8)',
                bordercolor='rgba(100,150,255,0.3)',
                borderwidth=1,
                font=dict(size=12),
            ),
            hovermode='closest',
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=annotations,
            height=500,
        )
    )
    return fig


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(
    page_title="Papandayan Pathfinder",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a1228 0%, #0d1f45 50%, #0a2a1f 100%);
    color: #e8f0fe;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1a3a 0%, #0a2a1f 100%);
    border-right: 1px solid rgba(100,180,255,0.15);
}
[data-testid="stSidebar"] * { color: #cdd9f5 !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(100,150,255,0.2);
    border-radius: 12px;
    padding: 16px !important;
    backdrop-filter: blur(8px);
    transition: transform 0.2s, box-shadow 0.2s;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(64,196,255,0.15);
}
[data-testid="stMetricLabel"] { color: #90a4ae !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: #e8f0fe !important; font-size: 1.6rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #1565C0, #00897B);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 2rem;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    transition: all 0.25s;
    box-shadow: 0 4px 15px rgba(21,101,192,0.4);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(21,101,192,0.6);
    filter: brightness(1.15);
}

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #64b5f6;
    border-left: 4px solid #1565C0;
    padding-left: 10px;
    margin: 20px 0 12px 0;
    letter-spacing: 0.5px;
}
.algo-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 8px;
}
.ucs-badge  { background: rgba(255,75,75,0.15);  border:1px solid #FF4B4B;  color:#FF4B4B; }
.astar-badge{ background: rgba(247,183,49,0.15); border:1px solid #F7B731; color:#F7B731; }

.route-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(100,150,255,0.15);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.88rem;
    color: #b0bec5;
    margin-top: 4px;
    word-break: break-word;
}
.route-highlight { color: #e8f0fe; font-weight: 600; }

.vis-container {
    background: rgba(10,18,40,0.7);
    border: 1px solid rgba(100,150,255,0.2);
    border-radius: 16px;
    padding: 8px;
    backdrop-filter: blur(12px);
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #40C4FF, #00E676);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 0.95rem;
    color: #78909c;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🏔️ Papandayan Pathfinder")
    st.markdown("""
    <div style='font-size:0.85rem; color:#90a4ae; line-height:1.6; margin-bottom:20px;'>
    Aplikasi perbandingan algoritma pencarian rute terpendek
    <b style='color:#64b5f6;'>UCS</b> dan <b style='color:#F7B731;'>A*</b>
    pada jaringan jalur pendakian Gunung Papandayan, Garut.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**⚙️ Konfigurasi Rute**")

    start_node = st.selectbox(
        "🚩 Titik Awal",
        options=ALL_NODES,
        index=ALL_NODES.index('Camp David'),
        key="start_node",
    )

    goal_node = st.selectbox(
        "🏁 Titik Tujuan",
        options=ALL_NODES,
        index=ALL_NODES.index('Puncak'),
        key="goal_node",
    )

    st.markdown("---")
    run_btn = st.button("🔍 Cari Rute Terpendek", use_container_width=True)

    # --- Info heuristik dinamis ---
    st.markdown("---")
    with st.expander("📐 Heuristik h(n) Dinamis", expanded=False):
        preview_h = compute_heuristic(graph_papandayan, goal_node)
        st.markdown(
            f"<div style='font-size:0.75rem;color:#90a4ae;margin-bottom:6px;'>"
            f"Nilai h(n) setiap node menuju <b style='color:#64b5f6;'>{goal_node}</b>"
            f" (Dijkstra terbalik):</div>",
            unsafe_allow_html=True,
        )
        for node, val in preview_h.items():
            val_str = f"{val:,} m" if val != float('inf') else "∞"
            marker = "🏁" if node == goal_node else "📍"
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"font-size:0.76rem;color:#cdd9f5;margin:2px 0;'>"
                f"<span>{marker} {node}</span>"
                f"<b style='color:#40C4FF;'>{val_str}</b></div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.75rem; color:#546e7a;'>
    <b>Legenda Warna Graf:</b><br>
    🔵 {start_node} &nbsp; 🟢 {goal_node}<br>
    🔴 Jalur UCS &nbsp; 🟡 Jalur A*<br>
    ⚫ Node biasa
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MAIN CONTENT
# ============================================================
st.markdown('<div class="hero-title">🗺️ Papandayan Route Finder</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Perbandingan Algoritma Uniform Cost Search (UCS) vs A* Search · Jalur Pendakian Gunung Papandayan</div>', unsafe_allow_html=True)

if not run_btn:
    # Default state: show full graph without highlighted path
    st.markdown('<div class="section-header">📡 Peta Jaringan Jalur</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="vis-container">', unsafe_allow_html=True)
        fig_default = build_plotly_figure([], [], compute_heuristic(graph_papandayan, goal_node), start=start_node, goal=goal_node)
        st.plotly_chart(fig_default, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    st.info("👈 Pilih titik awal & tujuan di sidebar, lalu tekan **Cari Rute Terpendek** untuk memulai analisis.")

else:
    if start_node == goal_node:
        st.error("⚠️ Titik awal dan tujuan tidak boleh sama. Pilih lokasi yang berbeda.")
        st.stop()

    # Hitung heuristik dinamis berdasarkan goal yang dipilih
    dyn_heuristic = compute_heuristic(graph_papandayan, goal_node)

    # Run both algorithms
    with st.spinner("Menjalankan algoritma UCS dan A*..."):
        res_ucs   = run_algorithm("UCS", start_node, goal_node, dyn_heuristic)
        res_astar = run_algorithm("A*",  start_node, goal_node, dyn_heuristic)

    # ---- Section: Metrics ----
    st.markdown('<div class="section-header">📊 Perbandingan Performa Algoritma</div>', unsafe_allow_html=True)

    col_ucs, col_astar = st.columns(2)

    with col_ucs:
        st.markdown('<span class="algo-badge ucs-badge">🔴 UCS — Uniform Cost Search</span>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📏 Jarak", f"{res_ucs['cost']:,} m")
        c2.metric("⏱️ Waktu", f"{res_ucs['time_ms']:.3f} ms")
        c3.metric("💾 Memori", f"{res_ucs['memory_kb']:.2f} KB")
        c4.metric("🔁 Nodes", f"{res_ucs['expanded']}")

        ucs_route = " → ".join(res_ucs["path"]) if res_ucs["path"] else "Rute tidak ditemukan"
        st.markdown(f'<div class="route-box"><b class="route-highlight">Rute:</b> {ucs_route}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with col_astar:
        st.markdown('<span class="algo-badge astar-badge">🟡 A* — A-Star Search</span>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)

        # delta negatif = A* lebih hemat (lebih baik); delta_inverse membalik tanda
        # agar Streamlit menampilkan HIJAU saat A* lebih baik (lebih kecil)
        delta_cost = res_astar['cost'] - res_ucs['cost']
        delta_t    = res_astar['time_ms']    - res_ucs['time_ms']
        delta_m    = res_astar['memory_kb']  - res_ucs['memory_kb']
        delta_n    = res_astar['expanded']   - res_ucs['expanded']

        c1.metric("📏 Jarak",  f"{res_astar['cost']:,} m",
                  delta=f"{delta_cost:+,} m" if res_ucs['cost'] != float('inf') else None,
                  delta_color="inverse")
        c2.metric("⏱️ Waktu",  f"{res_astar['time_ms']:.3f} ms",
                  delta=f"{delta_t:+.3f} ms", delta_color="inverse")
        c3.metric("💾 Memori", f"{res_astar['memory_kb']:.2f} KB",
                  delta=f"{delta_m:+.2f} KB", delta_color="inverse")
        c4.metric("🔁 Nodes",  f"{res_astar['expanded']}",
                  delta=f"{delta_n:+d}", delta_color="inverse")

        astar_route = " → ".join(res_astar["path"]) if res_astar["path"] else "Rute tidak ditemukan"
        st.markdown(f'<div class="route-box"><b class="route-highlight">Rute:</b> {astar_route}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Section: Visualisasi ----
    st.markdown('<div class="section-header">📡 Visualisasi Graf Interaktif</div>', unsafe_allow_html=True)
    st.caption("💡 Hover node untuk info · Scroll untuk zoom · Drag untuk pan")

    with st.container():
        st.markdown('<div class="vis-container">', unsafe_allow_html=True)
        fig = build_plotly_figure(res_ucs["path"], res_astar["path"], dyn_heuristic, start=start_node, goal=goal_node)
        st.plotly_chart(fig, use_container_width=True, config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
            "displaylogo": False,
        })
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Section: Insight ----
    st.markdown('<div class="section-header">💡 Analisis & Insight</div>', unsafe_allow_html=True)

    same_path   = res_ucs["path"] == res_astar["path"]
    faster      = "A*" if res_astar['time_ms']   < res_ucs['time_ms']   else "UCS"
    efficient   = "A*" if res_astar['expanded']  <= res_ucs['expanded'] else "UCS"
    mem_winner  = "A*" if res_astar['memory_kb'] <= res_ucs['memory_kb'] else "UCS"
    fc = "#F7B731"; uc = "#FF4B4B"

    i1, i2, i3, i4 = st.columns(4)
    card = "background:rgba(255,255,255,0.04);border:1px solid rgba(100,150,255,0.2);border-radius:12px;padding:16px;font-size:0.84rem;height:100%;"

    with i1:
        icon = "✅" if same_path else "⚠️"
        txt  = "Rute <b>sama</b>" if same_path else "Rute <b>berbeda</b>"
        st.markdown(f"<div style='{card}'><b style='color:#64b5f6;'>🛤️ Konsistensi Rute</b><br><br>{icon} Kedua algoritma {txt}.</div>", unsafe_allow_html=True)

    with i2:
        col = fc if faster == "A*" else uc
        st.markdown(f"<div style='{card}'><b style='color:#64b5f6;'>⚡ Kecepatan</b><br><br>"
                    f"<b style='color:{col};'>{faster}</b> lebih cepat<br>"
                    f"selisih <b>{abs(delta_t):.3f} ms</b>.</div>", unsafe_allow_html=True)

    with i3:
        col = fc if efficient == "A*" else uc
        st.markdown(f"<div style='{card}'><b style='color:#64b5f6;'>🔁 Ekspansi Node</b><br><br>"
                    f"<b style='color:{col};'>{efficient}</b> lebih efisien<br>"
                    f"(<b>{min(res_ucs['expanded'],res_astar['expanded'])}</b> vs "
                    f"<b>{max(res_ucs['expanded'],res_astar['expanded'])}</b> node).</div>", unsafe_allow_html=True)

    with i4:
        col = fc if mem_winner == "A*" else uc
        st.markdown(f"<div style='{card}'><b style='color:#64b5f6;'>💾 Penggunaan Memori</b><br><br>"
                    f"<b style='color:{col};'>{mem_winner}</b> lebih hemat memori<br>"
                    f"(<b>{min(res_ucs['memory_kb'],res_astar['memory_kb']):.2f}</b> vs "
                    f"<b>{max(res_ucs['memory_kb'],res_astar['memory_kb']):.2f}</b> KB).</div>", unsafe_allow_html=True)

    # ---- Tabel Ringkasan ----
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📋 Tabel Ringkasan Perbandingan</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <table style='width:100%;border-collapse:collapse;font-size:0.88rem;color:#cdd9f5;'>
      <thead>
        <tr style='border-bottom:1px solid rgba(100,150,255,0.3);'>
          <th style='text-align:left;padding:10px 14px;color:#64b5f6;'>Metrik</th>
          <th style='text-align:center;padding:10px 14px;color:#FF4B4B;'>🔴 UCS</th>
          <th style='text-align:center;padding:10px 14px;color:#F7B731;'>🟡 A*</th>
          <th style='text-align:center;padding:10px 14px;color:#90a4ae;'>Selisih</th>
        </tr>
      </thead>
      <tbody>
        <tr style='border-bottom:1px solid rgba(100,150,255,0.1);background:rgba(255,255,255,0.02);'>
          <td style='padding:9px 14px;'>📏 Jarak (m)</td>
          <td style='text-align:center;padding:9px 14px;'>{res_ucs['cost']:,}</td>
          <td style='text-align:center;padding:9px 14px;'>{res_astar['cost']:,}</td>
          <td style='text-align:center;padding:9px 14px;color:{"#00E676" if delta_cost==0 else ("#FF4B4B" if delta_cost>0 else "#00E676")};'>{delta_cost:+,}</td>
        </tr>
        <tr style='border-bottom:1px solid rgba(100,150,255,0.1);'>
          <td style='padding:9px 14px;'>⏱️ Waktu (ms)</td>
          <td style='text-align:center;padding:9px 14px;'>{res_ucs['time_ms']:.4f}</td>
          <td style='text-align:center;padding:9px 14px;'>{res_astar['time_ms']:.4f}</td>
          <td style='text-align:center;padding:9px 14px;color:{"#00E676" if delta_t<0 else "#FF4B4B"};'>{delta_t:+.4f}</td>
        </tr>
        <tr style='border-bottom:1px solid rgba(100,150,255,0.1);background:rgba(255,255,255,0.02);'>
          <td style='padding:9px 14px;'>💾 Memori (KB)</td>
          <td style='text-align:center;padding:9px 14px;'>{res_ucs['memory_kb']:.2f}</td>
          <td style='text-align:center;padding:9px 14px;'>{res_astar['memory_kb']:.2f}</td>
          <td style='text-align:center;padding:9px 14px;color:{"#00E676" if delta_m<0 else "#FF4B4B"};'>{delta_m:+.2f}</td>
        </tr>
        <tr>
          <td style='padding:9px 14px;'>🔁 Node Diekspansi</td>
          <td style='text-align:center;padding:9px 14px;'>{res_ucs['expanded']}</td>
          <td style='text-align:center;padding:9px 14px;'>{res_astar['expanded']}</td>
          <td style='text-align:center;padding:9px 14px;color:{"#00E676" if delta_n<0 else ("#90a4ae" if delta_n==0 else "#FF4B4B")};'>{delta_n:+d}</td>
        </tr>
      </tbody>
    </table>
    <p style='font-size:0.72rem;color:#546e7a;margin-top:6px;'>* Selisih dihitung dari A* − UCS. Hijau = A* lebih baik, Merah = UCS lebih baik.</p>
    """, unsafe_allow_html=True)
