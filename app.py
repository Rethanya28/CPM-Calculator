from flask import Flask, render_template, request, redirect, url_for,  jsonify
import networkx as nx
from pyvis.network import Network
from collections import deque
import pandas as pd
import io

app = Flask(__name__)

G = nx.DiGraph()

def calculate_duration(optimistic, mostlikely, pessimistic):
    return round((optimistic + 4 * mostlikely + pessimistic) / 6)

def update_graph_from_excel(file):
    global G
    G.clear()
    
    df = pd.read_excel(file)
    
    # Identify nodes without precedents
    no_precedent_nodes = set(df['Activity']) - set(df['Precedent'].dropna().unique())

    dummy_node_count = 0

    for _, row in df.iterrows():
        activity = row['Activity']
        precedent = row['Precedent']
        optimistic = row['Optimistic']
        mostlikely = row['Most Likely']
        pessimistic = row['Pessimistic']
        duration = calculate_duration(optimistic, mostlikely, pessimistic)
        
        # Add the activity node
        G.add_node(activity, duration=duration)
        
        if pd.notna(precedent):
            precedents = str(precedent).split(',')
            for pre in precedents:
                pre = pre.strip()
                if pre:
                    G.add_edge(pre, activity, weight=duration)
        else:
            # For nodes with no precedents, add dummy nodes
            dummy_node = f'S'
            dummy_node_count += 1
            G.add_node(dummy_node, duration=0)
            G.add_edge(dummy_node, activity, weight=0)
    
    # Add the end node if necessary
    hanging_nodes = [node for node in G.nodes if G.out_degree(node) == 0]
    for i, node in enumerate(hanging_nodes):
        end_node = f'End'
        G.add_edge(node, end_node, weight=0)

def forward_pass(G, start):
    earliest_start = {node: 0 for node in G.nodes}
    earliest_finish = {node: 0 for node in G.nodes}

    for node in nx.topological_sort(G):
        for pred in G.predecessors(node):
            earliest_start[node] = max(earliest_start[node], earliest_finish[pred])
        earliest_finish[node] = earliest_start[node] + G.nodes[node].get('duration', 0)

    return earliest_start, earliest_finish

def backward_pass(G, earliest_finish):
    latest_finish = {node: float('inf') for node in G.nodes}
    latest_start = {node: float('inf') for node in G.nodes}

    finish_time = max(earliest_finish.values())
    for node in G.nodes:
        if G.out_degree(node) == 0:
            latest_finish[node] = finish_time

    for node in reversed(list(nx.topological_sort(G))):
        for succ in G.successors(node):
            latest_finish[node] = min(latest_finish[node], latest_start[succ])
        latest_start[node] = latest_finish[node] - G.nodes[node].get('duration', 0)

    return latest_start, latest_finish

def identify_critical_path(G, earliest_start, latest_start):
    critical_path = []
    for node in G.nodes:
        if earliest_start[node] == latest_start[node]:
            critical_path.append(node)
    return critical_path

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify(success=False), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(success=False), 400

    if file and file.filename.endswith('.xlsx'):
        update_graph_from_excel(file)
        return jsonify(success=True), 200
    else:
        return jsonify(success=False), 400


@app.route('/forward_pass')
def forward():
    global earliest_start, earliest_finish
    earliest_start, earliest_finish = forward_pass(G, 'S')
    generate_network(G, earliest_start, earliest_finish)
    return render_template('graph.html')

@app.route('/backward_pass')
def backward():
    global earliest_start, earliest_finish, latest_start, latest_finish
    earliest_start, earliest_finish = forward_pass(G, 'S')
    latest_start, latest_finish = backward_pass(G, earliest_finish)
    generate_network(G, earliest_start, earliest_finish, latest_start, latest_finish)
    return render_template('graph.html')

@app.route('/critical_path')
def critical():
    global earliest_start, earliest_finish, latest_start, latest_finish, critical_path
    earliest_start, earliest_finish = forward_pass(G, 'S')
    latest_start, latest_finish = backward_pass(G, earliest_finish)
    critical_path = identify_critical_path(G, earliest_start, latest_start)
    generate_network(G, earliest_start, earliest_finish, latest_start, latest_finish, critical_path)
    return render_template('graph.html')

@app.route('/displaygraph')
def display_graph():
    show()
    return render_template('graph.html')

def bfs_layout(G, start):
    # Perform BFS traversal
    levels = {}
    queue = deque([(start, 0)])  # (node, level)
    visited = set()
    while queue:
        node, level = queue.popleft()
        if node not in visited:
            visited.add(node)
            levels[node] = level
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    queue.append((neighbor, level + 1))
  # Determine positions
    pos = {}
    level_nodes = {}
    for node, level in levels.items():
        if level not in level_nodes:
            level_nodes[level] = []
        level_nodes[level].append(node)
    max_width = max(len(nodes) for nodes in level_nodes.values())
    for level, nodes in level_nodes.items():
        width = len(nodes)
        for i, node in enumerate(nodes):
            pos[node] = (i - width / 2, -level)
    return pos

def generate_network(G, earliest_start, earliest_finish, latest_start=None, latest_finish=None, critical_path=None):
    pos = bfs_layout(G, 'S')
    net = Network(directed=True)

    # Add nodes with customized colors and labels
    for node, (x, y) in pos.items():
        est = earliest_start.get(node, '-')
        lst = latest_start.get(node, '-') if latest_start else '-'
        eft = earliest_finish.get(node, '-')
        lft = latest_finish.get(node, '-') if latest_finish else '-'
        
        # Calculate float as the difference between latest start and earliest start
        float_time = lst - est if lst != '-' and est != '-' else '-'
        
        label = (f"{node}\n"
             f"EStart: {est}\nEFinish: {eft}\n"
             f"LStart: {lst}\nLFinish: {lft}\n"
             f"Float: {float_time}")

        
        color = 'red' if critical_path and node in critical_path else 'blue'
        net.add_node(node, label=label, x=x*170, y=y*170, color=color)


    # Add edges with black color
    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, label=data['weight'], color='black')

    # Get the finish time of the "End" node
    end_finish_time = earliest_finish.get('End', 'Unknown')

    # Set options for the network
    net.set_options("""
    var options = {
        "nodes": {
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "shape": "box",
            "font": {
                "size": 16,
                "face": "arial",
                "color": "black"
            }
        },
        "edges": {
            "color": {
                "inherit": false
            },
            "smooth": {
                "type": "cubicBezier",
                "forceDirection": "horizontal",
                "roundness": 0.5
            }
        },
        "physics": {
            "enabled": false
        }
    }
""")


    # Write the initial HTML
    net.write_html("templates/graph.html")
    add_finish_time_to_html(end_finish_time)
    add_css_to_html()

def add_finish_time_to_html(end_finish_time):
    # Read the generated HTML
    with open("templates/graph.html", "r") as file:
        html_content = file.read()

    # Define the Finish Time display
    finish_time_html = f"""
    <div style="text-align: center; margin-top: 10px;">
        <strong>Finish Time of End Node: {end_finish_time}</strong>
    </div>
    """

    # Find the insertion point in the HTML content (before the closing </body> tag)
    insertion_point = html_content.find('</body>')
    if insertion_point != -1:
        # Insert Finish Time display before the closing </body> tag
        html_content = html_content[:insertion_point] + finish_time_html + html_content[insertion_point:]

    # Write the modified HTML back to the file
    with open("templates/graph.html", "w") as file:
        file.write(html_content)

def add_css_to_html():
    # Read the generated HTML
    with open("templates/graph.html", "r") as file:
        html_content = file.read()

    # Define the CSS to be added
    css = """
    <style>
        body {
            font-family: Arial, sans-serif;
        }
        #mynetwork {
            width: 100%;
            height: 700px;
        }
    </style>
    """

    # Find the insertion point in the HTML content (after <head>)
    insertion_point = html_content.find('</head>')
    if insertion_point != -1:
        # Insert CSS before the closing </head> tag
        html_content = html_content[:insertion_point] + css + html_content[insertion_point:]

    # Write the modified HTML back to the file
    with open("templates/graph.html", "w") as file:
        file.write(html_content)


def show():
    pos = bfs_layout(G, 'S')
    net = Network(directed=True)

    # Add nodes with customized colors
    for node, (x, y) in pos.items():
        label = (f"{node}")
        color = 'blue'
        net.add_node(node, label=label, x=x*150, y=y*150, color=color)

    # Add edges with black color
    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, label=data['weight'], color='black')

    net.set_options("""
        var options = {
            "nodes": {
                "borderWidth": 2,
                "borderWidthSelected": 4,
                "shape": "box",
                "font": {
                    "size": 16,
                    "face": "arial",
                    "color": "black"
                }
            },
            "edges": {
                "color": {
                    "inherit": false
                },
                "smooth": {
                    "type": "cubicBezier",
                    "forceDirection": "horizontal",
                    "roundness": 0.5
                }
            },
            "physics": {
                "enabled": false
            }
        }
    """)


    net.write_html("templates/graph.html")

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
