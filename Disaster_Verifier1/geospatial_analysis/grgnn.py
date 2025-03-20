import networkx as nx
import numpy as np
from utils import haversine_distance  # Import helper function from utils.py

def construct_geospatial_graph(events, proximity_threshold):
    """
    Construct a graph from a list of events.
    
    Each event is expected to be a dictionary with at least:
      - 'id': unique identifier
      - 'coords': tuple (lat, lon)
      - 'label': disaster type or any verification label
      
    Edges are added between events if the Haversine distance is below the threshold.
    """
    G = nx.Graph()
    # Add nodes
    for event in events:
        G.add_node(event['id'], coords=event['coords'], label=event.get('label', None))
    
    # Add edges based on proximity threshold
    node_ids = list(G.nodes)
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            id1 = node_ids[i]
            id2 = node_ids[j]
            coord1 = G.nodes[id1]['coords']
            coord2 = G.nodes[id2]['coords']
            distance = haversine_distance(coord1, coord2)
            if distance < proximity_threshold:
                # Edge weight is the distance
                G.add_edge(id1, id2, weight=distance)
    return G

def generate_node_embeddings(graph):
    """
    Generate a simple node embedding for each node.
    
    This placeholder uses the node degree and clustering coefficient
    to form a two-dimensional feature vector for each node.
    """
    embeddings = {}
    degrees = dict(graph.degree)
    clustering = nx.clustering(graph)
    
    for node in graph.nodes:
        # Simple 2D embedding: [degree, clustering coefficient]
        embeddings[node] = np.array([degrees[node], clustering[node]])
    return embeddings

def grgnn_predict(graph, new_event, proximity_threshold=50):
    """
    Predict the disaster label for a new event based on the nearest node in the graph.
    
    For a new event, construct a temporary graph node embedding similar to training nodes,
    then compare with the generated node embeddings of the existing graph using Euclidean distance.
    
    new_event: dictionary with 'id', 'coords'
    proximity_threshold: not directly used here, but can help filter distant nodes if needed.
    """
    # Generate embeddings for the current graph
    embeddings = generate_node_embeddings(graph)
    
    # Compute embedding for new event.
    # For simplicity, we compute degree as 0 (new event not connected) and clustering as 0.
    new_embedding = np.array([0, 0])
    
    # Compare new event embedding with each node embedding using Euclidean distance
    min_distance = float('inf')
    best_match_label = None
    for node, emb in embeddings.items():
        euclidean_distance = np.linalg.norm(new_embedding - emb)
        if euclidean_distance < min_distance:
            min_distance = euclidean_distance
            best_match_label = graph.nodes[node].get('label', None)
    return best_match_label

# Example usage:
if __name__ == '__main__':
    # Dummy events for graph construction
    events = [
        {'id': 'E1', 'coords': (40.7128, -74.0060), 'label': 'disaster_A'},
        {'id': 'E2', 'coords': (40.730610, -73.935242), 'label': 'disaster_A'},
        {'id': 'E3', 'coords': (34.0522, -118.2437), 'label': 'disaster_B'}
    ]
    
    # Construct the graph with a proximity threshold of 100 km
    G = construct_geospatial_graph(events, proximity_threshold=100)
    
    # New event to verify
    new_event = {'id': 'New', 'coords': (40.740610, -73.930242)}
    predicted_label = grgnn_predict(G, new_event)
    print("Predicted label using simplified GR-GNN:", predicted_label)