import ndex2
import pandas as pd
import networkx as nx
import ndex2.client
from ndex2.cx2 import CX2Network, RawCX2NetworkFactory, NetworkXToCX2NetworkFactory, PandasDataFrameToCX2NetworkFactory, CX2NetworkPandasDataFrameFactory, CX2NetworkXFactory 
import json

# 1. Initialize an anonymous connection to the public NDEx server
client = ndex2.client.Ndex2("http://public.ndexbio.org")

# 2. Search for networks matching a keyword
# Returns a dictionary containing a list of network summaries
search_results = client.search_networks(
    search_string='Alzheimers nodeCount:[10 TO 55]', 
    start=0,             # Number of initial records to skip (useful for pagination)
    size=100              # Maximum number of network results to return
)

# 3. Parse and print network metadata from results
print(f"Found {len(search_results.get('networks', []))} networks:\n")

# for network in search_results.get('networks', []):
#     name = network.get('name')
#     uuid = network.get('externalId')
#     node_count = network.get('nodeCount')
#     edge_count = network.get('edgeCount')
    
#     print(f"Name: {name}")
#     print(f"UUID: {uuid}")
#     print(f"Size: {node_count} nodes, {edge_count} edges")
#     print("-" * 40)

networks = pd.DataFrame(search_results.get('networks', []))

print(networks.head(10))

networks.to_csv("AD_small_networks")


# # 1. Download the legacy NiceCX network representation
# network_uuid = "95ec4c84-5c6b-11ec-b3be-0ac135e8bacf"
# nice_cx = ndex2.create_nice_cx_from_server(server="public.ndexbio.org", uuid=network_uuid)

# # 2. Convert to a NetworkX Graph object
# graph = nice_cx.to_networkx()

# # 3. Flatten the networkx node attributes dictionary directly into a DataFrame
# nodes_df = pd.DataFrame([
#     {"id": node_id, **attributes} 
#     for node_id, attributes in graph.nodes(data=True)
# ])




# 1. Download the network as a CX2 object directly from the server
network_uuid = "6b40f2d4-deef-11ea-99da-0ac135e8bacf"
client = ndex2.client.Ndex2("http://ndexbio.org")
client_resp = client.get_network_as_cx2_stream(network_uuid)
# Create CX2Network factory
factory = RawCX2NetworkFactory()
# Convert downloaded network to CX2Network object
cx2_network = factory.get_cx2network(json.loads(client_resp.content))

# 2. Instantiate the second factory to convert CX2 elements into DataFrames
pandas_factory = CX2NetworkPandasDataFrameFactory()

# 3. Extract the node list table into a pandas DataFrame
nodes_df = pandas_factory.get_dataframe(cx2_network)

# View the structure
#print(nodes_df.head())

#print(cx2_network.get_nodes())

for id, node in cx2_network.get_nodes().items():
    properties = node['v']
    name = properties['name']
    print(name)