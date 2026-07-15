import weaviate

client = weaviate.connect_to_local(host="127.0.0.1",port=8080,grpc_port=50051,)
print("Weaviate ready:", client.is_ready())
print("Weaviate live:", client.is_live())
client.close()