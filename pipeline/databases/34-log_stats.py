import pymongo

# Connect to the MongoDB server (assuming it's running locally)
client = pymongo.MongoClient("localhost", 27017)

# Access the 'logs' database
db = client.logs

# Access the 'nginx' collection
collection = db.nginx

# Get the total number of documents in the collection
total_logs = collection.count_documents({})

# Get the count for each HTTP method
methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
method_counts = {method: collection.count_documents({"method": method}) for method in methods}

# Get the count for the specific method and path
status_check_count = collection.count_documents({"method": "GET", "path": "/status"})

# Print the results
print(f"{total_logs} logs")
print("Methods:")
for method in methods:
    print(f"    method {method}: {method_counts[method]}")
print(f"{status_check_count} status check")

# Close the MongoDB connection
client.close()
